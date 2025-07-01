import os
import tensorflow as tf
from keras import Model, Input
from keras.layers import LSTM, Dense, concatenate, Bidirectional
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam
from src.models.base_model import BaseModel
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def quantile_loss(q):
    """
    Pinball loss function for quantile q.
    This loss function is used to train a model to predict a specific quantile.
    The error (e) is the difference between the true value (y_true) and the prediction (y_pred).
    The loss is calculated as a weighted average of the errors, where the weights depend on the quantile 'q'.
    """
    def loss(y_true, y_pred):
        e = y_true - y_pred
        return tf.reduce_mean(tf.maximum(q * e, (q - 1) * e))
    return loss


class LSTMQuantileModel(BaseModel):
    """
    Multi-output LSTM for quantile regression.
    Outputs: lower (q_low), median (0.5), upper (q_up).
    This model is designed to predict an interval rather than a single point estimate.
    It takes both sequential and static data as input.
    """
    def __init__(self, params: dict):
        """
        Initializes the model with given parameters.
        :param params: A dictionary of parameters for building and training the model.
                       Expected keys include 'quantiles', 'input_shape_seq', 'input_shape_stat', etc.
        """
        super().__init__(params)
        # Quantiles are expected in the params dictionary
        self.quantiles = params.get('quantiles', [0.05, 0.5, 0.95])
        self.build_model()

    def build_model(self):
        """
        Builds the Keras model architecture.
        The model has two input branches that are later concatenated.
        """
        p = self.params

        # LSTM sequence branch
        input_seq = Input(shape=p['input_shape_seq'], name='seq_input')
        
        # A Bidirectional LSTM processes the sequence, capturing dependencies from both past and future contexts.
        x_seq = Bidirectional(LSTM(p.get('lstm_units', 64), return_sequences=False))(input_seq)

        # Static features branch
        input_stat = Input(shape=(p['input_shape_stat'],), name='static_input')

        # Fusion of the two branches
        x = concatenate([x_seq, input_stat])
        x = Dense(p.get('dense_units', 128), activation='relu')(x)

        # Output heads for each quantile
        # Each head is a Dense layer that outputs a single value for its respective quantile.
        lower = Dense(1, activation='linear', name='lower')(x)
        median = Dense(1, activation='linear', name='median')(x)
        upper = Dense(1, activation='linear', name='upper')(x)

        self.model = Model(inputs=[input_seq, input_stat], outputs=[lower, median, upper])

        # Compilation with pinball loss for each output
        losses = {
            'lower': quantile_loss(self.quantiles[0]),
            'median': quantile_loss(self.quantiles[1]),
            'upper': quantile_loss(self.quantiles[2]),
        }
        self.model.compile(
            optimizer=p.get('optimizer', Adam(learning_rate=0.001)),
            loss=losses,
            metrics={'median': ['mae']}  # Monitor Mean Absolute Error for the median prediction
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Trains the model on the provided data.
        Handles target variable scaling and prepares data for multi-output training.
        """
        p = self.params
        # Manages y scaling as in the BaseModel
        if p.get('scale_mode') == 'standard':
            self.y_scaler = StandardScaler()
            y_train = self.y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
            if y_val is not None:
                y_val = self.y_scaler.transform(y_val.reshape(-1, 1)).ravel()
        elif p.get('scale_mode') == 'minmax':
            self.y_scaler = MinMaxScaler()
            y_train = self.y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
            if y_val is not None:
                y_val = self.y_scaler.transform(y_val.reshape(-1, 1)).ravel()
        elif p.get('scale_mode') == 'divide':
            self.scale_factor = p.get('scale_factor', 1)
            y_train = y_train / self.scale_factor
            if y_val is not None:
                y_val = y_val / self.scale_factor


        # Duplicate the target for the 3 outputs, as they all learn from the same true value.
        y_tr = [y_train, y_train, y_train]
        if X_val is not None and y_val is not None:
            y_va = [y_val, y_val, y_val]
            val_data = (X_val, y_va)
        else:
            val_data = None

        # Callbacks for training
        callbacks = []
        if p.get('checkpoint_path'):
            os.makedirs(os.path.dirname(p['checkpoint_path']), exist_ok=True)
            callbacks.append(ModelCheckpoint(
                filepath=p['checkpoint_path'], monitor='val_loss', save_best_only=True, verbose=1
            ))
        callbacks.append(ReduceLROnPlateau(
            monitor='val_loss', factor=0.6, patience=3, min_lr=1e-6, verbose=1
        ))
        if p.get('early_stop_patience'):
            callbacks.append(EarlyStopping(
                monitor='val_loss', patience=p['early_stop_patience'], restore_best_weights=True, verbose=1
            ))

        return self.model.fit(
            X_train, y_tr,
            validation_data=val_data,
            epochs=p.get('epochs', 100),
            batch_size=p.get('batch_size', 32),
            callbacks=callbacks,
            verbose=p.get('verbose', 1)
        )

    def predict(self, X):
        """
        Makes predictions with the trained model.
        Returns a dictionary of numpy arrays (n,).
        """
        preds = self.model.predict(X, verbose=1)
        lower, median, upper = preds
        
        # Inverse transform if y was scaled during training
        if hasattr(self, 'y_scaler'):
            lower = self.y_scaler.inverse_transform(lower).ravel()
            median = self.y_scaler.inverse_transform(median).ravel()
            upper = self.y_scaler.inverse_transform(upper).ravel()
        elif hasattr(self, 'scale_factor'):
            lower = lower.ravel() * self.scale_factor
            median = median.ravel() * self.scale_factor
            upper = upper.ravel() * self.scale_factor
        else:
            lower = lower.ravel()
            median = median.ravel()
            upper = upper.ravel()
        return {'lower': lower, 'median': median, 'upper': upper}