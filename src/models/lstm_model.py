import os
from keras import Model, Input
from keras.layers import LSTM, Dense, concatenate, Bidirectional
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.models import load_model
from keras.optimizers import Adam
from src.models.base_model import BaseModel
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class LSTMModel(BaseModel):
    """
    Implements an LSTM model for sequence regression that also accepts static features.
    Dual-input architecture: LSTM branch for sequence and dense branch for static features.
    """
    def __init__(self, params: dict):
        super().__init__(params)
        self.build_model()

    def build_model(self):
        p = self.params

        # Sequence Input Branch (LSTM)
        input_seq = Input(shape=p['input_shape_seq'], name='seq_input')
        x_seq = Bidirectional(LSTM(p.get('lstm_units', 64), return_sequences=False))(input_seq)

        
        # Static Features Input Branch
        input_stat = Input(shape=(p['input_shape_stat'],), name='static_input')

        # Merge Branches
        combined = concatenate([x_seq, input_stat])

        # Final Prediction Head
        x = Dense(p.get('dense_units', 128), activation='relu')(combined)
        out = Dense(1, activation='linear')(x)

        # Instantiate and Compile Model
        self.model = Model(inputs=[input_seq, input_stat], outputs=out)
        optimizer = p.get('optimizer', Adam(learning_rate=0.001))
        self.model.compile(
            optimizer=optimizer,
            loss=p.get('loss', 'mse'),
            metrics=p.get('metrics', ['mae'])
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        p = self.params
        callbacks = []

        self.scale_mode = p.get('scale_mode', None)

        if self.scale_mode == 'standard':
            self.y_scaler = StandardScaler()
            y_train = self.y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
            if y_val is not None:
                y_val = self.y_scaler.transform(y_val.reshape(-1, 1)).ravel()

        elif self.scale_mode == 'divide':
            self.scale_factor = 60000
            y_train = y_train / self.scale_factor
            if y_val is not None:
                y_val = y_val / self.scale_factor
                
        elif self.scale_mode == 'minmax':
            self.y_scaler = MinMaxScaler()
            y_train = self.y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
            if y_val is not None:
                y_val = self.y_scaler.transform(y_val.reshape(-1, 1)).ravel()
            

        # Checkpoint callback
        if p.get('checkpoint_path'):
            os.makedirs(os.path.dirname(p['checkpoint_path']), exist_ok=True)
            callbacks.append(
                ModelCheckpoint(
                    filepath=p['checkpoint_path'],
                    monitor='val_loss',
                    save_best_only=True,
                    save_weights_only=False,
                    verbose=1
                )
            )

        # Reduce LR on plateau
        callbacks.append(
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.6,
                patience=3,
                min_lr=1e-6,
                verbose=1
            )
        )

        # Early stopping
        if p.get('early_stop_patience'):
            callbacks.append(
                EarlyStopping(
                    monitor='val_loss',
                    patience=p['early_stop_patience'],
                    restore_best_weights=True,
                    verbose=1
                )
            )

        # Prepare validation data
        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None

        return self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=p.get('epochs', 100),
            batch_size=p.get('batch_size', 32),
            callbacks=callbacks,
            verbose=p.get('verbose', 1)
        )

    def predict(self, X):
        preds = self.model.predict(X, verbose=1)
        if getattr(self, 'scale_mode', None) == 'standard':
            preds = self.y_scaler.inverse_transform(preds.reshape(-1, 1)).ravel()
        elif getattr(self, 'scale_mode', None) == 'divide':
            preds = preds * self.scale_factor
            
        elif getattr(self, 'scale_mode', None) == 'minmax':
            preds = self.y_scaler.inverse_transform(preds.reshape(-1, 1)).ravel()
            
        return preds
    

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)

    @classmethod
    def load(cls, path: str):
        model = load_model(path)
        inst = cls.__new__(cls)
        inst.params = {}
        inst.model = model
        return inst
