import os
from keras import Model, Input
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, concatenate
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.models import load_model
from keras.optimizers import Adam
from src.models.base_model import BaseModel
from sklearn.preprocessing import StandardScaler
import numpy as np 


class CNNModel(BaseModel):
    """
    Implements a 1D-CNN model for sequence regression that also accepts static features.

    This model uses a dual-input architecture:
    1. A convolutional branch to process time-series (lag) features.
    2. A dense branch for static, non-sequential features.
    The outputs of both branches are concatenated before the final prediction layers.
    """

    def __init__(self, params: dict):
        """
        Initializes the model with a dictionary of parameters.

        Args:
            params (dict): A dictionary containing model parameters, including:
              - input_shape_seq (tuple): Shape of the sequence input, e.g., (n_timesteps, 1).
              - input_shape_stat (int): Number of static features.
              - filters (int): Number of filters for the Conv1D layer.
              - kernel_size (int): Kernel size for the Conv1D layer.
              - pool_size (int): Pool size for the MaxPooling1D layer.
              - dense_units (int): Number of neurons in the dense hidden layer.
              - optimizer (keras.Optimizer): A Keras optimizer instance.
              - epochs (int), batch_size (int), verbose (int).
        """
        super().__init__(params)
        self.build_model()

    def build_model(self):
        """
        Builds the dual-input Keras model using the functional API.
        """
        p = self.params

        # Sequence Input Branch (CNN)
        input_seq = Input(shape=p['input_shape_seq'], name='seq_input')
        x_seq = Conv1D(p['filters'], p['kernel_size'], activation='relu')(input_seq)
        x_seq = MaxPooling1D(p['pool_size'])(x_seq)
        x_seq = Flatten()(x_seq)

        # Static Features Input Branch 
        input_stat = Input(shape=(p['input_shape_stat'],), name='static_input')

        # Merge Branches 
        combined = concatenate([x_seq, input_stat])

        # Final Prediction Head 
        x = Dense(p['dense_units'], activation='relu')(combined)
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
        """
        Trains the model. For multi-input models, X_train and X_val should be
        dictionaries mapping input names to numpy arrays.
        """
        p = self.params
        callbacks = []
        
        if self.params.get('scale_y', False):
            self.y_scaler = StandardScaler()
            y_train = self.y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
            if y_val is not None:
                y_val = self.y_scaler.transform(y_val.reshape(-1, 1)).ravel()
                
                
        # Add a callback to save the best model checkpoint
        if p.get('checkpoint_path'):
            os.makedirs(os.path.dirname(p['checkpoint_path']), exist_ok=True)
            callbacks.append(
                ModelCheckpoint(
                    filepath=p['checkpoint_path'],
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=1
                )
            )
        
        # Add a callback to reduce learning rate on plateau
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.6,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
        callbacks.append(reduce_lr)

        # Format validation data if provided
        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None

        
        
        if p.get('early_stop_patience'):
            callbacks.append(
                EarlyStopping(
                monitor='val_loss',
                patience=p['early_stop_patience'],
                restore_best_weights=True,
                verbose=1
            )
        )
        return self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=p['epochs'],
            batch_size=p['batch_size'],
            callbacks=callbacks,
            verbose=p.get('verbose', 1)
        ) 


    def predict(self, X):
        """
        Generates predictions for the given input data X.
        """
        preds = self.model.predict(X, verbose=0)
        # Inverse transform the target if it was scaled during training
        if self.params.get('scale_y', False):
            preds = self.y_scaler.inverse_transform(preds.reshape(-1, 1)).ravel()
            
        return preds

    def save(self, path: str):
        """
        Saves the Keras model to the specified path.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)

    @classmethod
    def load(cls, path: str):
        """
        Loads a Keras model from the specified path.
        """
        model = load_model(path)
        inst = cls.__new__(cls)
        inst.params = {}
        inst.model = model
        return inst