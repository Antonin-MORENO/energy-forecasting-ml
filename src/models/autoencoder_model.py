import os
from keras import Model, Input
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.models import load_model
from src.models.base_model import BaseModel 
from keras.layers import Dense, BatchNormalization, Activation

class AutoEncoderModel(BaseModel):
    """
    Dense autoencoder for non-linear feature reduction.
    Inherits from BaseModel to reuse fit()/predict()/save()/load() methods.
    """
    def __init__(self, params: dict):
        """
        Constructor for the AutoEncoderModel.
        :param params: A dictionary of parameters for building and training the model.
        """
        super().__init__(params)
        self.build_model()

    def build_model(self):
        """
        Builds the autoencoder architecture using Keras.
        The architecture is defined by the parameters passed during instantiation.
        """
        p = self.params
        inp_dim = p['input_shape']

        
        # Input layer
        inp = Input(shape=(inp_dim,), name='ae_input')

        # Hidden layer 1: 32 neurons
        x = Dense(32, activation=None, name='enc_dense_1')(inp)
        x = BatchNormalization(name='enc_bn_1')(x)
        x = Activation('relu', name='enc_act_1')(x)

        # Hidden layer 2: 16 neurons
        x = Dense(16, activation=None, name='enc_dense_2')(x)
        x = BatchNormalization(name='enc_bn_2')(x)
        x = Activation('relu', name='enc_act_2')(x)

        # Bottleneck: 8 neurons (the compressed representation)
        bottleneck = Dense(8, activation='relu', name='bottleneck')(x)


        # Decoder reconstruction layer 2: 16 neurons
        x = Dense(16, activation=None, name='dec_dense_2')(x)
        x = BatchNormalization(name='dec_bn_2')(x)
        x = Activation('relu', name='dec_act_2')(x)

        # Decoder reconstruction layer 3: 32 neurons
        x = Dense(32, activation=None, name='dec_dense_3')(x)
        x = BatchNormalization(name='dec_bn_3')(x)
        x = Activation('relu', name='dec_act_3')(x)

        # Final output layer with dimension matching the input
        decoded = Dense(inp_dim, activation='linear', name='decoder_output')(x)

        # Instantiate the models
        # The full autoencoder maps input to its reconstruction
        self.autoencoder = Model(inputs=inp, outputs=decoded, name='SmoothFunnelAutoEncoder')
        # The encoder part maps input to the bottleneck
        self.encoder = Model(inputs=inp, outputs=bottleneck, name='SmoothFunnelEncoder')

        # Compile the autoencoder
        self.autoencoder.compile(
            optimizer=p.get('optimizer', 'adam'),
            loss=p.get('loss', 'mse'),
            metrics=p.get('metrics', [])
        )

    def fit(self, X_train, X_val=None):
        """
        Trains the autoencoder.
        - X_train: array [n_samples, n_features] for training.
        - X_val: array [n_val, n_features] for validation, optional.
        """
        p = self.params
        callbacks = []

        # Checkpoint: Save the best model during training
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

        # Reduce LR on Plateau: Reduce learning rate when a metric has stopped improving
        callbacks.append(
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=p.get('reduce_lr_factor', 0.6),
                patience=p.get('reduce_lr_patience', 3),
                min_lr=p.get('min_lr', 1e-6),
                verbose=1
            )
        )

        # Early Stopping: Stop training when validation loss stops improving
        if p.get('early_stop_patience'):
            callbacks.append(
                EarlyStopping(
                    monitor='val_loss',
                    patience=p['early_stop_patience'],
                    restore_best_weights=True,
                    verbose=1
                )
            )

        # Start training
        val_data = (X_val, X_val) if X_val is not None else None
        # Note: For an autoencoder, the input and target data are the same (X_train)
        return self.autoencoder.fit(
            X_train, X_train,
            validation_data=val_data,
            epochs=p.get('epochs', 100),
            batch_size=p.get('batch_size', 32),
            callbacks=callbacks,
            verbose=p.get('verbose', 1)
        )

    def encode(self, X):
        """Returns the latent code (bottleneck representation)."""
        return self.encoder.predict(X, verbose=0)

    def predict(self, X):
        """Returns the reconstructed data (useful for evaluating reconstruction error)."""
        return self.autoencoder.predict(X, verbose=0)

    def save(self, path: str):
        """Saves the complete autoencoder model."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.autoencoder.save(path)

    @classmethod
    def load(cls, path: str):
        """
        Loads a pre-trained model.
        This method also reconstructs the self.encoder from the 'bottleneck' layer.
        """
        # Load the full autoencoder
        ae = load_model(path)
        
        # Find the bottleneck layer
        bottleneck = ae.get_layer('bottleneck')
        # Recreate the encoder model from the autoencoder's input and the bottleneck's output
        encoder = Model(inputs=ae.input, outputs=bottleneck.output)

        # Create a new instance of this class without calling __init__
        inst = cls.__new__(cls)
        inst.params = {} # Params are not saved, so initialize as empty
        inst.autoencoder = ae
        inst.encoder = encoder
        return inst