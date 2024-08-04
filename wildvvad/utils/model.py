import keras
from keras.layers import Bidirectional, Dense, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential

import tensorflow as tf


class LAND_LSTM_Model:
    """
        Build Bidirectional LSTM Model based on the WildVVAD description.
        Each video sample is mapped onto a sequence of vectors,
        where each vector contains the frontal 3D coordinates of the 68 landmarks.
        This sequence is then fed into a bidirectional LSTM, [16], in which fully
        connected layers share the parameters across time. We employ
        ReLu activations and add batch normalization layers [17]
        between bidirectional LSTM layers.
    """

    def __init__(self):
        pass

    @staticmethod
    def build_land_lstm(input_shape, hp=None, num_td_dense_layers=1,
                        num_blstm_layers=2, dense_dims=96) -> Sequential:
        """Building model

        Args:
            hp: Optional hyperparameter input for keras tuner.
            input_shape(tuple): input shape of the data
            num_td_dense_layers(int): number of time distributed dense layers
            num_blstm_layers(int): number of bidirectional lstm layers
            dense_dims (int): number of dense dimensions for the lstm model, default = 512

        Returns:
            localModel (Sequential()): Returns created Land-LSTM model
        """
        land_lstm_model = Sequential(name="LandLSTM")

        # Flatten at first (providing original feature data)
        land_lstm_model.add(TimeDistributed(
            Flatten(input_shape=(input_shape[-2], input_shape[-1]))))

        if hp:
            dense_dims = hp.Int('units', min_value=32, max_value=512, step=32)

        # Add dense layer with input shape
        land_lstm_model.add(TimeDistributed(Dense(dense_dims, activation='relu'),
                                            input_shape=input_shape))
        for _ in range(num_td_dense_layers):
            land_lstm_model.add(TimeDistributed(Dense(dense_dims, activation='relu')))

        for j in range(num_blstm_layers):
            land_lstm_model.add(Bidirectional(
                layer=keras.layers.LSTM(93, activation='relu', return_sequences=True)))
            if j < num_blstm_layers - 1:
                land_lstm_model.add(BatchNormalization())

        # Flatten again to reduce dimensionality
        land_lstm_model.add(Flatten())

        # Opposite to paper: Use 'sigmoid' activation for binary classification
        land_lstm_model.add(Dense(1, activation='sigmoid'))

        return land_lstm_model
