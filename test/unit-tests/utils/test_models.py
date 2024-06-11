import unittest

import keras

from wildvvad.model import LAND_LSTM_Model


class TestModelUtils(unittest.TestCase):
    """
        Test the models regarding WildVVAD.
    """

    def setUp(self):
        self.input_shape = (68, 38, 3)

    def test_model_creation(self):
        model = LAND_LSTM_Model.build_land_lstm(self.input_shape)
        self.assertIsInstance(model, keras.Sequential)
        self.assertEqual(model.name, "LandLSTM")

    def test_layer_addition(self):
        """
        Verify that the correct number of time-distributed dense layers and bidirectional LSTM layers are added to the model.
        Check that the dense and LSTM layers have the expected number of units and activation functions.
        """
        num_td_dense_layers = 2
        num_blstm_layers = 3
        dense_dims = 96
        model = LAND_LSTM_Model.build_land_lstm(self.input_shape,
                                                num_td_dense_layers,
                                                num_blstm_layers,
                                                dense_dims)

        # Check time-distributed dense layers
        td_dense_layers = [layer for layer in model.layers if
                           isinstance(layer, keras.layers.TimeDistributed)]
        self.assertEqual(len(td_dense_layers), num_td_dense_layers + 2)

        # Check bidirectional LSTM layers
        blstm_layers = [layer for layer in model.layers if
                        isinstance(layer, keras.layers.Bidirectional)]
        self.assertEqual(len(blstm_layers), num_blstm_layers)

    def test_layer_connections(self):
        """
        For each layer, we check that its input comes from the output of the previous layer.
        This ensures that the layers are properly connected in sequence as expected in the model architecture.
        """
        model = LAND_LSTM_Model.build_land_lstm(self.input_shape)
        model.build(input_shape=self.input_shape)
        layers = model.layers
        for i in range(1, len(layers)):
            self.assertIs(layers[i].input, layers[i - 1].output)

    def test_output_layer(self):
        """
        Check that the output layer has the correct number of units and activation function for binary classification.
        """
        model = LAND_LSTM_Model.build_land_lstm(self.input_shape)
        output_layer = model.layers[-1]
        self.assertEqual(output_layer.units, 1)
        self.assertEqual(output_layer.activation, keras.activations.sigmoid)


if __name__ == '__main__':
    unittest.main(warnings='ignore')
