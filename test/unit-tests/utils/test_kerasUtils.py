import unittest

from vvadlrs3.utils.kerasUtils import (Hdf5DataGenerator, DataGenerator,
                                       DataGeneratorRAM, Models)

class TestKerasUtils(unittest.TestCase):
    """"""

    def setup(self):
        hdf5gen = Hdf5DataGenerator()
        datagen = DataGenerator()
        ramgen = DataGeneratorRAM()
        models = Models()

    def test_split_dataset(self):
        pass

    def test_len(self):
        pass

    def test_getitem(self):
        pass

    def test_on_epoch_end(self):
        pass

    def test_data_generation(self):
        pass

    def test_build_feature_lstm(self):
        pass

    def test_build_conv_lstm2d(self):
        pass

    def test_build0(self):
        pass

    def test_build_time_distributed(self):
        pass

    def test_build_timedistributed_functional(self):
        pass

    def test_build_baseline_model(self):
        pass

    def test_train_baseline_model(self):
        pass

    def test_save_history(self):
        pass

    train_model_general
    gen_data_general
    gen_data_internal_general
    hdf5_samples_to_memory
    get_model_memory_usage
    check_data_gen
    test_model
