# import
import glob
import os
import pickle
import random

import h5py
import keras
from kerastuner.tuners import RandomSearch
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import kerasUtils

from wildvvad.utils import helper_functions
from wildvvad.utils.model import LAND_LSTM_Model


# Step 1: Define your model architecture

# Step 2: Get and process your data
# The data should have the following structure:
# Binary data in a 1 dimensional numpy array
def _normalize(arr):
    """ Normalizes the features of the array to [-1, 1].

        Args:
            arr (numpy array): array with features to normalize

        Returns:
            arr_norm (numpy array): numpy array with features normalized to [-1, 1]
    """
    arrMax = np.max(arr)
    arrMin = np.min(arr)
    absMax = np.max([np.abs(arrMax), np.abs(arrMin)])
    return arr / absMax


def shift_to_positive_range(p_cloud):
    # Find the minimum values along each axis
    min_values = np.min(p_cloud, axis=0)

    # Subtract the minimum values from all points
    shifted_pcloud = p_cloud - min_values

    return shifted_pcloud


# Define constants
FOLDERS = ['negatives_rotated', 'positives_rotated']
SAMPLE_LIMIT = 100
SAMPLE_LENGTH = 38

# Initialize lists
sample_paths, x_train, y_train = [], [], []

# Load and process samples
for folder in FOLDERS:
    folder_path = os.path.join(".", "faceFeatures", folder, "*.pickle")
    for i, f in enumerate(glob.glob(folder_path)):
        if i >= SAMPLE_LIMIT:
            break
        sample_paths.append(os.path.join(os.getcwd(), f))
        with open(f, 'rb') as file:
            try:
                sample_file = pickle.load(file)
                if len(sample_file) >= SAMPLE_LENGTH:
                    x_train.append(sample_file)
                    y_train.append(0 if folder == "negatives_rotated" else 1)
            except Exception as e:
                print(f"Error loading {f}: {e}")


# Normalize and resize samples
def normalize_and_resize(samples):
    normalized_resized_samples = []
    for sample in samples:
        sample = np.asarray(sample[:SAMPLE_LENGTH])
        for cloud in sample:
            max_y = np.max(shift_to_positive_range(cloud)[:, 1])
            cloud = _normalize(cloud)
        normalized_resized_samples.append(sample)
    return normalized_resized_samples


x_train_normalized_resized = normalize_and_resize(x_train)

# Shuffle and split data
data = list(zip(x_train_normalized_resized, y_train))
random.shuffle(data)
x_data, y_data = zip(*data)
x_data, y_data = list(x_data), list(y_data)

x_train, x_temp, y_train, y_temp = train_test_split(x_data, y_data, test_size=0.2,
                                                    random_state=42)
x_test, x_val, y_test, y_val = train_test_split(x_temp, y_temp, test_size=0.5,
                                                random_state=42)

# Save test data to HDF5
with h5py.File('wildvvad_testdata_wildvvadmodel_V11_rotated.h5', 'w') as hf:
    hf.create_dataset('x_test', data=x_test)
    hf.create_dataset('y_test', data=y_test)

# Convert lists to numpy arrays
x_train, y_train = np.asarray(x_train), np.asarray(y_train)
x_test, y_test = np.asarray(x_test), np.asarray(y_test)
x_val, y_val = np.asarray(x_val), np.asarray(y_val)

# Step 3: Define the search space for hyperparameters
# In this model, only the configurable dense layers will be tuned

tuner = RandomSearch(
    LAND_LSTM_Model.build_land_lstm_tuner,
    objective='val_binary_accuracy',
    max_trials=10,  # Number of hyperparameter combinations to try
    executions_per_trial=1,  # Number of models to train per trial
    directory='tuner_test',  # Directory to save the tuning logs and checkpoints
    project_name='WildVVAD_tuner'  # Name of the tuning project
)

# Step 4: Configure the tuner
tuner.search_space_summary()

callbacks_array = []
callbacks_array.append(tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(".",
                          "checkpoints/tuner_test-{epoch:02d}.hdf5"),
    verbose=1,
    save_weights_only=False,
    save_freq=10
))

callbacks_array.append(tf.keras.callbacks.TensorBoard(
    log_dir=os.path.join(".", 'tensorboard_logs'),
    update_freq='epoch'
))
callbacks_array.append(
    keras.callbacks.EarlyStopping(monitor='val_binary_accuracy', patience=15))

callbacks_array.append(
    helper_functions.CSVLogger(
        filename=os.path.join(".", 'tuner_outputs_csv.log')))
callbacks_array.append(
    tf.keras.callbacks.TensorBoard('./tuner_test/WildVVAD_tuner'))

# Step 5: Search for the best hyperparameters
tuner.search(x_train, y_train, epochs=60, validation_data=(x_val, y_val))

# Step 6: Save the best model and its corresponding hyperparameters
best_model = tuner.get_best_models(num_models=1)[0]
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
print(best_hyperparameters.values)

try:
    with open('best_hyperparameters.txt', 'w') as f:
        f.write(best_hyperparameters.get_config_as_str())
except Exception as e:
    print("Saving hyperparameters did not work because of ", e)

best_model.build(best_hyperparameters, input_shape=((None,) + x_train.shape[1:]))

best_model.save('best_model.h5')

# Evaluate the best model
loss, accuracy = best_model.evaluate(x_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)
