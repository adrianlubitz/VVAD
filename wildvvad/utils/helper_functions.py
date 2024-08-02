"""
This callback is inspired by the CSVLogger callback from Keras.
reference:
    https://keras.io/api/callbacks/csv_logger/
"""
import collections
import csv
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras


def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.learning_rate

    return lr


class CSVLogger(keras.callbacks.Callback):
    """Callback that streams epoch results to a CSV file.

    Supports all values that can be represented as a string,
    including 1D iterables such as `np.ndarray`.

    This is a custom implementation which is simiar to keras CSVLogger. However,
    the following changes are applied:

        Handling of Log Keys:

            Custom Implementation: Determines keys at the end of the first epoch,
            including keys starting with val_ if not present initially.

            Keras CSVLogger: Assumes keys are consistent across epochs and does not
            explicitly check for val_ keys after the first epoch.

        Handling of Iterable Values:

            Custom Implementation: Converts 1D iterables to strings in the format
            "[value1, value2, ...]".
            Keras CSVLogger: Converts lists and arrays to strings without adding
            additional formatting.

        Duration Logging:

            Custom Implementation: Adds a duration key to log the time taken for each
            epoch.
            Keras CSVLogger: Does not log the duration by default.

        Header Handling:

            Custom Implementation: Checks if a header should be appended based on
            whether the file exists and has content.
            Keras CSVLogger: Similar header handling but with less explicit checking.

        File Handling:

            Custom Implementation: Uses tf.io.gfile.GFile for file operations, which
            supports various filesystems and is TensorFlow-compatible.
            Keras CSVLogger: Uses standard Python file operations with open.

        Exception Handling:

            Custom Implementation: Provides custom handling for zero-dimensional numpy
            arrays.
            Keras CSVLogger: Assumes all values can be stringified directly and does
            not have special handling for edge cases.


    ```python
    csv_logger = CSVLogger('training.log')
    model.fit(X_train, Y_train, callbacks=[csv_logger])
    ```

    Args:
        filename: Filename of the CSV file, e.g. `'run/log.csv'`.
        separator: String used to separate elements in the CSV file.
        append: Boolean. True: append if file exists (useful for continuing
            training). False: overwrite existing file.
    """

    def __init__(self, filename, separator=",", append=False):
        self.sep = separator
        self.filename = tf.compat.path_to_str(filename)
        self.append = append
        self.writer = None
        self.keys = None
        self.append_header = True
        super().__init__()

    def on_train_begin(self, logs=None):
        if self.append:
            if tf.io.gfile.exists(self.filename):
                with tf.io.gfile.GFile(self.filename, "r") as f:
                    self.append_header = not bool(len(f.readline()))
            mode = "a"
        else:
            mode = "w"
        self.csv_file = tf.io.gfile.GFile(self.filename, mode)

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        epoch_time_end = time.time()
        duration = epoch_time_end - self.epoch_time_start

        logs["duration"] = duration

        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, str):
                return k
            elif (
                    isinstance(k, collections.abc.Iterable)
                    and not is_zero_dim_ndarray
            ):
                return f"\"[{', '.join(map(str, k))}]\""
            else:
                return k

        if self.keys is None:
            self.keys = sorted(logs.keys())
            # When validation_freq > 1, `val_` keys are not in first epoch logs
            # Add the `val_` keys so that its part of the fieldnames of writer.
            val_keys_found = False
            for key in self.keys:
                if key.startswith("val_"):
                    val_keys_found = True
                    break
            if not val_keys_found:
                self.keys.extend(["val_" + k for k in self.keys])

        if not self.writer:

            class CustomDialect(csv.excel):
                delimiter = self.sep

            fieldnames = ["epoch"] + self.keys

            self.writer = csv.DictWriter(
                self.csv_file, fieldnames=fieldnames, dialect=CustomDialect
            )
            if self.append_header:
                self.writer.writeheader()

        row_dict = collections.OrderedDict({"epoch": epoch})
        row_dict.update(
            (key, handle_value(logs.get(key, "NA"))) for key in self.keys
        )
        self.writer.writerow(row_dict)
        self.csv_file.flush()

    def on_train_end(self, logs=None):
        self.csv_file.close()
        self.writer = None
