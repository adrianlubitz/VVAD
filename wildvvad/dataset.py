import os
import pickle
import random

import numpy as np

from wildvvad.sample import Sample


class dataSet:
    """
    This class contains all methods to create or load a dataset suitable for the
    LandLSTM model.
    """

    def __init__(self):
        """
        Class initialization: Init sample methods
        """
        self.sample = Sample()

    def create_vector_dataset_from_videos(self, path: str = './utils',
                                          folders=['speaking_videos', 'silent_videos'],
                                          save_to: str = '',
                                          ) -> bool:
        """
        Preprocesses the video files and creates data set for the model.
        The data set consist of pickle files (list objects). Each represents one sample.
        Given the path to the video folders

        Args:
            path (str) : path to folders (pos, neg). Folders name must be
                'speaking_videos' and 'silent_videos'
            folders (numpy array) : expected folders for positive and negative samples
            save_to (str) : path to save the dataset files to (default empty)
        Returns:
            ok (bool): Returns result of data creation (True = Ok, False = Error)
        """

        subfolders = [f.name for f in os.scandir(path) if f.is_dir()]

        # look for folder existence
        for folder in folders:
            if folder not in subfolders:
                print(f"Folder {folder} with videos not found! Function terminated.")
                return

        # go through each file in folder
        for folder in folders:
            print(f"Enter folder {folder}")
            for filename in os.scandir(os.path.join(path, folder)):
                if filename.is_file() and filename.name.endswith('.mp4'):
                    print(f"Found file {filename}")

                    pickle_filename = os.path.join(
                        save_to, filename).replace('.mp4', '.pickle')

                    # Check if pickle file already exists
                    if os.path.exists(pickle_filename):
                        print(
                            f"Pickle file {pickle_filename} already exists. Skipping.")
                        continue

                    # convert to list of landmarks with face forward
                    print(f"Get sample from {os.path.join(filename)}")
                    current_sample = self.sample.load_video_sample_from_disk(
                        file_path=os.path.join(filename))

                    video_sample = []

                    for image in current_sample:
                        preds = self.sample.get_face_landmark_from_sample(image)[-1]
                        # calculate euclidean distance and normalize
                        # outmost eye corner is landmark 36 (right eye) and
                        # landmark 45 (left eye)
                        # get euclidean distance
                        corner_right_eye = preds[36]
                        corner_left_eye = preds[45]
                        euclidean_distance = np.linalg.norm(
                            corner_left_eye - corner_right_eye)
                        # normalize on euclidean distance
                        for i in range(len(preds)):
                            preds[i] = (1 / euclidean_distance) * preds[i]
                        print("Normalized to euclidean distance.")

                        rotated_landmarks = self.sample.align_3d_face(preds)

                        video_sample.append(rotated_landmarks)

                    print(f"Safe file with label. Label "
                          f"is {True if folder == 'speaking_videos' else False}")
                    sample_with_label = {
                        "sample": video_sample,
                        "label": True if folder == "speaking_videos" else False
                    }

                    # save as pickle file
                    save_path = os.path.join(save_to, filename)
                    with open(pickle_filename, 'wb') as handle:
                        pickle.dump(sample_with_label, handle,
                                    protocol=pickle.HIGHEST_PROTOCOL)

    def load_data_set_from_pickles(
            self, path: str = './utils',
            folders=['speaking_videos', 'silent_videos'],
            random_seed: int = 42) -> list:
        """
        Load complete dataset from available sample pickle files

        Args:
            path (str) : path to folders (pos, neg). Folders name must be
                        'speaking_videos' and 'silent_videos'
            folders (numpy array) : expected folders for positive and negative samples
            random_seed (int) : value for repeatable random shuffling of data

        Returns:
            dataset (list): Returns all data as list of dict (sample, label)
        """

        loaded_dataset = []

        subfolders = [f.name for f in os.scandir(path) if f.is_dir()]

        # look for folder existence
        for folder in folders:
            if folder not in subfolders:
                print(f"Folder {folder} with videos not found! Function terminated.")
                return

        # go through each file in folder
        for folder in folders:
            print(f"Enter folder {folder}")

            data = self.sample.load_sample_objects_from_disk(
                folder_path=os.path.join(path, folder))

            for datapoint in data:
                loaded_dataset.append(datapoint)

            print(f"Length of dataset is {len(loaded_dataset)}")

        random.seed(random_seed)
        random.shuffle(loaded_dataset)

        return loaded_dataset


if __name__ == '__main__':
    dataset = dataSet()
    # dataset.create_vector_dataset_from_videos()
    # dataset = dataset.load_data_set_from_pickles()
    # kerasUtilities = kerasUtils()
    # kerasUtilities.train_test_split(dataset=dataset)
