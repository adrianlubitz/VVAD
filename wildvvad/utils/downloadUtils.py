import os
import os.path
import shutil
import time
import urllib.request
import zipfile

from tqdm import tqdm


class downloadUtils:
    def __init__(self):
        """
        Initializing the URLs and file names for the WildVVAD video files to download
        """

        self.speaking_videos_url = \
            "http://perception.inrialpes.fr/Free_Access_Data/WVVAD/speaking_videos.zip"
        self.speaking_videos_file = 'speaking_videos.zip'
        self.speaking_videos_folder = 'speaking_videos'

        self.silent_videos_url = \
            "http://perception.inrialpes.fr/Free_Access_Data/WVVAD/silent_videos.zip"

        self.silent_videos_file = 'silent_videos.zip'
        self.silent_videos_folder = 'silent_videos'

    def download_and_save_speaking_videos(self) -> None:
        """
        This method will download, save, and extract all speaking_videos videos from
        the WildVVAD data set.
        """

        print("Download speaking_videos videos.")
        self.download_and_save_videos(self.speaking_videos_url,
                                      self.speaking_videos_folder,
                                      self.speaking_videos_file)

    def download_and_save_silent_videos(self) -> None:
        """
        This method will download, save, and extract all silent_videos videos from
        the WildVVAD data set.
        """
        print("Download silent_videos videos.")
        self.download_and_save_videos(self.silent_videos_url,
                                      self.silent_videos_folder,
                                      self.silent_videos_file)

    @staticmethod
    def download_and_save_videos(url: str, folder_name: str, file_name: str,
                                 max_retries: int = 5, delay: int = 5) -> None:
        """
        General method to download, save, and extract zip files from URLs


        Args:
            url (str): URL to zip file to download
            folder_name (str): Folder to save the file into
            file_name (str): Name to save the file as
            max_retries (int): Maximum number of retries for downloading the file.
            delay (int): Delay in seconds between retry attempts.
        """

        if not os.path.exists(folder_name):
            os.mkdir(folder_name)

        retries = 0
        file_exists = False
        while not file_exists and retries < max_retries:
            try:
                with urllib.request.urlopen(url) as response, open(
                        os.path.join(folder_name, file_name), 'wb') as out_file:
                    shutil.copyfileobj(response, out_file)
                    with zipfile.ZipFile(os.path.join(folder_name, file_name)) as zf:
                        zf.extractall(folder_name)
                    file_exists = True
            except urllib.error.URLError as e:
                retries += 1
                print(
                    f"Failed to download the file. Attempt {retries} of {max_retries}. Error: {e}")
                time.sleep(delay)
            except zipfile.BadZipFile as e:
                print(
                    f"Downloaded file is not a zip file or it is corrupted. Error: {e}")
                break
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                break

        if not file_exists:
            print(
                "Failed to download the file after multiple attempts. Please check your internet connection and try again.")


class DownloadProgressBar(tqdm):
    """
    Download progress bar used during zip file download. Indicating progress.
    """

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


if __name__ == "__main__":
    # Testing:
    utils = downloadUtils()
    utils.download_and_save_speaking_videos()
