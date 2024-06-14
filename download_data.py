import os
from kaggle.api.kaggle_api_extended import KaggleApi

def download_kaggle_dataset(dataset, path):
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset, path=path, unzip=True)

def download():
    dataset_name = 'ashishkumarak/spotify-reviews-playstore-daily-update'
    download_path = 'data/'

    if not os.path.exists(download_path):
        os.makedirs(download_path)

    download_kaggle_dataset(dataset_name, download_path)

    print("Pobrane pliki:", os.listdir(download_path))

if __name__ == "__main__":
    download()
