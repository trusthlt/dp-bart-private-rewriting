import requests
import zipfile
import shutil
import os
import pathlib
import re
from bs4 import BeautifulSoup
from tqdm.auto import tqdm
from datasets import load_dataset
import pdb

project_assets_folder = os.path.join(pathlib.Path(__file__).parent.resolve(),
                                     'assets')


def download_asset(name, asset_dir=project_assets_folder):
    """ Download asset (such as a dataset or embeddings).

    :param name: Name of the asset.
    :param asset_dir: Parent folder of all assets.
    :return:
    """
    data_subdir = os.path.join(asset_dir, 'raw', name)
    if not os.path.exists(data_subdir):
        os.makedirs(data_subdir)

    print(f'The assets will be saved in: {data_subdir}')

    if name == 'reddit_mental_health':
        r = requests.get('https://zenodo.org/record/3941387')
        soup = BeautifulSoup(r.text, 'lxml')
        links = [line.get('href')
                 for line in soup.find_all('link', rel='alternate')]
        raw_files = [re.findall(r'/(\w+\.csv)', link)[0] for link in links]
        raw_files_full = [os.path.join(data_subdir, f) for f in raw_files]
        available = _check_if_raw_available(raw_files_full)
        if not available:
            assert len(links) == len(raw_files)
            for idx, link in enumerate(links):
                save_path = os.path.join(data_subdir, raw_files[idx])
                _download_file(link, save_path)
    elif name == 'drugscom_reviews_rating':
        # Check if raw files already available
        raw_files = ['drugsComTrain_raw.tsv', 'drugsComTest_raw.tsv']
        raw_files = [os.path.join(data_subdir, f) for f in raw_files]
        available = _check_if_raw_available(raw_files)
        if not available:
            # If not, download raw files and extract from zip
            url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00462/drugsCom_raw.zip'
            save_path = os.path.join(data_subdir, 'drugsCom_raw.zip')
            _download_file(url, save_path)
            _extract_zip(save_path, data_subdir)
    elif name == 'drugscom_reviews_condition':
        raw_files = ['drugsComTrain_raw.tsv', 'drugsComTest_raw.tsv']
        raw_files = [os.path.join(data_subdir, f) for f in raw_files]
        available = _check_if_raw_available(raw_files)
        if not available:
            url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00462/drugsCom_raw.zip'
            save_path = os.path.join(data_subdir, 'drugsCom_raw.zip')
            _download_file(url, data_subdir)
            _extract_zip(save_path, data_subdir)
    elif name == 'amazon_reviews_subset':
        # Go straight to the preprocessing script
        pass
    elif name == 'glove':
        raise NotImplementedError
    elif name == 'word2vec_googlenews':
        raise NotImplementedError
        url = 'https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz'
        save_path = os.path.join(data_subdir, 'GoogleNews-vectors-negative300.bin.gz')
        _download_file(url, save_path)
    else:
        raise NotImplementedError('This asset is not available for download as of now.')


def _check_if_raw_available(raw_files):
    for fname in raw_files:
        present = os.path.isfile(fname)
        if not present:
            return False
    return True


def _download_file(url, save_path):
    print('Downloading:')
    print(f'\tfrom: {url}')
    print(f'\tto: {save_path}')
    with requests.get(url, stream=True) as r:
        file_size = _file_size_human_readable(len(r.content))
        print("File size:", file_size)
        with open(save_path, 'wb+') as f:
            f.write(r.content)
    print('Finished downloading.')


def _extract_zip(in_path, out_dir_path):
    print('Extracting Zip:')
    print(f'\tto: {out_dir_path}')
    with zipfile.ZipFile(in_path, 'r') as zip_ref:
        zip_ref.extractall(out_dir_path)


def _file_size_human_readable(size):
    units = ["", "K", "M", "G", "T"]
    for unit in units:
        if abs(size) < 1024.0:
            return f"{size:3.2f}{unit}B"
        size /= 1024.0
    return f"{size:.2f}PB"
