import torch
from torch.utils.data import Dataset, DataLoader

import re
import os
import numpy as np
from sklearn.preprocessing import normalize

from scipy.io import loadmat
from utils.constants import CLASSES_INDEX


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

# train + validate, test
def dataloader(url: str, num_sample, std=True) -> (list, list):
    train_dataset, test_dataset = [], []
    label_prefix = 'new_mmsi'
    exp = re.compile(r"(?<=_)\d+")
    if os.path.exists(url) and os.path.isdir(url):
        files = os.listdir(url)
        for file in files:
            pathname = os.path.join(url, file)
            if os.path.exists(pathname) and os.path.isfile(pathname) and pathname.find(".mat"):
                raw_data = loadmat(pathname)
                label = re.findall(exp, file)[0]
                data = raw_data.get(label_prefix + label, [])[0]
                train_data = data[:num_sample]

                for value in train_data:
                    if CLASSES_INDEX.get(int(label), None) is not None:
                        v = np.reshape(value, len(value))
                        if std:
                            v = standardization(v)
                        train_dataset.append((v, CLASSES_INDEX[int(label)]))

                test_data = data[num_sample:]

                for value in test_data:
                    if CLASSES_INDEX.get(int(label), None) is not None:
                        v = np.reshape(value, len(value))
                        if std:
                            v = standardization(v)
                        test_dataset.append((v, CLASSES_INDEX[int(label)]))

    return train_dataset, test_dataset


class AISTransform(object):
    def __init__(self, train_size):
        self.train_size = train_size

    def __call__(self, sample):
        signal, label = sample
        if isinstance(self.train_size, slice):
            signal = signal[self.train_size]
        else:
            signal = signal[:self.train_size]
        signal = torch.tensor(signal, dtype=torch.float)
        return signal, label


class AISDataset(Dataset):
    def __init__(self, url: str, num_sample=80, transform=None, train=True):
        super(AISDataset, self).__init__()
        self.dataset = dataloader(url, num_sample)[0 if train else 1]
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.dataset[idx]

        if self.transform is not None:
            sample = self.transform(sample)
        return sample
