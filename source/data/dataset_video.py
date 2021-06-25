import random
from os import listdir
from os.path import join

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas
import torch
import torch.nn.functional as F
import torch.utils.data as data

################################################################################
### TrainSet


class VideoSampler(torch.utils.data.Sampler[int]):
    r"""Samples elements sequentially, always in the same order.
    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, shuffle=True):

        self.shuffle = shuffle

        self.dt = data_source.data[0].to_numpy()

        if self.shuffle:
            self.indexes = np.random.permutation(np.unique(self.dt))
        else:
            self.indexes = np.unique(self.dt)

    def __iter__(self):
        return iter(self.indexes)

    def __len__(self):
        return len(np.unique(self.dt))


def collate_pad_frames(batch):

    batch = list(batch)

    X_lengths = [tens[0].shape[0] for tens in batch]

    m_len = max(X_lengths)

    inputt = []
    target = []
    masks = []

    for ii in range(len(batch)):
        diff = m_len - X_lengths[ii]
        batch[ii] = list(batch[ii])
        pad = torch.zeros(diff, 18)
        pad_gt = torch.zeros(diff, 3)
        inputt.append(torch.cat([batch[ii][0], pad], 0))
        target.append(batch[ii][1])

        masks.append(
            torch.cat([torch.ones(X_lengths[ii]), torch.zeros(diff)]).type(
                torch.ByteTensor
            )
        )

    inputt = torch.stack(inputt)
    target = torch.stack(target)
    mask = torch.stack(masks)

    if len(batch[0]) == 3:
        return inputt, target, mask, X_lengths, batch[0][2]
    else:
        return inputt, target, mask, X_lengths


class DatasetFromFolder(data.Dataset):
    def __init__(self, data_file):
        super(DatasetFromFolder, self).__init__()

        self.data_file = data_file

        self.data = pandas.read_csv(data_file, header=None)

        dt = self.data[0].to_numpy()

        self.len = len(np.unique(dt))

    def __getitem__(self, index):

        # print(index)
        dt = self.data[self.data[0] == index]

        dt = dt.to_numpy()

        ins = []

        for ii in range(dt.shape[0]):
            ins.append(torch.from_numpy(dt[ii, 5:].astype(np.float32)))
        target = torch.from_numpy(dt[0, 2:5].astype(np.float32))

        inputt = torch.stack(ins)

        return inputt, target

    def __len__(self):
        return self.len


################################################################################
### ValidationSet


class ValsetFromFolder(data.Dataset):
    def __init__(self, data_file):
        super(ValsetFromFolder, self).__init__()

        self.data_file = data_file

        self.data = pandas.read_csv(data_file, header=None)

        dt = self.data[0].to_numpy()

        self.len = len(np.unique(dt))

    def __getitem__(self, index):

        dt = self.data[self.data[0] == index]

        dt = dt.to_numpy()

        ins = []

        for ii in range(dt.shape[0]):
            ins.append(torch.from_numpy(dt[ii, 5:].astype(np.float32)))
        target = torch.from_numpy(dt[0, 2:5].astype(np.float32))

        inputt = torch.stack(ins)
        return inputt, target

    def __len__(self):
        return self.len


################################################################################
### TestSet


class TestsetFromFolder(data.Dataset):
    def __init__(self, data_file):
        super(TestsetFromFolder, self).__init__()

        self.data_file = data_file

        self.data = pandas.read_csv(data_file, header=None)

        dt = self.data[0].to_numpy()

        self.len = len(np.unique(dt))

    def __getitem__(self, index):

        dt = self.data[self.data[0] == index]

        name = str(index)

        dt = dt.to_numpy()

        ins = []

        for ii in range(dt.shape[0]):
            ins.append(torch.from_numpy(dt[ii, 5:].astype(np.float32)))
        target = torch.from_numpy(dt[0, 2:5].astype(np.float32))

        inputt = torch.stack(ins)

        return inputt, target, name

    def __len__(self):
        return self.len
