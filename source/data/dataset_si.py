import random
from os import listdir
from os.path import join

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas
import torch
import torch.utils.data as data

################################################################################
# TrainSet


class DatasetFromFolder(data.Dataset):
    def __init__(self, data_file, reduced=None):
        super(DatasetFromFolder, self).__init__()

        self.data_file = data_file

        db = pandas.read_csv(data_file, header=None)

        self.data = db.to_numpy()

        self.reduced = reduced

    def __getitem__(self, index):

        dt = self.data[index]

        name = dt[0]
        gt = dt[1:4].astype(np.float32)

        if self.reduced == '1':  # remove SoG
            inputt = dt[7:].astype(np.float32)
        elif self.reduced == '2':  # remove GW
            inputt = dt[7:19].astype(np.float32)
        elif self.reduced == '3':  # remove GGW
            inputt = np.concatenate(
                (dt[7:13].astype(np.float32), dt[16:19].astype(np.float32)), axis=0)
        elif self.reduced == '4':  # remove GE1
            inputt = np.concatenate(
                (dt[10:13].astype(np.float32), dt[16:19].astype(np.float32)), axis=0)
        elif self.reduced == '5':  # remove GW
            inputt = dt[10:13].astype(np.float32)
        else:
            inputt = dt[4:].astype(np.float32)

        inputt = torch.from_numpy(inputt)
        target = torch.from_numpy(gt)

        return inputt, target

    def __len__(self):
        return self.data.shape[0]


################################################################################
# ValidationSet


class ValsetFromFolder(data.Dataset):
    def __init__(self, data_file, reduced=None):
        super(ValsetFromFolder, self).__init__()

        self.data_file = data_file

        db = pandas.read_csv(data_file, header=None)

        self.data = db.to_numpy()

        self.reduced = reduced

    def __getitem__(self, index):

        dt = self.data[index]

        name = dt[0]
        gt = dt[1:4].astype(np.float32)

        if self.reduced == '1':  # remove SoG
            inputt = dt[7:].astype(np.float32)
        elif self.reduced == '2':  # remove GW
            inputt = dt[7:19].astype(np.float32)
        elif self.reduced == '3':  # remove GGW
            inputt = np.concatenate(
                (dt[7:13].astype(np.float32), dt[16:19].astype(np.float32)), axis=0)
        elif self.reduced == '4':  # remove GE1
            inputt = np.concatenate(
                (dt[10:13].astype(np.float32), dt[16:19].astype(np.float32)), axis=0)
        elif self.reduced == '5':  # remove GW
            inputt = dt[10:13].astype(np.float32)
        else:
            inputt = dt[4:].astype(np.float32)

        inputt = torch.from_numpy(inputt)
        target = torch.from_numpy(gt)

        return inputt, target, name

    def __len__(self):
        return self.data.shape[0]


################################################################################
# TestSet


class TestsetFromFolder(data.Dataset):
    def __init__(self, data_file, reduced=None):
        super(TestsetFromFolder, self).__init__()

        self.data_file = data_file

        db = pandas.read_csv(data_file, header=None)

        self.data = db.to_numpy()

        self.reduced = reduced

    def __getitem__(self, index):

        dt = self.data[index]

        name = dt[0]
        gt = dt[1:4].astype(np.float32)
        inputt = dt[4:].astype(np.float32)

        if self.reduced == '1':  # remove SoG
            inputt = dt[7:].astype(np.float32)
        elif self.reduced == '2':  # remove GW
            inputt = dt[7:19].astype(np.float32)
        elif self.reduced == '3':  # remove GGW
            inputt = np.concatenate(
                (dt[7:13].astype(np.float32), dt[16:19].astype(np.float32)), axis=0)
        elif self.reduced == '4':  # remove GE1
            inputt = np.concatenate(
                (dt[10:13].astype(np.float32), dt[16:19].astype(np.float32)), axis=0)
        elif self.reduced == '5':  # remove GW
            inputt = dt[10:13].astype(np.float32)
        else:
            inputt = dt[4:].astype(np.float32)

        inputt = torch.from_numpy(inputt)
        target = torch.from_numpy(gt)

        return inputt, target, name

    def __len__(self):
        return self.data.shape[0]
