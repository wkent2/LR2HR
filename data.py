import os
import re
import typing
import h5py
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

import warnings

warnings.filterwarnings("ignore")


def shuffle_lists(*a, rng=None):
    """
    a,bc, ... = shuffle_lists(a,b,c,..., rng=None)
    Shuffle multiple lists randomly but such that they are shuffled in the same order
    as each other.
    """
    if rng is None:
        rng = np.random.default_rng()
    temp = list(zip(*a))
    rng.shuffle(temp)
    return zip(*temp)


def remove_bad_data(X, y):
    X = X[~np.isnan(y).any(axis=1), ...]
    y = y[~np.isnan(y).any(axis=1), :]
    return (X, y)


def data_augment_3d(X, Y, factor=4, shuffle=True, rng=None):
    """
    X_aug, Y_aug = data_augment_3d(X,Y,factor=4, shuffle=True, rng=None)
    X and Y should be a LIST of data that correspond to each other.
    Each entry of X must be 3-dimensional.

    Apply random permute and flip operations to augment a dataset of size N to
    a dataset of size factor*N.
    Augmented data will be shuffled in, with the appropriate Y values at
    matching indices (i.e. if X_aug[i] is a flip of X_aug[j], Y[i] and Y[j] will
    be identical too).

    If shuffle=False, they won't be shuffled; new data will be at the end of the
    old data.

    To turn a numpy array of shape [N,...] (where ... is the size of
    each individual dataset) into a list like this function needs, do list(X)
    To turn the output back into a numpy array, just do np.array(X_aug) and
    np.array(Y_aug).
    """
    print("Augmenting data by a factor of",factor)
    if rng is None:
        rng = np.random.default_rng()

    X_additional = []
    Y_additional = []
    N_orig = len(X)
    N_additional = int((factor - 1) * N_orig)
    permute_axes = [0, 1, 2]
    for _ in range(N_additional):
        # Pick a random X and Y
        i = rng.integers(0, N_orig)
        x = X[i]
        y = Y[i]
        did_anything = False
        # Do a while loop and choose randomly whether or not to do each operation.
        # If we don't do any of them at all, start over - we don't want a straight
        # copied array!
        while not did_anything:
            # Permute axes, or not
            if rng.binomial(1, 0.5):  # Coin flip
                # Randomly shuffle the axes, but also make sure they don't end up
                # back at [0,1,2] which would do nothing at all
                rng.shuffle(permute_axes)
                while permute_axes == [0, 1, 2]:
                    rng.shuffle(permute_axes)
                x = np.transpose(x, permute_axes)
                did_anything = True
            # Flip on axis 0, or not... same with 1 and 2
            flipaxes = []
            if rng.binomial(1, 0.5):  # coin flip
                flipaxes.append(0)
            if rng.binomial(1, 0.5):
                flipaxes.append(1)
            if rng.binomial(1, 0.5):
                flipaxes.append(2)
            # If any axes are there at all, do the operation
            if len(flipaxes):
                x = np.flip(x, flipaxes)
                did_anything = True
        # Okay we got here, x should now be all flipped and permuted around.
        # Save it, and the corresponding y value(s), to the lists.
        X_additional.append(x)
        Y_additional.append(y)

    # Now append them to the end of the lists X and Y
    X_aug = X + X_additional
    Y_aug = Y + Y_additional

    # Now shuffle their order, but in the same way
    if shuffle:
        X_aug, Y_aug = shuffle_lists(X_aug, Y_aug, rng=rng)

    return (X_aug, Y_aug)


def test_train_split_by_job_group(X, y, names, test_frac=0.2, shuffle=True, rng=None):
    """
    X_test, y_test, X_train, y_train, i_test = test_train_split_by_job_group(X,y,names,test_frac = 0.05, shuffle=True, rng=None)
    Given X data, y data, and a corresponding list of encoded names like
    AAABBBCCCDEFGG, splits X and y data up, making sure to keep any given
    microstructure together (not splitting the affiliated data apart)
    as it evolves through time (incl. at different temperatures).
    This way there won't be twinned data between the test and train set.

    X and Y and names should be a LIST of data that correspond to each other.
    To turn a numpy array of shape [N,...] (where ... is the size of
    each individual dataset) into a list like this function needs, do list(X)
    To turn the output back into a numpy array, just do np.array(X) and
    np.array(Y).

    Output i_test is a list of the index values from X and y that became part
    of the test dataset. Might be useful if you want to replicate the split.
    """
    # Jobcodes are like AAABBBCCCDEFGG where A-D are nominal initial parameters,
    # and EFGG represent time, temperature, and subvolume ID
    # We want to keep A-D and the subvol ID (GG) together across the split.
    # Also, if there are multiple temperatures, we should probably keep them
    # together across the split, too, since they will be similar to each other.
    # this can be best accomplished by:
    #   Create a modified jobcode AAABBBCCCDGG - no time or temperature
    #   Reduce to unique values only
    #   Keep together microstructures that start with AAABBBCCCD and end with GG
    #     - this will keep all timesteps and temperatures together
    jobcodes = sorted(list(set([x[:10] + x[-2:] for x in names])))
    if shuffle:
        if rng is None:
            print("rng: generating new RNG in test_train_split_by_job_group")
            rng = np.random.default_rng()
        rng.shuffle(jobcodes)
    # Now do the test train split on these jobcodes.
    # We will assume there are an equal number of microstructures for each.
    # Not always true due to errors but hopefully it will balance out.
    # If not we might need to modify this code to re-shuffle until it finds a
    # configuration that gives the right test/train split value.
    split_index = int(np.round(test_frac * len(jobcodes)))
    jobcodes_test = jobcodes[:split_index]
    jobcodes_train = jobcodes[split_index:]
    X_test = []
    y_test = []
    X_train = []
    y_train = []
    i_test = []
    # Save names too (WFK)
    names_test = []
    names_train = []

    for i, name in enumerate(names):
        # Convert this full AAABBBCCCDEFGG name to the modified type (AAABBBCCCDGG)
        # to match the entries in list "jobcodes"
        name_modified = name[:10] + name[-2:]
        if name_modified in jobcodes_train:
            X_train.append(X[i])
            y_train.append(y[i])
            names_train.append(name)
        elif name_modified in jobcodes_test:
            # This should really just be an "else" but I'm doing this for debug
            X_test.append(X[i])
            y_test.append(y[i])
            i_test.append(i)
            names_test.append(name)
        else:
            print("  Strange! " + name + "did not match with any jobcodes.")

    final_test_frac = len(X_test) / (len(X_test) + len(X_train))

    print(
        "Test/train split: was targeting a fraction of",
        test_frac,
        ", actual fraction after keeping sets together is",
        final_test_frac,
    )

    return (X_test, y_test, X_train, y_train,names_test,names_train, i_test)


class Microstructures(Dataset):

    def __init__(
        self,
        file_path,
        output_val: typing.Union[typing.List, int],
        transform=None,
        augment=False,
        factor=4,
        remove_bad=True,
        contrast='A',
        job_group=None,
    ):
        # Load dataset
        self.file_path = file_path
        self.output_val = output_val
        self.transform = transform
        self.augment = augment
        self.factor = factor
        self.remove_bad = remove_bad
        self.contrast=contrast
        self.job_group = job_group



        with h5py.File(file_path, "r") as f:
            self.X = f["X"][:]  # Adjust 'your_dataset_key' as needed
            self.y = f["y"][:]
            self.names = f["names"][:]

        if type(output_val) == int:
            self.y = self.y[:, output_val : output_val + 1]
        elif len(output_val) == 1:
            self.y = self.y[:, output_val[0] : output_val[0] + 1]
        else:
            self.y = self.y[:, output_val]

        if remove_bad:
            self.X, self.y = remove_bad_data(self.X, self.y)

        if transform:
            self.X, _, _ = self.norm_images(self.X,self.contrast)

        if self.augment:
            self.X, self.y = data_augment_3d(
                list(self.X), list(self.y), factor=self.factor
            )
            # Convert back to numpy arrays after augmentation
            self.X = np.array(self.X)
            self.y = np.array(self.y)

    def __len__(self):
        return len(self.X)

    def norm_images(self, X,contrast):

        print("Normalizing data with",contrast,"contrast")

        if contrast == 'A':
            a = 13500
            b = 55500
        elif contrast == 'C':
            a = 13500
            b = 76500
        return ((X - a)/(b - a), a, b)

    def __getitem__(self, idx):
        x, y = self.X[idx], self.y[idx]
        
        x,y = torch.tensor(x[None, ...], dtype=torch.float), torch.tensor(y, dtype=torch.float)
        
        return x,y

    @staticmethod
    def split_by_job_group(full_dataset, test_frac=0.2, shuffle=True,seed=42):
        """
        Splits the dataset into training and validation sets based on job groups.
        This static method assumes 'names' attribute is available in the dataset.
        """
        X_test, y_test, X_train, y_train,names_test,names_train,_ = test_train_split_by_job_group(
            list(full_dataset.X),
            list(full_dataset.y),
            list(full_dataset.names),
            test_frac=test_frac,
            shuffle=shuffle,
            rng=np.random.default_rng(seed=seed)
        )

        # Convert lists back to datasets
        train_dataset = Microstructures(
            full_dataset.file_path,
            full_dataset.output_val,
            transform=full_dataset.transform,
            augment=False,
            remove_bad=False,
        )
        train_dataset.X, train_dataset.y = np.array(X_train), np.array(y_train)
        train_dataset.names = names_train

        val_dataset = Microstructures(
            full_dataset.file_path,
            full_dataset.output_val,
            transform=full_dataset.transform,
            augment=False,
            remove_bad=False,
        )
        val_dataset.X, val_dataset.y = np.array(X_test), np.array(y_test)
        val_dataset.names = names_test

        return train_dataset, val_dataset
