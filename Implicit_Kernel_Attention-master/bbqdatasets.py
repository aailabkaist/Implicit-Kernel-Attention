import pandas as pd
import os
import numpy as np

import matplotlib.image as mpimg

DEFAULT_DIR = os.path.join("bbq", "datasets")


def mauna_loa(rootDir=DEFAULT_DIR, raw_data=False,
              **readcsvkwargs):
    """
    Loads the Mauna Loa C02 dataset from 1965 to 2016
    (years with complete data...)
    :param rootDir:
    :param readcsvkwargs:
    :return:
    """
    if raw_data:
        file_path = os.path.join(rootDir, "raw_datasets",
                                 "mauna-loa-c02-1965-2016.csv")
        dataset = pd.read_csv(file_path, usecols=[2, 3], skiprows=54,
                              engine="python", **readcsvkwargs)
        data = np.array(dataset.values)
        return data
    else:
        file_path = os.path.join(rootDir, "co2")
        train = np.genfromtxt(os.path.join(file_path, "train.csv"),
                              delimiter=",")
        test = np.genfromtxt(os.path.join(file_path, "test.csv"),
                             delimiter=",")
        return train, test


def airline_passengers(rootDir=DEFAULT_DIR, raw_data=False,
                       **readcsvkwargs):
    """
    Loads the "international-airline-passenger" dataset from
    https://datamarket.com/data/set/22u3/international-airline-passengers-monthly-totals-in-thousands-jan-49-dec-60#!ds=22u3&display=line
    :param rootDir:
    :param readcsvkwargs:
    :return:
    """
    if raw_data:
        file_path = os.path.join(rootDir, "raw_datasets",
                                 "international-airline-passengers.csv")
        dataset = pd.read_csv(file_path, usecols=[1], engine='python',
                              skipfooter=3,
                              **readcsvkwargs)
        data_raw = np.hstack([np.array(dataset.index).reshape(-1, 1),
                              np.array(dataset.values).reshape(-1, 1)])
        return data_raw
    else:
        file_path = os.path.join(rootDir, "airline_passengers")
        train = np.genfromtxt(os.path.join(file_path, "train.csv"),
                              delimiter=",")
        test = np.genfromtxt(os.path.join(file_path, "test.csv"),
                             delimiter=",")
        return train, test


def concrete(rootDir=DEFAULT_DIR, raw_data=False,
             **readcsvkwargs):
    """
    Loads the "Concrete Compressive Strength" dataset from
    https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength
    :param rootDir:
    :param readcsvkwargs:
    :return:
    """
    if raw_data:
        file_path = os.path.join(rootDir, "raw_datasets",
                                 "Concrete_Data-1.csv")
        dataset = pd.read_csv(file_path, engine='python',
                              **readcsvkwargs)
        return np.array(dataset.values)
    else:
        file_path = os.path.join(rootDir, "concrete")
        train = np.genfromtxt(os.path.join(file_path, "train.csv"),
                              delimiter=",")
        test = np.genfromtxt(os.path.join(file_path, "test.csv"),
                             delimiter=",")
        return train, test


def airfoil_noise(rootDir=DEFAULT_DIR, raw_data=False,
                  **readcsvkwargs):
    """
    Loads the "Airfoil self-noise" dataset from
    https://archive.ics.uci.edu/ml/datasets/Airfoil+Self-Noise
    :param rootDir:
    :param readcsvkwargs:
    :return:
    """
    if raw_data:
        file_path = os.path.join(rootDir, "raw_datasets",
                                 "airfoil_self_noise.csv")
        dataset = pd.read_csv(file_path, engine='python',
                              **readcsvkwargs)
        return np.array(dataset.values)
    else:
        file_path = os.path.join(rootDir, "airfoil_noise")
        train = np.genfromtxt(os.path.join(file_path, "train.csv"),
                              delimiter=",")
        test = np.genfromtxt(os.path.join(file_path, "test.csv"),
                             delimiter=",")
        return train, test


def textures_2D(rootDir=DEFAULT_DIR, texture_name="pores", raw_data=False,
                **readcsvkwargs):
    """
    TOTAL = 1690
    TRAIN: 12675
    TEST: 4225

    res: (130 across, 130 up)
    i.e. (130,130)

    we have a cutout of (65,65)

    #1    (0 to 129 , 130)  then  (0 to 31, 32)
    #2    (0 to 31, 32)     then  (32 to 96, 65)
    #3    (97 to 129,  33)  then  (32 to 96,  65)
    #4    (0 to 129,  130)   then  (97 to 129, 33)
    """
    if raw_data:
        rgb_img = mpimg.imread(os.path.join(rootDir, "raw_datasets",
                                            '{}.png'.format(texture_name)))
        gimg = rgb2gray(rgb_img)

        # The training set
        xtrn1 = np.mgrid[
               0:129:complex(0, 130),
               0:31:complex(0, 32)].reshape(2, -1).T.astype(np.int)
        xtrn2 = np.mgrid[
               0:31:complex(0, 32),
               32:96:complex(0, 65)].reshape(2, -1).T.astype(np.int)
        xtrn3 = np.mgrid[
               97:129:complex(0, 33),
               32:96:complex(0, 65)].reshape(2, -1).T.astype(np.int)
        xtrn4 = np.mgrid[
               0:129:complex(0, 130),
               97:129:complex(0, 33)].reshape(2, -1).T.astype(np.int)

        X_trn = np.vstack((xtrn1, xtrn2, xtrn3, xtrn4))

        X_tst = np.mgrid[
               32:96:complex(0, 65),
               32:96:complex(0, 65)].reshape(2, -1).T.astype(np.int)

        Y_trn = gimg[X_trn[:, 0], X_trn[:, 1]].reshape(-1, 1)
        Y_tst = gimg[X_tst[:, 0], X_tst[:, 1]].reshape(-1, 1)

        return X_trn, X_tst, Y_trn, Y_tst
    else:
        file_path = os.path.join(rootDir, "textures_2D", texture_name)
        train = np.genfromtxt(os.path.join(file_path, "train.csv"),
                              delimiter=",")
        test = np.genfromtxt(os.path.join(file_path, "test.csv"),
                             delimiter=",")
        return train, test

def rgb2gray(rgb):
   """
   Convert an mpimg into grayscale
   :param rgb:
   :return:
   USAGE
   >> import matplotlib.pyplot as plt
   >> import matplotlib.image as mpimg
   >> img = mpimg.imread('image.png')
   >> gray = rgb2gray(img)
   >> plt.imshow(gray, cmap=plt.get_cmap('gray'))
   >> plt.show()
   """
   return np.dot(rgb[:, :, :3], [0.299, 0.587, 0.114])
