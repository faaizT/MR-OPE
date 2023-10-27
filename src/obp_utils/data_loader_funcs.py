import pandas as pd
import numpy as np
from keras.datasets import mnist
from sklearn.datasets import load_digits


def load_letter_data():
    data = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data",
        header=None,
    )
    X = data.iloc[:, 1:].to_numpy()
    y = pd.factorize(data.iloc[:, 0])[0]
    return X, y


def load_mnist_data():
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    X, y = np.concatenate((train_X, test_X), axis=0), np.concatenate(
        (train_y, test_y), axis=0
    )
    X = X.reshape(X.shape[0], -1)
    return X, y


def load_data_optdigits():
    data_train = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra",
        header=None,
    )
    data_test = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes",
        header=None,
    )
    X = np.concatenate(
        [data_train.iloc[:, :-1].to_numpy(), data_test.iloc[:, :-1].to_numpy()], axis=0
    )
    y = np.concatenate(
        [
            pd.factorize(data_train.iloc[:, -1])[0],
            pd.factorize(data_test.iloc[:, -1])[0],
        ],
        axis=0,
    )
    return X, y


def load_data_yeast():
    data = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data",
        sep="\s+",
        header=None,
    )
    X = data.iloc[:, 1:-1].to_numpy()
    y = pd.factorize(data.iloc[:, -1])[0]
    return X, y


def load_data_satimage():
    data_train = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/satimage/sat.trn",
        sep="\s+",
        header=None,
    )
    data_test = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/satimage/sat.tst",
        sep="\s+",
        header=None,
    )
    X = np.concatenate(
        [data_train.iloc[:, :-1].to_numpy(), data_test.iloc[:, :-1].to_numpy()], axis=0
    )
    y = np.concatenate(
        [
            pd.factorize(data_train.iloc[:, -1])[0],
            pd.factorize(data_test.iloc[:, -1])[0],
        ],
        axis=0,
    )
    return X, y


def load_data_pendigits():
    data_train = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.tra",
        header=None,
    )
    data_test = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.tes",
        header=None,
    )
    X = np.concatenate(
        [data_train.iloc[:, :-1].to_numpy(), data_test.iloc[:, :-1].to_numpy()], axis=0
    )
    y = np.concatenate(
        [
            pd.factorize(data_train.iloc[:, -1])[0],
            pd.factorize(data_test.iloc[:, -1])[0],
        ],
        axis=0,
    )
    return X, y


def load_data_ecoli():
    data = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/ecoli/ecoli.data",
        sep="\s+",
        header=None,
    )
    X = data.iloc[:, 1:-1]
    y = pd.factorize(data.iloc[:, -1])[0]
    return X, y


def load_data(dataset):
    if dataset == "digits":
        X, y = load_digits(return_X_y=True)
    elif dataset == "mnist":
        X, y = load_mnist_data()
    elif dataset == "letter":
        X, y = load_letter_data()
    elif dataset == "yeast":
        X, y = load_data_yeast()
    elif dataset == "satimage":
        X, y = load_data_satimage()
    elif dataset == "pendigits":
        X, y = load_data_pendigits()
    elif dataset == "optdigits":
        X, y = load_data_optdigits()
    else:
        raise ValueError(
            "dataset must be one of ['digits', 'mnist', 'letter', 'yeast', 'satimage', 'pendigits', 'optdigits']"
        )
    return X, y
