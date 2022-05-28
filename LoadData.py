import numpy as np
import gzip
from copy import copy


def load_training_images(file):
    with gzip.open(file, 'r') as f:
        # first 4 bytes is a magic number
        magic_number = int.from_bytes(f.read(4), 'big')
        # second 4 bytes is the number of images
        image_count = int.from_bytes(f.read(4), 'big')
        # third 4 bytes is the row count
        row_count = int.from_bytes(f.read(4), 'big')
        # fourth 4 bytes is the column count
        column_count = int.from_bytes(f.read(4), 'big')
        # rest is the image pixel data, each pixel is stored as an unsigned byte
        # pixel values are 0 to 255
        image_data = f.read()
        images = np.frombuffer(image_data, dtype=np.uint8)\
            .reshape((image_count, row_count, column_count))
        return images


def load_training_labels(file):
    with gzip.open(file, 'r') as f:
        # first 4 bytes is a magic number
        magic_number = int.from_bytes(f.read(4), 'big')
        # second 4 bytes is the number of labels
        label_count = int.from_bytes(f.read(4), 'big')
        # rest is the label data, each label is stored as unsigned byte
        # label values are 0 to 9
        label_data = f.read()
        labels = np.frombuffer(label_data, dtype=np.uint8)
        return labels


def prepare_data(x, y):
    labels = np.zeros((len(y), 10))
    for i in range(len(y)):
        labels[i, y[i]] = 1

    args = copy(x.reshape(x.shape[0], -1)).astype(np.float64)
    args /= (255 / 2)
    args -= 1

    return args, labels


def load_data(path='D:/Projects/Data/'):
    train_args = load_training_images(path + 'train-images-idx3-ubyte.gz')
    train_labels = load_training_labels(path + 'train-labels-idx1-ubyte.gz')
    test_args = load_training_images(path + 't10k-images-idx3-ubyte.gz')
    test_labels = load_training_labels(path + 't10k-labels-idx1-ubyte.gz')
    return train_args, train_labels, test_args, test_labels
