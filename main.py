import numpy as np
import pandas
import gzip
import copy
from NeuralNet import NeuralNet


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

train_args = load_training_images('D:/Projects/Data/train-images-idx3-ubyte.gz')
train_labels = load_training_labels('D:/Projects/Data/train-labels-idx1-ubyte.gz')
test_args = load_training_images('D:/Projects/Data/t10k-images-idx3-ubyte.gz')
test_labels = load_training_labels('D:/Projects/Data/t10k-labels-idx1-ubyte.gz')
n = NeuralNet([15, 10], 784)
x = copy.copy(train_args.reshape(train_args.shape[0], -1)).astype(np.float64)
x /= (255/2)
x -= 1

x_test = copy.copy(test_args.reshape(test_args.shape[0], -1)).astype(np.float64)
x_test /= (255/2)
x_test -= 1

n.train(x, train_labels)

count = 0
for i in range(len(test_labels)):
    prediction = n.predict(x_test[i, :].reshape(1, 784))
    if prediction == test_labels[i]:
        count += 1

print(count/len(test_labels))
