import gzip
import os
import numpy as np

train_data = os.path.join("data", "mnist", "train-images-idx3-ubyte.gz")
train_labels = os.path.join("data", "mnist", "train-labels-idx1-ubyte.gz")

test_data = os.path.join("data", "mnist", "t10k-images-idx3-ubyte.gz")
test_labels = os.path.join("data", "mnist", "t10k-labels-idx1-ubyte.gz")

def mnist_images(filename):
    with gzip.open(filename, "rb") as file:
        data = np.frombuffer(file.read(), np.uint8, offset=16)
        return data.reshape(-1, 28, 28)

x_train = mnist_images(train_data)

matloplib

