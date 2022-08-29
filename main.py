import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import RobustScaler
import sys

FILES = ["cifar-10-python/cifar-10-batches-py/data_batch_1",
         "cifar-10-python/cifar-10-batches-py/data_batch_2",
         "cifar-10-python/cifar-10-batches-py/data_batch_3",
         "cifar-10-python/cifar-10-batches-py/data_batch_4",
         "cifar-10-python/cifar-10-batches-py/data_batch_5"]

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_training_data(n):  # n is the number of examples batches that we will use to train our MLP
    try:
        if 1 <= n <= 5:
            data = []
            classes = []
            for i in range(n):
                batch = unpickle(FILES[i])
                data.append(batch[b'data'])
                classes.extend(batch[b'labels'])
            RGB_pixels = data[0]
            for j in range(len(data)-1):
                RGB_pixels = np.vstack((RGB_pixels, data[j+1]))
            return RGB_pixels, np.array(classes)
        else:
            raise BaseException("Incorrect number of training batches!\n"
                                "(CIFAR-10 dataset has 5 training batches)")
    except BaseException as e:
        print(e)
        sys.exit(1)

def load_test_data():
    test = unpickle("cifar-10-python/cifar-10-batches-py/test_batch")
    test_RGB_pixels = np.array(test[b'data'])
    test_classes = np.array(test[b'labels'])
    return test_RGB_pixels, test_classes

def normalize_data(train_pixels, test_pixels):  # such that σ^2 = 1 and μ = 0
    sc = RobustScaler()
    scaler = sc.fit(train_pixels)
    train_pixels_scaled = scaler.transform(train_pixels)
    test_pixels_scaled = scaler.transform(test_pixels)
    return train_pixels_scaled, test_pixels_scaled

