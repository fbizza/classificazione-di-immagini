import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
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

def load_training_data(n):
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
            raise BaseException("Incorrect number of training batches! \n(CIFAR-10 dataset has 5 training batches)")
    except BaseException as e:
        print(e)
        sys.exit(1)

x, y = load_training_data(5)

