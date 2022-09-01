import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import sys

FILES = ["cifar-10-python/cifar-10-batches-py/data_batch_1",
         "cifar-10-python/cifar-10-batches-py/data_batch_2",
         "cifar-10-python/cifar-10-batches-py/data_batch_3",
         "cifar-10-python/cifar-10-batches-py/data_batch_4",
         "cifar-10-python/cifar-10-batches-py/data_batch_5"]

CLASSES = ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]


##--------------DATA LOADING--------------##
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
            for j in range(len(data) - 1):
                RGB_pixels = np.vstack((RGB_pixels, data[j + 1]))
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


##--------------DATA PRE-PROCESSING--------------##
def normalize_data(train_pixels, test_pixels):  # such that σ = 1 and μ = 0
    sc = StandardScaler()
    scaler = sc.fit(train_pixels)
    train_pixels_scaled = scaler.transform(train_pixels)
    test_pixels_scaled = scaler.transform(test_pixels)
    return train_pixels_scaled, test_pixels_scaled


##--------------TRAINING and PREDICTION--------------##
def train_and_predict(n):  # n is the number of examples batches that we will use to train our MLP
    train_RGB_pixels, train_classes = load_training_data(n)
    test_RGB_pixels, test_classes = load_test_data()
    x, y = normalize_data(train_RGB_pixels, test_RGB_pixels)
    clf = MLPClassifier((50,), tol=0.000001, activation="relu", max_iter=20,
                        solver="adam", n_iter_no_change=20, verbose=True)
    print("START OF TRAINING")
    clf.fit(x, train_classes)
    print("END OF TRAINING\n")
    predictions = clf.predict(y)
    return predictions, test_classes, clf


##--------------EVALUATION and PLOTS--------------##
def evaluate(predictions, truth, clf):
    accuracy = accuracy_score(predictions, truth)
    print('Accuracy: {:.2f}\n'.format(accuracy))
    report = classification_report(y_true=truth, y_pred=predictions, target_names=CLASSES)
    print(report)
    loss_vector = clf.loss_curve_
    x_axis = list(range(0, len(loss_vector), 1))
    plt.plot(x_axis, loss_vector)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss curve")
    ConfusionMatrixDisplay.from_predictions(y_true=truth, y_pred=predictions, display_labels=CLASSES,
                                            cmap="BuGn", colorbar=False, xticks_rotation="vertical")
    plt.show()


##--------------MAIN PROGRAM--------------##
predictions, truth, clf = train_and_predict(5)
evaluate(predictions, truth, clf)
