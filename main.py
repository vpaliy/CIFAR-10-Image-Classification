import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

import pickle as pickle
import numpy as np
import os

from softmax import SGDSoftmaxClassifier, SoftmaxClassifier


def load_data(train_paths, test_path):
    X_train, y_train = get_batch(train_paths[0])
    for i, path in enumerate(train_paths[1:]):
        data, labels = get_batch(path)
        X_train = np.concatenate([X_train, data], axis=0)
        y_train = np.concatenate([y_train, labels], axis=0)
    # load the test 
    X_test, y_test = get_batch(test_path)
    y_test = np.array(y_test)
    y_train = np.array(y_train)
    return X_train, y_train, X_test, y_test


def get_batch(path):
    with open(path, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data[b'data'], data[b'labels']


def normalize(X_train, X_test):
    mean = np.mean(X_train, axis=0)
    X_train = X_train - mean
    X_test = X_test - mean
    X_train = np.divide(X_train, 255.)
    X_test = np.divide(X_test, 255.)
    return X_train, X_test


if __name__ == '__main__':
    path = os.getcwd() + '/cifar-10-batches-py/'
    TRAIN_FILENAMES = [os.path.join(path, 'data_batch_' + str(i)) for i in range(1, 6)]
    TEST_FILENAME = os.path.join(path, 'test_batch') 

    X_train, y_train, X_test, y_test = load_data(TRAIN_FILENAMES, TEST_FILENAME)
    X_train, X_test = normalize(X_train, X_test)

    model = SGDSoftmaxClassifier()
    errors = model.fit(X_train, y_train, 5)
    
    y_pred = model.predict(X_test)


    print('Accuracy score:', accuracy_score(y_pred, y_test))

    plt.plot(range(len(errors)), errors, 'b-')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.show()

    weights = model.W[:,:-1].T
    weights = weights.reshape(10, 32, 32, 3)

    w_min, w_max = np.min(weights), np.max(weights)

    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    for i in range(10):
        plt.subplot(2, 5, i + 1)

        # Rescale the weights to be between 0 and 255 for image representation
        w_img = 255.0 * (weights[i].squeeze() - w_min) / (w_max - w_min)
        plt.imshow(w_img.astype('uint8'))
        plt.axis('off')
        plt.title(classes[i])
    plt.show()

    