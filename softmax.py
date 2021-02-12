import numpy as np

from base import RidgeRegularization


def softmax(scores):
    scores -= np.max(scores)
    return (np.exp(scores).T / np.sum(np.exp(scores), axis=1)).T
    

_default_reg = RidgeRegularization()


class SoftmaxClassifier(object):
    def __init__(self, regularization=_default_reg):
        self.regularization = regularization
    
    def fit(self, X, y, epochs=10, eta=1, tol=0.001):
        X = np.insert(X, 0, 1, axis=1)
        n_classes = len(np.unique(y))
        n_features = X.shape[1]
        # create the weights matrix
        self.W = np.zeros((n_features, n_classes))
        y = self.one_hot(y, n_classes)
        errors = []
        # do epochs
        for _ in range(epochs):
            z = X.dot(self.W)
            probs = softmax(z)
            error = -np.mean(y * np.log(np.max(probs))) + self.regularization(self.W)
            grad = (-1.0 / X.shape[0]) * np.dot(X.T, (y-probs)) + self.regularization.grad(self.W)
            errors.append(error)
            self.W = self.W - eta*grad
            if np.linalg.norm(grad) <= tol:
                break
        return errors

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        return np.argmax(softmax(X.dot(self.W)), 1)

    def one_hot(self, y, n_classes):
        y = np.asarray(y, dtype='int32')
        if len(y) > 1:
            y = y.reshape(-1)
        if not n_classes:
            n_classes = np.max(y) + 1
        y_matrix = np.zeros((len(y), n_classes))
        y_matrix[np.arange(len(y)), y] = 1
        return y_matrix
    


class SGDSoftmaxClassifier(SoftmaxClassifier):
    def fit(self, X, y, epochs=10, eta=1e-1, tol=1e-5, batch_size=200):
        N = len(y)
        if not batch_size or batch_size > N:
            batch_size = int(0.1 * N)
        n_classes = len(np.unique(y))
        self.W = np.zeros((X.shape[1]+1, n_classes))
        errors = []
        y = self.one_hot(y, n_classes)
        for epoch in range(epochs):
            indices = np.random.permutation(N)
            X = X[indices]
            y = y[indices]
            for i in range(0, N, batch_size):
                Xi = X[i:i+batch_size]
                yi = y[i:i+batch_size]
                Xi = np.insert(Xi, 0, 1, axis=1)
                z = Xi.dot(self.W)
                probs = softmax(z)
                grad = (-1.0 / batch_size) * np.dot(Xi.T, (yi-probs)) + self.regularization.grad(self.W)
                self.W = self.W - eta*grad 
            Xi = np.insert(X, 0, 1, axis=1)
            error = -np.mean(y * np.log(softmax(Xi.dot(self.W)))) + self.regularization(self.W)
            errors.append(error)
            print('Epoch:{0}  Error:{1}'.format(epoch+1, error))
        return errors

