import numpy as np
from collections import Counter, namedtuple


class GaussianNaiveBayesClassifier():
    def __init__(self):
        pass
    
    def fit(self, X, y):
        self.priors_ = np.bincount(y) / len(y)
        self.n_classes_ = np.max(y) + 1

        self.means_ = np.array([X[np.where(y == i)].mean(axis=0) for i in range(self.n_classes_)])
        self.stds_ = np.array([X[np.where(y == i)].std(axis=0) for i in range(self.n_classes_)])

        return self

    def predict_proba(self, X):
        res = []
        for i in range(len(X)):
            probas = []
            for j in range(self.n_classes_):
                probas.append((1 / np.sqrt(2 * np.pi * self.stds_[j] ** 2) * np.exp(
                    -0.5 * ((X[i] - self.means_[j]) / self.stds_[j]) ** 2)).prod() * self.priors_[j])
            probas = np.array(probas)
            res.append(probas / probas.sum())

        return np.array(res)

    def predict(self, X):
        res = self.predict_proba(X)

        return res.argmax(axis=1)


class LR():
    def __init__(self, lr=0.01, steps=5000):
        self.lr = lr
        self.steps = steps

    def s(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        self.coefs = np.zeros(X.shape[1])  # weights

        for _ in range(self.steps):
            h = self.s(np.dot(X, self.coefs))
            self.coefs -= self.lr * \
                          np.dot(X.T, (h - y)) / y.size  # gradient  step

    def predict(self, X):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        return self.s(np.dot(X, self.coefs)).round()


class KNN():
    def __init__(self, nn=5):
        self.nn = nn

    def dists(self, X):
        num_test = X.shape[0]
        num_train = self.X.shape[0]

        t = np.dot(X, self.X.T)
        dists = np.sqrt(-2 * t + np.square(self.X).sum(1) +
                        np.matrix(np.square(X).sum(1)).T)
        return dists

    def fit(self, X, y):  
        self.X = X
        self.y = y

    def predict(self, X):
        dists = self.dists(X)
        preds = np.zeros(dists.shape[0])

        for i in range(dists.shape[0]):
            labels = self.y[np.argsort(dists[i, :])].flatten()
            top_nn_y = labels[:self.nn]
            preds[i] = Counter(top_nn_y).most_common(1)[0][0]
        return preds


class Node():
    def __init__(self, predicted_class):
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None
