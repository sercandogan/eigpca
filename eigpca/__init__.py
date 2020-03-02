import numpy as np
import matplotlib.pyplot as plt


class PCA(object):
    def __init__(self, corr=True):
        self.corr = corr

    def fit_transform(self, X, n_components=None):
        self.fit(X)
        return self.transform(X, n_components)

    def transform(self, X, n_components=None):
        if X.shape[1] != self.p_:
            raise ValueError(f"X must have {self.p_} variables. ")
        if n_components and (n_components > self.p_ or n_components < 1):
            ValueError("Components cannot be greater than number of variables or less than 1")
        if not n_components:
            return np.dot(X, self.eig_vecs[:, self.eig_vals > 1])
        else:
            return np.dot(X, self.eig_vecs[:, :n_components])

    def fit(self, X):
        X = self._check_shape(X)
        self.p_ = X.shape[1]  # variables
        self.n_ = X.shape[0]  # observations

        xs = X - X.mean(axis=0)  # standardised
        S = np.dot(xs.T, xs) / (self.n_ - 1)  # variance-covariance matrix
        if self.corr:
            # calculate correlation with matrix multiplication
            d = S.diagonal().T * np.identity(self.p_)
            d_sqrt_inv = (1 / np.sqrt(d).diagonal()).T * np.identity(self.p_)
            corr = np.dot(np.dot(d_sqrt_inv, S), d_sqrt_inv)

            eig_vals, eig_vecs = np.linalg.eig(corr)
        else:
            eig_vals, eig_vecs = np.linalg.eig(S)

        idx = eig_vals.argsort()[::-1]
        self.eig_vals = eig_vals[idx]
        self.eig_vecs = eig_vecs[:, idx]
        self.explained_variance_ratio_ = self.eig_vals / self.p_

    def _check_shape(self, X):
        X = np.array(X)
        if X.ndim != 2:
            raise ValueError("X must be 2 dimensional")

        if X.shape[0] < X.shape[1]:
            raise ValueError("Number of observations must be greater than number of variables (n > p)")

        if X.dtype not in ["float64", "float32", "int64", "int32"]:
            raise ValueError("dtype must be one among float64, float32, int64, int32")

        return X

    def plot(self, y="eig"):
        if y == "eig":
            plt.plot(range(1, self.p_ + 1), self.eig_vals, 'o-')
            plt.xticks(np.arange(1, self.p_ + 1, step=1))
            plt.xlabel("Component Number")
            plt.ylabel("Eigenvalue")
            plt.title("Scree Plot")
        elif y == "pov":
            plt.plot(range(1, self.p_ + 1), self.explained_variance_ratio_, 'o-')
            plt.xticks(np.arange(1, self.p_ + 1, step=1))
            plt.xlabel("Component Number")
            plt.ylabel("Propotion of Variance")
            plt.title("Scree Plot")
        else:
            raise ValueError("y must be only eig, pov")
