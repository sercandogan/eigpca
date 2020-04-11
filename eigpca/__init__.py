import numpy as np
import matplotlib.pyplot as plt


class PCA(object):
    """
    Principal component analysis (PCA)
    PCA is defined as an orthogonal linear transformation that transforms
    the data to a new coordinate system which ıs used for feature reduction.

    It provides via eigen-decomposition of the covariance/correlation matrix
    on the contrary of sckit-learn which does via SVD.

    Parameters:
    ----------
    corr: bool, default: True
        if True, the eigen-decomposition will be calculated
        using correlation matrix.

    Attributes:
    ----------
    p_: int
        Number of variables
    n_: int
        Number of observations
    eig_vals: array
        Eigen values of the matrix
    eig_vecs: matrix
        Eigen vector-matrix of the matrix
    explained_variance_ratio_: array
        The amount of variance explained by each of components

    Examples:
    ---------
        from eigpca import PCA
        X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
        pca = PCA()
        pca.fit(X)
        pca.transform(X, n_components=1)
    """

    def __init__(self, corr=True):

        self.corr = corr

    def fit_transform(self, X, n_components=None):
        """
            Fit and apply dimensionality reduction
        """
        self.fit(X)
        return self.transform(X, n_components)

    def transform(self, X, n_components=None):
        """
        Apply dimensionality reduction

        Parameters
        ----------
        X: array-like, shape (n_observations(n), n_variables(p))
            The data
        n_components: int, default: None
            Number of components
            If it's None, then it will be set number of eigen values that are greater than 1.
            Reason: Those with eigenvalues less than 1.00 are not considered to be stable.
            They account for less variability than does a single variable and are not retained in the analysis.
            In this sense, you end up with fewer factors than original number of variables. (Girden, 2001)
            reference: https://stats.stackexchange.com/a/80318

        Returns
        -------
        X_new: array-like, shape (n_observations, n_components)
            Transformed values



        """
        X = self._check_shape(X)

        if X.shape[1] != self.p_:
            raise ValueError(f"X must have {self.p_} variables. ")
        if n_components and (n_components > self.p_ or n_components < 1):
            ValueError("Components cannot be greater than number of variables or less than 1")
        if not n_components:
            return np.dot(X, self.eig_vecs[:, self.eig_vals > 1])
        else:
            return np.dot(X, self.eig_vecs[:, :n_components])

    def fit(self, X):
        """
            Eigen-decomposition of X

            Parameters
            ----------
            X: array-like, shape (n_observations(n), n_variables(p))
                The data

        """
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
        """
        Check shape of data
        """
        X = np.array(X)
        if X.ndim != 2:
            raise ValueError("X must be 2 dimensional")

        if X.shape[0] < X.shape[1]:
            raise ValueError("Number of observations must be greater than number of variables (n > p)")

        if X.dtype not in ["float64", "float32", "int64", "int32"]:
            raise ValueError("dtype must be one among float64, float32, int64, int32")

        return X

    def plot(self, y="eig"):
        """
        Scree plot for deciding number of components
        Parameters
        ----------
        y: str, eig or pov
            y axis
                eig: Eigen value
                pov: Propotion of variance

        """
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
