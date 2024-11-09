from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm
import numpy as np


class SyntheticGen:
    def __init__(self, n_neighbors=5):
        self.nn = NearestNeighbors(n_neighbors=n_neighbors)

    def fit(self, X, y):
        self.nn.fit(X, y)
        return self

    def generate(self, X, y, n_samples=None):
        n_samples = n_samples if n_samples is not None else X.shape[0]

        X_synthetic = np.zeros((n_samples, X.shape[1]))
        y_synthetic = np.zeros(n_samples)

        for i in tqdm(range(n_samples), desc='Generating'):
            idx = np.random.randint(0, len(X))
            neighbors = self.nn.kneighbors([X[idx]], return_distance=False)[0]
            neighbor_idx = np.random.choice(neighbors[1:])
            alpha = np.random.rand()
            X_new = X[idx] + alpha * (X[neighbor_idx] - X[idx])
            X_synthetic[i, :] = X_new
            y_synthetic[i] = y[idx]
        return np.vstack((X, X_synthetic)), np.hstack((y, y_synthetic))
