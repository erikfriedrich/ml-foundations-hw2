"""
Skeleton code for creating the Random Kitchen Sinks (RKS) and 1-hidden-layer MLP (MLP) models.

Note you are *welcome* to completely ignore this file and use your own code, but it's here if you want to use it as a starting point.
"""
import numpy as np
from typing import List, Tuple

import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

"""
For Random Kitchen Sinks (RKS)
"""
class RandomKitchenSinks:
    """
    Random Kitchen Sinks featurization with ridge regression.
    """
    def __init__(self, D: int = 1000, sigma: float = 1.0, ridge_lambda: float = 1e-3, d: int = 2, random_state: int = 0):
        """
        Potential hyper-parameters you can consider (feel free to add more):
            D: total number of random features (= 2h in the assignment text; if you use both sin and cos features, D should be an even number)
            sigma: scale parameter for the RBF kernel, used for initializing the weight parameters
            ridge_lambda: ridge regularization strength
            d: dimension of the covariates
            random_state: seed for drawing the random features
        """
        # h pairs of feature functions (sin and cos) => 2h = D
        assert D % 2 == 0, 'D must be even (h sin/cos pairs)'

        self.D = D
        self.sigma = sigma
        self.ridge_lambda = ridge_lambda

        rng = np.random.default_rng(random_state)
        h = D // 2

        # b_j ~ U[0, 2 pi] and w_j ~ N(0, I / sigma^2)
        self.b = rng.uniform(low = 0, high = 2 * np.pi, size = h)                 # (h,)
        self.w = rng.multivariate_normal(mean = np.zeros(d),
                                         cov = np.identity(d) / self.sigma ** 2,
                                         size = h)                                 # (h, d)

    def _features(self, X: np.ndarray) -> np.ndarray:
        """
        Convert the input data X into the random kitchen sinks features. You will likely need to use the parameters initialized in the __init__ method.
        The resulting features should be of shape (n, D or D[+1]), where the plus one is for the intercept term.
        """
        w_transpose = np.transpose(self.w)
        b_tiled = np.tile(self.b, (len(X), 1))

        lin_comb = np.matmul(X, w_transpose) + b_tiled                            # (n, h)

        # no intercept column: Ridge fits an (unpenalized) intercept itself
        Z = np.concatenate([np.sin(lin_comb), np.cos(lin_comb)], axis = 1)       # (n, D)

        return Z

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the ridge regression model to the (1) features computed from the input data X and (2) target values y.
        """
        phi = self._features(X)  # shape (n, D)
        self.model_regression = Ridge(alpha = self.ridge_lambda)
        self.model_regression.fit(phi, y)
        self.coef_ = self.model_regression.coef_
        self.intercept_ = self.model_regression.intercept_

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the target values for the input data X.
        """
        phi = self._features(X)  # shape (n, D)
        return phi @ self.coef_ + self.intercept_  # shape (n,)

def sweep_rks_model(sweep_sigma: List[float], X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> Tuple:
    """
    Sweeps over the kernel scale sigma (D and ridge_lambda fixed at their defaults), it then returns the best model in terms of validation MSE.
    Args:
        sweep_sigma: list of kernel scales to sweep over
        X_train: training input data
        y_train: training target values
        X_val: validation input data
        y_val: validation target values
    Returns:
        a tuple (best_model, best_val_mse, best_sigma, train_mses, val_mses)
    """
    best_model = None
    best_val_mse = float('inf')
    best_sigma = None
    train_mses, val_mses = [], []

    for sigma in sweep_sigma:
        model = RandomKitchenSinks(D = 1000, sigma = sigma, d = X_train.shape[1])
        model.fit(X_train, y_train)

        train_mse = mean_squared_error(y_train, model.predict(X_train))
        val_mse = mean_squared_error(y_val, model.predict(X_val))
        train_mses.append(train_mse)
        val_mses.append(val_mse)

        if val_mse < best_val_mse:
            best_model = model
            best_val_mse = val_mse
            best_sigma = sigma

    return best_model, best_val_mse, best_sigma, train_mses, val_mses

"""
For 1-hidden-layer MLP (MLP)
"""
class MLP(nn.Module):
    """1-hidden-layer MLP for regression."""
    def __init__(self, d_in: int, d_hidden: int = 128, d_out: int = 1, lr: float = 1e-3, batch_size: int = 128, num_epochs: int = 500, activation: str = "relu", device: str = "cpu", random_state: int = 0):
        """
        Potential hyper-parameters you can consider (feel free to add more):
            d_hidden: hidden dimension
            activation: activation function
            lr: learning rate
            batch_size: batch size
            num_epochs: number of passes over the training data
        """
        super().__init__()
        # CPU by default, so that the results do not depend on whether a GPU is available
        # (GPU and CPU round slightly differently; the CPU is not slower for a network this small)
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        torch.manual_seed(random_state)  # same initial weights for every configuration
        activations = {"relu": nn.ReLU(), "tanh": nn.Tanh()}
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            activations[activation],
            nn.Linear(d_hidden, d_out)
        )
        self.to(self.device)

    def fit(self, X: np.ndarray, y: np.ndarray):
        # turn the numpy data into torch tensors and move to the device
        X = torch.from_numpy(X).float().to(self.device)
        y = torch.from_numpy(y).float().view(-1, 1).to(self.device)  # (n, 1), matches the network output

        self.net.train()  # make sure to set the model to training mode
        optimizer = torch.optim.Adam(self.net.parameters(), lr = self.lr)
        loss_fn = nn.MSELoss()

        for epoch in range(self.num_epochs):
            perm = torch.randperm(len(X), device = self.device)

            for start in range(0, len(X), self.batch_size):
                idx = perm[start:start + self.batch_size]

                optimizer.zero_grad()
                loss = loss_fn(self.forward(X[idx]), y[idx])

                loss.backward()
                optimizer.step()

        return self

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.net(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            X = torch.from_numpy(X).float().to(self.device)
            return self.forward(X).detach().cpu().numpy().ravel()  # shape (n,)

def mlp_sweep(sweep_values: List[int], X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> Tuple:
    """
    Sweeps over the batch size (all other hyper-parameters fixed at their defaults), it then returns the best model in terms of validation MSE.
    Returns:
        a tuple (best_model, best_val_mse, best_batch_size, train_mses, val_mses)
    """
    best_model = None
    best_val_mse = float('inf')
    best_batch_size = None
    train_mses, val_mses = [], []

    for batch_size in sweep_values:
        model = MLP(d_in = X_train.shape[1], batch_size = batch_size)
        model.fit(X_train, y_train)

        train_mse = mean_squared_error(y_train, model.predict(X_train))
        val_mse = mean_squared_error(y_val, model.predict(X_val))
        train_mses.append(train_mse)
        val_mses.append(val_mse)
        print(f'batch size {batch_size}: train MSE {train_mse:.4f}, val MSE {val_mse:.4f}')

        if val_mse < best_val_mse:
            best_model = model
            best_val_mse = val_mse
            best_batch_size = batch_size

    return best_model, best_val_mse, best_batch_size, train_mses, val_mses
