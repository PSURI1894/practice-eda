"""A neural-network regressor written from scratch in NumPy — forward pass, backpropagation,
and the Adam optimizer — so the mechanics are fully visible (and it runs without PyTorch,
which Smart App Control blocks on this machine).

It plugs into the Part-5 forecasting pipeline: `.fit(X, y)` / `.predict(X)` with the same
interface as the LightGBM model, so `ml_forecast.recursive_forecast` drives it unchanged.
"""
from __future__ import annotations

import numpy as np


def _relu(z):
    return np.maximum(0, z)


def _relu_grad(z):
    return (z > 0).astype(z.dtype)


class MLPRegressor:
    """Multilayer perceptron (fully-connected feedforward net) for regression.

    hidden : sizes of the hidden layers, e.g. (32, 16) = two layers.
    Trains by mini-batch gradient descent with Adam; inputs and target are standardized
    internally (essential for stable neural-net training).
    """

    def __init__(self, hidden=(32, 16), lr=0.01, epochs=400, batch=32,
                 l2=1e-4, seed=0):
        self.hidden = tuple(hidden)
        self.lr, self.epochs, self.batch, self.l2, self.seed = lr, epochs, batch, l2, seed
        self.loss_history_ = []

    # ---- internals -------------------------------------------------------
    def _init_params(self, n_in):
        rng = np.random.default_rng(self.seed)
        sizes = [n_in, *self.hidden, 1]
        self.W, self.b = [], []
        for i in range(len(sizes) - 1):
            # He initialisation (good for ReLU): variance 2/fan_in
            self.W.append(rng.normal(0, np.sqrt(2 / sizes[i]), (sizes[i], sizes[i + 1])))
            self.b.append(np.zeros(sizes[i + 1]))

    def _forward(self, X):
        acts, pre = [X], []                      # a0 = input
        a = X
        for i in range(len(self.W)):
            z = a @ self.W[i] + self.b[i]
            pre.append(z)
            a = z if i == len(self.W) - 1 else _relu(z)   # linear output, ReLU hidden
            acts.append(a)
        return acts, pre

    # ---- API -------------------------------------------------------------
    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y, float).ravel()
        self.x_mu, self.x_sd = X.mean(0), X.std(0) + 1e-8
        self.y_mu, self.y_sd = y.mean(), y.std() + 1e-8
        Xs = (X - self.x_mu) / self.x_sd
        ys = ((y - self.y_mu) / self.y_sd).reshape(-1, 1)

        self._init_params(Xs.shape[1])
        mW = [np.zeros_like(w) for w in self.W]; vW = [np.zeros_like(w) for w in self.W]
        mb = [np.zeros_like(b) for b in self.b]; vb = [np.zeros_like(b) for b in self.b]
        b1, b2, eps = 0.9, 0.999, 1e-8
        rng = np.random.default_rng(self.seed)
        n = Xs.shape[0]; t = 0

        for _ in range(self.epochs):
            idx = rng.permutation(n)
            ep_loss = 0.0
            for s in range(0, n, self.batch):
                bi = idx[s:s + self.batch]
                xb, yb = Xs[bi], ys[bi]
                acts, pre = self._forward(xb)
                yhat = acts[-1]
                ep_loss += np.sum((yhat - yb) ** 2)
                # backprop (MSE)
                dz = 2 * (yhat - yb) / xb.shape[0]
                t += 1
                for i in reversed(range(len(self.W))):
                    dW = acts[i].T @ dz + self.l2 * self.W[i]
                    db = dz.sum(0)
                    if i > 0:
                        dz = (dz @ self.W[i].T) * _relu_grad(pre[i - 1])
                    # Adam update
                    for p, g, m, v in [(self.W[i], dW, mW[i], vW[i]), (self.b[i], db, mb[i], vb[i])]:
                        m[...] = b1 * m + (1 - b1) * g
                        v[...] = b2 * v + (1 - b2) * g * g
                        mhat = m / (1 - b1 ** t); vhat = v / (1 - b2 ** t)
                        p -= self.lr * mhat / (np.sqrt(vhat) + eps)
            self.loss_history_.append(ep_loss / n)
        return self

    def predict(self, X):
        X = np.asarray(X, float)
        Xs = (X - self.x_mu) / self.x_sd
        ys = self._forward(Xs)[0][-1].ravel()
        return ys * self.y_sd + self.y_mu
