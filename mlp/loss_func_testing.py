import numpy as np
from LossFunctions import LossFunction

C, m = 10, 256
rng = np.random.default_rng(0)
Y = np.zeros((C,m)); Y[rng.integers(C, size=m), np.arange(m)] = 1
Z = rng.normal(size=(C,m))
Z -= Z.max(axis=0, keepdims=True)
A = np.exp(Z) / np.exp(Z).sum(axis=0, keepdims=True)

L = LossFunction.cross_entropy_loss(Y, A)
print("CE ~", L)           # should be ~2.3 for random preds
print("sum probs cols ~1:", np.allclose(A.sum(axis=0),1,atol=1e-6))
