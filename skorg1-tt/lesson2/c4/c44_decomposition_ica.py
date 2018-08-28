# coding=utf-8
"""
略过，pca有看过概念，ica不看了先略过
"""

# Generate sample data
import numpy as np
from scipy import signal

from sklearn import decomposition

time = np.linspace(0, 10, 2000)  # 0 ~ 10, 2000均分点

s1 = np.sin(2 * time)  # Signal 1 : sinusoidal signal
s2 = np.sign(np.sin(3 * time))  # Signal 2 : square signal
s3 = signal.sawtooth(2 * np.pi * time)  # Signal 3: saw tooth signal
S = np.c_[s1, s2, s3]
S += 0.2 * np.random.normal(size=S.shape)  # Add noise
S /= S.std(axis=0)  # Standardize data

# Mix data
A = np.array([[1, 1, 1], [0.5, 2, 1], [1.5, 1, 2]])  # Mixing matrix
X = np.dot(S, A.T)  # Generate observations

# Compute ICA
ica = decomposition.FastICA()
S_ = ica.fit_transform(X)  # Get the estimated sources
A_ = ica.mixing_.T
is_foo = np.allclose(X, np.dot(S_, A_) + ica.mean_)
print(is_foo)
