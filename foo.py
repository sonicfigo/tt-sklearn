# coding=utf-8
"""

"""
import numpy as np

l1 = list(xrange(10))
print l1

for i in l1[-10:]:
    print i

from string import lowercase

l2 = zip(xrange(10), lowercase[:10])
print l2
print lowercase
# print l2[:5, :]


A = np.array([1, 1, 1])
print (A.shape)

B = np.array([[1, 1, 1]])
print (B.shape)

X = np.c_[.5, 1].T
print('\n===================1')
for _ in range(6):
    print(.1 * np.random.normal(size=(2, 1)) + X)


print('\n===================2')
print(np.random.normal(0, 1, size=(2, 1)))
