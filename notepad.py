# coding=utf-8
"""

"""

l1 = list(xrange(10))
print l1

for i in l1[-10:]:
    print i

from string import lowercase

l2 = zip(xrange(10), lowercase[:10])
print l2
print lowercase
# print l2[:5, :]



import numpy as np

A = np.array([1, 1, 1])
print (A.shape)

B = np.array([[1, 1, 1]])
print (B.shape)

import numpy as np
