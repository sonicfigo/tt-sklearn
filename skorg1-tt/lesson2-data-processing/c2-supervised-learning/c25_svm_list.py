# coding=utf-8
"""

"""
from sklearn import svm

print('\n=================== SVC')
print(svm.SVC())

for fig_num, kernel in enumerate(['linear', 'rbf', 'poly']):
    print('\n=================== %s' % kernel)
    print(svm.SVC(kernel=kernel))

print('\n=================== LinearSVC')
print(svm.LinearSVC())
