# coding=utf-8
"""

"""

from sklearn.metrics import confusion_matrix, classification_report

y_true = [2, 1, 0, 1, 2, 0]
y_pred = [2, 0, 0, 1, 2, 1]

"""
https://blog.csdn.net/m0_38061927/article/details/77198990

.混淆矩阵
混淆矩阵是机器学习中总结分类模型预测结果的情形分析表，
以矩阵形式将数据集中的记录按照真实的类别与分类模型作出的分类判断两个标准进行汇总。
这个名字来源于它可以非常容易的表明多个类别是否有混淆（也就是一个class被预测成另一个class）
"""
C = confusion_matrix(y_true, y_pred)
print(C)

"""
报告
"""
print(classification_report(y_true, y_pred))
