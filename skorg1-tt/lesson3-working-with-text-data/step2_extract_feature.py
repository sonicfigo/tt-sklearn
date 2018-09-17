# coding=utf-8
"""
Extracting features from text files

In order to perform machine learning on text documents, we first need to turn the text content into numerical feature vectors.

把文字，转成机器学习可用的 数字 feature 向量

存储方案：
- 所有documents：doc1, doc2, ..., doc2257

- 所有documents出现过的word，标上一个index
j1: hello
j2: world

- 每个document 出现的频率
    - hello
        - X[doc1, j1] = 10
        - X[doc2, j1] = 5
    - world 出现的频率
        - X[doc1, j2] = 10
        - X[doc2, j2] = 3

估算此方案的存储大小：
- 5000 个documents
- 100，000 个 distinct word
- 存储为 float32，既4 * 8位， 4bytes

总量为 5000 * 100,000 * 4 ≈ 2G

"""
