# Lesson2(6个chapter)

> A tutorial on statistical-learning for scientific data processing，数据处理
- http://scikit-learn.org/stable/tutorial/statistical_inference/index.html


## chapter1
> Statistical learning: the setting and the estimator object in scikit-learn
- 基础知识
- sklearn学习时的fit数据一定是要2d的，遇到2d以上的，要转为2d
- estimator 是关键，就是model

## chapter2
> Supervised learning: predicting an output variable from high-dimensional observations
- 各种具体model的学习，预测例子
- knn，linear regression，logistic regression
- svm
    - kernel： lineral, rbf, poly
    - LinearSVC
    - decision boundaries，普通的分界线
    - decision function，带有 margin 的分界函数
- 数据处理
    - shrinkage
    - sparsity
- 多类别分类

## chapter3
> Model selection: choosing estimators and their parameters
- 模型的参数自动选择，手工调整，手工验证等

## chapter4
> Unsupervised learning: seeking representations of the data
- 非监督学习
- Clustering: grouping observations together
    - 目标：已知要分为N类，进行observations的聚类
    - VQ：vector quantization， 数据压缩，使用类似KNN(k=n)的方式，把零散的数据，归到n类中
    - hierarchical clustering：
        1. agglomerative
            - 最终会生成一个 connectivity graph
            - feature agglomerative，用特征聚类，解决数据少，feature多的窘境
        2. divisive
    
- Decompositions: from a signal to components and loadings
    -TODO  选择并提取feature中最关键的成分，略过，PCA 大致了解了，ICA还未细读

## chapter5
> Putting it all together
- Pipelining
- Face recognition with eigenfaces

- Open problem: Stock Market Structure，略过，不看
