# coding=utf-8
"""
官网例子，pca + svc 完整过程
总结:
数据：7个人，共 1288 张照片，size为 高50, 宽37
步骤：2个，降维和学习

===================================================
pca 降维
    只需要用到 X_train，既只需要fit样本，没有关于predict的任何操作，相当于非监督学习的概念

1. 随机 966 张 data，作为 X_train，shape 就是（966， 1850）
    - 维度有 1850 个，既是 50 * 37 个像素点

2. 指定一个PCA，降维成 150
    - 不要拿 966 张的 * 1850 个点做 feature 来训练，高维，太细腻
    - 而是总结 966 * 1850(想象那个知乎pca贴的gif) ，降维，揉成 150 个人脸的块特征做 feature 来训练

3. PCA 具备了降维能力
    - 966张照片得来的 150 本征脸，从数据上看，是来自fit后降维的 pca.components_ 这个属性
    - pca.components_，shape是(150, 1850)，既 （新feature数，老feature数），体现了新老feature的pca关系
    - 这个聪明的 PCA 后续可以去精简 1850 feature 成 150 个feature，给gscv 做 train 和 predict用

既是 PCA 通过这7个人，966张照片，学习到了7个人的本征脸（eigenfaces）信息


===================================================
SVM + gscv 学习并预测
    预测这7个人剩下的 322 张照片，成功率高哦

1. 原 X_train 和 X_test 数据，都用聪明的 pca 先降维
2. gscv 对 SVC 模型调参 'C' 和 'gamma'
3. 学习966张 pca 过的 X_train
4. 预测322张 pca 过的 X_test

gscv 知识点
    - fit： pca过的 X_train, y_train
    - 调参数： C 和 gamma
    - predict： pca过的 X_test




原注释
=========================================================================================================================================================
===================================================
Faces recognition example using eigenfaces and SVMs
===================================================

The dataset used in this example is a preprocessed excerpt of the
"Labeled Faces in the Wild", aka LFW_:

  http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz (233MB)

.. _LFW: http://vis-www.cs.umass.edu/lfw/

Expected results for the top 5 most represented people in the dataset:

================== ============ ======= ========== =======
                   precision    recall  f1-score   support
================== ============ ======= ========== =======
     Ariel Sharon       0.67      0.92      0.77        13
     Colin Powell       0.75      0.78      0.76        60
  Donald Rumsfeld       0.78      0.67      0.72        27
    George W Bush       0.86      0.86      0.86       146
Gerhard Schroeder       0.76      0.76      0.76        25
      Hugo Chavez       0.67      0.67      0.67        15
       Tony Blair       0.81      0.69      0.75        36

      avg / total       0.80      0.80      0.80       322
================== ============ ======= ========== =======

照片下载目录
/Users/figo/scikit_learn_data/lfw_home/

"""

from time import time
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC

# region 准备数据

# 只取lfw数据里 >70 张照片的人，也就是大部分是名人
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# introspect the images arrays to find the shapes (for plotting)
# (1288, 50, 37)，1288张图片，每张大小 50 * 37 =1850个feature
n_samples, h, w = lfw_people.images.shape

target_names = lfw_people.target_names  # (7, ) 7个人名
n_classes = target_names.shape[0]  # int 7 ，7个类别，既7个人名

"""
两个问题：
1. 什么n才是最好的
2. 最大只能是 966，也就是分得很细，7个人的 966张照片，都变成966类，但predict不了，这么大取值作罢。

太大或太小都会有问题
"""
n_components = 150  # 原来有 1850 个features, 只提取 150 个components


def _prepare_data():
    # for machine learning we use the 2 data directly
    # (as relative pixel positions info is ignored by this model)
    X = lfw_people.data  # (1288, 1850)
    n_features = X.shape[1]  # 1850

    # the label to predict is the id of the person
    y = lfw_people.target  # (1288,)

    print("Total dataset size:")
    print("n_samples:  %d 张照片" % n_samples)  # 1288
    print("n_features: %d 个feature" % n_features)  # 1850
    print("n_classes:  %d 个人(既类别)" % n_classes)  # 7

    return train_test_split(X, y, test_size=0.25, random_state=42)

# X_train (966, 1850)
X_train, X_test, y_train, y_test = _prepare_data()
print('train/test: %d / %d\n' % (X_train.shape[0], X_test.shape[0]))


# endregion

# region 核心2步骤：pca extract 及 predict test数据
def pca_extract_eigenfaces():
    """
    Compute a PCA (eigenfaces) on the face dataset
    (treated as unlabeled dataset):
    unsupervised feature extraction / dimensionality reduction
    """

    print("""###########################################################################
    Extracting the top %d eigenfaces from %d faces""" % (n_components, X_train.shape[0]))
    t0 = time()
    pca = PCA(n_components=n_components,
              svd_solver='randomized',
              whiten=True
              )
    # fit 与否，与gscv无关系，而是与pca.components_ 及 pca.transform(...)有关系
    pca.fit(X_train)
    print("done in %0.3fs" % (time() - t0))

    # pca.components_， fit 后是 (150, 1850)
    # reshape 转为 -> (150类, 50, 37)，用来展示图片
    eigenfaces = pca.components_.reshape((n_components, h, w))

    return pca, eigenfaces  # (150, 50, 37)


def predict_by_gscv_svc(pca):
    """
    使用 PCA 转换出数据 X_pca
    X_train_pca + y_train 给 gscv fit ，自动选到 best estimator
    X_test_pca            给 gscv 进行 predict，得到y_predict
    """
    print("""###########################################################################
    Projecting the input data on the eigenfaces orthonormal basis""")
    t0 = time()
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    print("done in %0.3fs" % (time() - t0))

    print("""###########################################################################
    Train a SVM classification model
    Fitting the classifier to the training set""")
    t0 = time()
    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                  'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    gscv_svc = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
    gscv_svc = gscv_svc.fit(X_train_pca, y_train)
    print("done in %0.3fs" % (time() - t0))

    print("Best estimator found by grid search:")
    print(gscv_svc.best_estimator_)

    print("""###########################################################################
    Predicting people's names on the test set
    Quantitative evaluation of the model quality on the test set""")
    t0 = time()
    y_predict = gscv_svc.predict(X_test_pca)
    print("done in %0.3fs" % (time() - t0))

    # 打报告
    print(classification_report(y_test, y_predict, target_names=target_names))

    print('\n=================== 混淆矩阵，总结分类模型预测结果')
    print('列head：预测值，行head：真实类，对角线数量越多，既是正确的数量越多')
    print(confusion_matrix(y_test, y_predict, labels=range(n_classes)))

    return y_predict


pca, eigenfaces_150_50_37_to_show_only = pca_extract_eigenfaces()

y_pred = predict_by_gscv_svc(pca)


# endregion

# region 辅助show照片

# Qualitative evaluation of the predictions using matplotlib
def _plot_gallery(images, titles, h, w, n_row=6, n_col=6):
    """
    Helper function to plot a gallery of portraits

    :param images: (322张, 1850数据)， 1850 要打成 50 * 37 才可以显示图片
    :param titles:
    :param h:
    :param w:
    :param n_row:   几行照片
    :param n_col:   几列照片
    :return:
    """
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)

        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


def _title(y_pred, y_test, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    # return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)
    return 'XXXXXX' if pred_name != true_name else ''


# endregion

# region show照片
def plot_origin_face_and_predict_result():
    """
    原始脸,清晰
    plot the result of the prediction on a portion of the test set
    X_test (322, 1850)
    """
    prediction_titles = [_title(y_pred, y_test, i)
                         for i in range(y_pred.shape[0])]
    _plot_gallery(X_test, prediction_titles, h, w)


plot_origin_face_and_predict_result()


# 纯粹用来展示下966 张照片，extract出来的本征脸，长得什么样
def plot_eigenfaces():
    """
    本征脸，模糊
    plot the gallery of the most significative eigenfaces
    eigenfaces (150, 50, 37)
    """
    eigenface_titles = ["eigenface %d" % i for i in
                        range(eigenfaces_150_50_37_to_show_only.shape[0])]
    _plot_gallery(eigenfaces_150_50_37_to_show_only, eigenface_titles, h, w)


# plot_eigenfaces()

# endregion

plt.show()
