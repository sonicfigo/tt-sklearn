# coding=utf-8
"""
基于 c51_face.py 的例子，想找出n_components = 多少 才是最佳的? 那就用pipe 包一层

n_components = range(100, 210, 10)，既是
[100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]

调试三个方面：
1. 参数1，PCA(svd_solver='randomized')
2. 参数2，PCA(whiten=True)
3. pca.fit 与否
对学习效果的影响：
    - 无参数，无fit：       100最好
    - 有参数1,2，无fit：    110
    - 有参数1,2，有fit：    100

实践证明，fit关系不大，主要是参数1，2，再细查下 1，2哪个更关键
实践证明：
    - 参数2，whiten=true 更关键
    - 参数1，svd_solver='randomized' 次要
    - 两者合起来，当然效果最好

"""
from pprint import pprint

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.svm import SVC

lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
target_names = lfw_people.target_names  # (7, ) 7个人名
n_classes = target_names.shape[0]  # int 7 ，7个类别，既7个人名

# n_components = range(50, 210, 30)  # 80
n_components = range(90, 210, 20)  # 110


def _prepare_data():
    # 只取lfw数据里 >70 张照片的人，也就是大部分是名人

    # introspect the images arrays to find the shapes (for plotting)
    # (1288, 50, 37)，1288张图片，每张大小 50 * 37 =1850个feature
    n_samples, h, w = lfw_people.images.shape

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

    # #############################################################################
    # Split into a training set and a test set using a stratified k fold

    # split into a training and testing set
    return train_test_split(
        X, y, test_size=0.25, random_state=42)


X_train, X_test, y_train, y_test = _prepare_data()
print('train/test: %d / %d\n' % (X_train.shape[0], X_test.shape[0]))


def fill_pipeline():
    """
    1. pca
    2. svc
    """

    # m1_pca = PCA()
    m1_pca = PCA(svd_solver='randomized', whiten=True)  # 与官网里子一致的后2个参数，否则分数很差
    # m1_pca.fit(X_train)

    m2_svc = SVC(kernel='rbf', class_weight='balanced')

    pipe = Pipeline(steps=[('pca', m1_pca),
                           ('svc', m2_svc)])
    print('\n===================原 estimator')
    pprint(pipe.named_steps)
    return pipe


def fit_gscv(pipe):
    Cs = [1e3, 5e3, 1e4, 5e4, 1e5]  # 1000最好
    gammas = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]  # 0.001最好
    gscv_pipe = GridSearchCV(pipe, dict(pca__n_components=n_components,
                                        svc__C=Cs,
                                        svc__gamma=gammas))
    gscv_pipe.fit(X_train, y_train)
    best_pac = gscv_pipe.best_estimator_.named_steps['pca']
    best_svc = gscv_pipe.best_estimator_.named_steps['svc']
    print('\n===================best estimator')
    pprint(gscv_pipe.best_estimator_.named_steps)

    print('\n===================')
    print best_pac.n_components
    print best_svc.C
    print best_svc.gamma


pipe = fill_pipeline()
fit_gscv(pipe)
