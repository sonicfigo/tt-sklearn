# coding=utf-8
"""
real-world数据里，经常需要降维。若将多个 feature extraction 方式，联合在一起是有好处的
本例子用 FeatureUnion ，合并两个降维类：
- PCA
- univariate selection


未读懂输出的那些信息，怎么比对高分怎么来的?
"""
from pprint import pprint

from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

iris = load_iris()

X, y = iris.data, iris.target


def build_feature_stacker():
    # This dataset is way too high-dimensional. Better do PCA:
    pca = PCA(n_components=2)

    # Maybe some original features where good, too?
    selection = SelectKBest(k=1)

    # Build estimator from PCA and Univariate selection:
    combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])

    # Use combined features to transform dataset:
    X_features = combined_features.fit(X, y).transform(X)

    # print(X.shape)  # (150, 3)
    # print(X_features.shape)  # (150, 3)
    return combined_features


svm = SVC(kernel="linear")

# Do grid search over k, n_components and C:
combined_features = build_feature_stacker()
pipeline = Pipeline([("features", combined_features),
                     ("svm", svm)])

param_grid = dict(features__pca__n_components=[1, 2, 3],
                  features__univ_select__k=[1, 2],
                  svm__C=[0.1, 1, 10])

grid_search = GridSearchCV(pipeline, param_grid=param_grid, verbose=10)
grid_search.fit(X, y)
pprint(grid_search.best_estimator_)
