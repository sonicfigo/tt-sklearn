# coding=utf-8
"""
如果不经过pca，成功率会是多少？

答案：很差
"""
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC

lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
target_names = lfw_people.target_names  # (7, ) 7个人名
n_classes = target_names.shape[0]  # int 7 ，7个类别，既7个人名


def _prepare_data():
    X = lfw_people.data  # (1288, 1850)
    y = lfw_people.target  # (1288,)
    return train_test_split(X, y, test_size=0.25, random_state=42)


X_train, X_test, y_train, y_test = _prepare_data()


def predict_by_gscv_svc():
    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                  'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    gscv_svc = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
    gscv_svc = gscv_svc.fit(X_train, y_train)

    print(gscv_svc.best_estimator_)

    y_predict = gscv_svc.predict(X_test)
    print(classification_report(y_test, y_predict, target_names=target_names))
    print(confusion_matrix(y_test, y_predict, labels=range(n_classes)))


predict_by_gscv_svc()
