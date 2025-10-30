import sklearn
from sklearn import svm
from sklearn.model_selection import train_test_split
import numpy as np
from project_script.helper import *#import helper functions
import project_script.list as lt

header = load_header('../data/dataset/x_test.csv')

col_list = lt.col_list  # Example: take first 10 columns to expand

x_train_expanded, expanded_col_names = expand_dataset_col(col_list, '../data/dataset/x_train.csv', False)
x_train_full = x_train_expanded#load_data_column('x_train_expanded.csv', column_list=None)
y_train_full = load_data_col('../data/dataset/y_train.csv', cols=(1))

x_train, x_test, y_train, y_test = train_test_split(x_train_full, y_train_full, test_size=0.2, random_state=1)

clf = svm.SVC(max_iter=2000)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("SVM Classifier Accuracy:", accuracy)
#ROC AUC
from sklearn.metrics import roc_auc_score
roc_auc = roc_auc_score(y_test, y_pred)
print("SVM Classifier ROC AUC:", roc_auc)
from sklearn.metrics import classification_report
print("Classification Report:\n", classification_report(y_test, y_pred))
#grid search for svm using sklearn
from sklearn.model_selection import GridSearchCV
#lots of parameters to try
param_grid = {
    'C': [0.5], #[0.1, 1, 10],
    'kernel': ['poly'],#['linear', 'rbf', 'poly'],
    'gamma': ['auto'], #['scale', 'auto'],
    'class_weight': ['balanced'], #[None, 'balanced'],
    'degree': [2,3] #[2, 3, 4]
}
grid_search = GridSearchCV(clf, param_grid, cv=5)
grid_search.fit(x_train, y_train)
print("Best Parameters:", grid_search.best_params_)
