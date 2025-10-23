import sklearn
from sklearn import svm
clf = svm.SVC()
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
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto'],
    'class_weight': [None, 'balanced'],
    'degree': [2, 3, 4]
}
grid_search = GridSearchCV(clf, param_grid, cv=5)
grid_search.fit(x_train, y_train)
print("Best Parameters:", grid_search.best_params_)
