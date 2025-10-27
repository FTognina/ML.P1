import numpy as np
from helper import *#import helper functions
import list as lt
import os
import matplotlib.pyplot as plt

os.environ["OPENBLAS_NUM_THREADS"] = "8"
print("OPENBLAS_NUM_THREADS =", os.environ.get("OPENBLAS_NUM_THREADS"))

header = load_header('../data/dataset/x_test.csv')

col_list = [col for col in header[10:] if col != '_MICHD']
x_train_expanded, expanded_col_names = expand_dataset_col(col_list, '../data/dataset/x_train.csv', False)

x_train_full = x_train_expanded
y_train_full = load_data_col('../data/dataset/y_train.csv', cols=(1))

x_train,y_train,x_val,y_val = split_data_train_test_80(x_train_full, y_train_full, seed=1)
#print shapes
print("x_train shape:", x_train.shape," y_train shape:", y_train.shape)
print("x_val shape:", x_val.shape," y_val shape:", y_val.shape)

model = LogisticRegression_costum(
    lr=0.1,
    max_iter=1000,
    penalty="l2",
    alpha=0.1,
    class_weight={-1: 1.0, 1: 30.0},
    bias_init=0.5,
    verbose=True,
    early_stopping=True,
    method='stochastic',
    patience=20,
    batch_size=10000
)
print("the shape of x_train is:", x_train.shape)
print("the shape of y_train is:", y_train.shape)
print("the shape of x_val is:", x_val.shape)
print("the shape of y_val is:", y_val.shape)

model.fit(x_train, y_train, x_val, y_val)
_, threshold = model.roc_curve(x_val, y_val, plot=True)
model.evaluate(x_val, y_val, threshold=threshold)

#shows feature importance of the logistic regression model
plot_feature_importance(model, expanded_col_names, num_features=20)

#Create a csv submission file with the predictions on the test set

#x_test_sub = load_data_col('../data/dataset/x_test.csv', cols=col_index)
x_test_extended, header_test_expanded = expand_dataset_col(col_list, '../data/dataset/x_test.csv' )
x_test_extended = fill_missing_column(expanded_col_names, header_test_expanded, x_test_extended)
y_pred_probabilities = model.predict_proba(X=x_test_extended)
print("Predicted probabilities from our model:", y_pred_probabilities.flatten()[:100])
y_pred = model.predict(X=x_test_extended, threshold=threshold)

try:
    ids = load_data_col('../data/dataset/x_test.csv', cols=(0,), has_header=True).astype(int)
except Exception:
    ids = np.arange(1, x_test_extended.shape[0] + 1)
create_csv_submission(ids, y_pred, "submission.csv")