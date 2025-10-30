import numpy as np
from helper import *#import helper functions
import list as lt
import os
import matplotlib.pyplot as plt
from datetime import datetime

os.environ["OPENBLAS_NUM_THREADS"] = "8"
print("OPENBLAS_NUM_THREADS =", os.environ.get("OPENBLAS_NUM_THREADS"))

path = './reports/'
if not os.path.exists(path):
    os.makedirs(path)
timestamp = path + str(datetime.now().strftime("%Y%m%d_%H%M%S"))

header = load_header('../data/dataset/x_test.csv')

col_list = [col for col in header[11:] if col != '_MICHD']

x_train_expanded, expanded_col_names = check_saved_extended_dataset(col_list, '../data/dataset/x_train_expanded_missing.csv' )


x_train_full = x_train_expanded
y_train_full = load_data_col('../data/dataset/y_train.csv', cols=(1))

x_train,y_train,x_val,y_val = split_data_train_test_80(x_train_full, y_train_full, seed=1)

print("x_train shape:", x_train.shape," y_train shape:", y_train.shape)
print("x_val shape:", x_val.shape," y_val shape:", y_val.shape)

model = LogisticRegression_costum(
    lr=0.05,
    max_iter=1000,
    penalty="elasticnet",
    alpha=0.05,
    class_weight={-1: 1.0, 1: 30.0},
    bias_init=1.0,
    verbose=True,
    early_stopping=True,
    method='batch',
    patience=20,
    batch_size=8192
)
print("the shape of x_train is:", x_train.shape)
print("the shape of y_train is:", y_train.shape)
print("the shape of x_val is:", x_val.shape)
print("the shape of y_val is:", y_val.shape)

plot_loss = model.fit(x_train, y_train, x_val, y_val)
plot_loss.savefig(str(timestamp) +'_loss_plot.png', bbox_inches='tight')

best_f1, threshold, plot_roc = model.roc_curve(x_val, y_val, plot=True)
plot_roc.savefig(str(timestamp) +'_roc_curve.png', bbox_inches='tight')

try:
    evaluation = model.evaluate(x_val, y_val, threshold)
    print(evaluation)
except Exception as e:
    print("Error during evaluation:", e)
    print("Threshold value:", threshold)
    print("y_val shape:", y_val.shape)
    print("x_val shape:", x_val.shape)
    evaluation = "Evaluation failed."

#shows feature importance of the logistic regression model
plot_feature = plot_feature_importance(model, expanded_col_names, nr_features=20)
plot_feature.savefig(str(timestamp) +'_feature_importance.png', bbox_inches='tight')
#plot_feature.show()
#Create a csv submission file with the predictions on the test set

#x_test_sub = load_data_col('../data/dataset/x_test.csv', cols=col_index)
x_test_extended, header_test_expanded = expand_dataset_col(col_list, '../data/dataset/x_test.csv' )
x_test_extended = fill_missing_column(expanded_col_names, header_test_expanded, x_test_extended)
y_pred_probabilities = model.predict_proba(X=x_test_extended)

y_pred = model.predict(X=x_test_extended, threshold=threshold)

create_csv_submission(y_pred, timestamp + "_submission.csv")
#pack the x,y,ypred values
values = {
    "x_train": x_train,
    "y_train": y_train,
    "x_val": x_val,
    "y_val": y_val,
    "y_pred_probabilities": y_pred_probabilities
}
save_report(model, evaluation, values, timestamp)

