import numpy as np
import pandas as pd
import csv
from sklearn import svm
from sklearn.model_selection import train_test_split, StratifiedKFold, ParameterGrid
from sklearn.metrics import accuracy_score, roc_auc_score
from helper import *      # your custom data functions
import list as lt         # your column list file

# === Load data ===
header = load_header('../data/dataset/x_test.csv')
col_list = lt.col_list

# Expand selected columns
x_train_expanded, expanded_col_names = expand_dataset_col(
    col_list, '../data/dataset/x_train.csv', False
)

# Load full feature + target data
x_train_full = x_train_expanded
y_train_full = load_data_col('../data/dataset/y_train.csv', cols=(1))

# Train/test split
x_train, x_test, y_train, y_test = train_test_split(
    x_train_full, y_train_full, test_size=0.2, random_state=1
)

# === Parameter grid ===
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto'],
    'class_weight': [None, 'balanced'],
    'degree': [2, 3, 4],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
log_csv = "svm_progress_log.csv"

best_params = None
best_auc = -np.inf
best_acc = -np.inf

with open(log_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["iter", "C", "kernel", "gamma", "class_weight", "degree",
                "mean_acc", "std_acc", "mean_auc", "std_auc"])

    for i, params in enumerate(ParameterGrid(param_grid), start=1):
        accs, aucs = [], []
        for tr, va in cv.split(x_train, y_train):
            clf = svm.SVC(max_iter=5000, **params)
            clf.fit(x_train[tr], y_train[tr])

            y_pred = clf.predict(x_train[va])
            accs.append(accuracy_score(y_train[va], y_pred))

            # Use decision scores for AUC
            if hasattr(clf, "decision_function"):
                scores = clf.decision_function(x_train[va])
            else:
                scores = clf.predict_proba(x_train[va])[:, 1]
            try:
                aucs.append(roc_auc_score(y_train[va], scores))
            except ValueError:
                pass  # skip fold with one class

        mean_acc = np.mean(accs) if accs else np.nan
        std_acc = np.std(accs) if accs else np.nan
        mean_auc = np.mean(aucs) if aucs else np.nan
        std_auc = np.std(aucs) if aucs else np.nan

        w.writerow([
            i, params["C"], params["kernel"], params["gamma"],
            params["class_weight"], params["degree"],
            mean_acc, std_acc, mean_auc, std_auc
        ])

        # Track best (prefer AUC, break ties by accuracy)
        auc_cmp = mean_auc if not np.isnan(mean_auc) else -np.inf
        if (auc_cmp > best_auc) or (auc_cmp == best_auc and mean_acc > best_acc):
            best_auc, best_acc, best_params = auc_cmp, mean_acc, params

        print(f"Iter {i:03d}: acc={mean_acc:.3f}±{std_acc:.3f} | AUC={mean_auc:.3f}±{std_auc:.3f}")

print(f"\nSaved results to {log_csv}")
print("Best params:", best_params)
print(f"Best CV AUC={best_auc:.4f}, ACC={best_acc:.4f}")

# === Retrain best model on full training set ===
best_clf = svm.SVC(max_iter=5000, **best_params).fit(x_train, y_train)
test_pred = best_clf.predict(x_test)

# Decision function or probabilities for test AUC
if hasattr(best_clf, "decision_function"):
    test_scores = best_clf.decision_function(x_test)
else:
    test_scores = best_clf.predict_proba(x_test)[:, 1]

test_acc = accuracy_score(y_test, test_pred)
test_auc = roc_auc_score(y_test, test_scores)
print(f"\nTEST RESULTS: ACC={test_acc:.4f} | AUC={test_auc:.4f}")
