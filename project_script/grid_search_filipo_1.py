#import libraries
import numpy as np
from project_script.helper import *#import helper functions
import project_script.list as lt
from datetime import datetime
from pathlib import Path

LOG_PATH_ALL = Path(f"grid_search_all_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt")

def write_log_iteration(f1, thresh, col_list, lr, alpha, class_weight_1, method, with_missing):
    """Append progress for every iteration."""
    with LOG_PATH_ALL.open("a", encoding="utf-8") as log_file:
        log_file.write(
            f"[ITER] F1={f1:.5f}, Thresh={thresh:.3f}, "
            f"Cols={col_list}, LR={lr}, Alpha={alpha}, CW1={class_weight_1}, "
            f"Method={method}, WithMissing={with_missing}\n"
        )
        log_file.flush()  # ensures write even if interrupted


def write_log(f1,thresh, col_list, lr, alpha, class_weight_1, method, with_missing):
    #log file with date and time in the file name
    with open(f"grid_search_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt", "a") as log_file:
        log_file.write(f"Best F1 Score: {f1:.4f}, Threshold: {thresh:.2f}, Columns: {col_list}, Learning Rate: {lr}, Alpha: {alpha}, Class Weight 1: {class_weight_1}\n")

#grid search for hyperparameters, returns at the end the best model found
best_f1_overall = 0
best_thresh_overall = 0
best_col_list = None
best_lr = 0
best_alpha = 0
best_class_weight_1 = 0
best_method = ''
best_with_missing = False
try:
    for with_missing in [True, False]:  # Just to keep the structure for future use
        for method in ['stochastic', 'full_batch']:
            for col_num, col_list in enumerate([lt.col_list, lt.col_list_2, lt.col_list_3]):
                header = load_header('../data/dataset/x_test.csv')
                x_train_expanded, expanded_col_names = expand_dataset_col(col_list, '../data/dataset/x_train.csv', with_missing=with_missing )
                x_train_full = x_train_expanded#load_data_column('x_train_expanded.csv', column_list=None)
                y_train_full = load_data_col('../data/dataset/y_train.csv', cols=(1))
                x_train,y_train,x_test,y_test = split_data_train_test_80(x_train_full, y_train_full, seed=1)
                for  lr in [0.1, 0.01, 0.001]:
                    for alpha in [0.1, 0.01, 0.001]:
                        for class_weight_1 in [10.0, 20.0, 30.0]:
                            print(f"Training with method={method}, col_list number={col_num}, lr={lr}, alpha={alpha}, class_weight_1={class_weight_1}")
                            model = LogisticRegression_costum(
                                lr=lr,
                                max_iter=1000,
                                penalty="elasticnet",
                                alpha=alpha,
                                class_weight={-1: 1.0, 1: class_weight_1},
                                bias_init=0.5,
                                verbose=False,
                                early_stopping=True,
                                method=method,
                                patience=20
                            )
                            model.fit(x_train, y_train, x_test, y_test)
                            y_pred = model.predict(x_test, threshold=0.5)
                            temp_f1, temp_thresh = model.roc_curve(x_test, y_test, plot=False)
                            write_log_iteration(temp_f1, temp_thresh, col_list, lr, alpha, class_weight_1, method, with_missing)

                            if temp_f1 > best_f1_overall:
                                best_f1_overall = temp_f1
                                best_thresh_overall = temp_thresh
                                best_col_list = col_list
                                best_lr = lr
                                best_alpha = alpha
                                best_class_weight_1 = class_weight_1
                                best_method = method
                                best_with_missing = with_missing
                                print(f"New Best F1 Score: {best_f1_overall:.4f} at threshold {best_thresh_overall:.2f}")
                            #log every iteration
except KeyboardInterrupt:
    print("Grid Search Interrupted")
    print(f"Best F1 Score Overall: {best_f1_overall:.4f} at threshold {best_thresh_overall:.2f}")
    print(f"Best Column List: {best_col_list}")
    print(f"Best Learning Rate: {best_lr}")
    print(f"Best Alpha: {best_alpha}")
    print(f"Best Class Weight 1: {best_class_weight_1}")
    print(f"Best Method: {best_method}")
    print(f"Best With Missing: {best_with_missing}")
    #save it on log file
    write_log(best_f1_overall, best_thresh_overall, best_col_list, best_lr, best_alpha, best_class_weight_1, best_method, best_with_missing)
    
print("Grid Search Complete")
print(f"Best F1 Score Overall: {best_f1_overall:.4f} at threshold {best_thresh_overall:.2f}")
print(f"Best Column List: {best_col_list}")
print(f"Best Learning Rate: {best_lr}")
print(f"Best Alpha: {best_alpha}")
print(f"Best Class Weight 1: {best_class_weight_1}")
print(f"Best Method: {best_method}")
print(f"Best With Missing: {best_with_missing}")
#save it on log file
write_log(best_f1_overall, best_thresh_overall, best_col_list, best_lr, best_alpha, best_class_weight_1, best_method, best_with_missing)

