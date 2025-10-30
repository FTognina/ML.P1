from helper import *#import helper functions
import os

os.environ["OPENBLAS_NUM_THREADS"] = "14"
print("OPENBLAS_NUM_THREADS =", os.environ.get("OPENBLAS_NUM_THREADS"))

path = './reports_grid/'
if not os.path.exists(path):
    os.makedirs(path)

header = load_header('../data/dataset/x_test.csv')

col_list = [col for col in header[11:] if col != '_MICHD']

def run(class_weight,penalty, with_missing, nr_run):
    path = './reports_grid/run_' + str(nr_run)+'_'
    if with_missing:
        x_train_expanded, expanded_col_names = expand_dataset_col(col_list, '../data/dataset/x_train.csv', with_missing)
    else:
        x_train_expanded, expanded_col_names = check_saved_extended_dataset(col_list, '../data/dataset/x_train_expanded.csv')
    x_train_full = x_train_expanded
    y_train_full = load_data_col('../data/dataset/y_train.csv', cols=(1))

    x_train,y_train,x_val,y_val = split_data_train_test_80(x_train_full, y_train_full, seed=1)

    model = LogisticRegression_costum(
        lr=0.1,
        max_iter=1000,
        penalty=penalty,
        alpha=0.1,
        class_weight={-1: 1.0, 1: class_weight},
        bias_init=0.5,
        verbose=True,
        early_stopping=True,
        patience=60,
    )
    print("x_train shape:", x_train.shape," y_train shape:", y_train.shape)
    print("x_val shape:", x_val.shape," y_val shape:", y_val.shape)

    plot_loss = model.fit(x_train, y_train, x_val, y_val)
    plot_loss.savefig(str(path) +'loss_plot.png', bbox_inches='tight')

    best_f1, threshold, plot_roc = model.roc_curve(x_val, y_val, plot=True)
    plot_roc.savefig(str(path) +'roc_curve.png', bbox_inches='tight')
    try:
        evaluation = model.evaluate(x_val, y_val, threshold)
        print(evaluation)
    except Exception as e:
        print("Error during evaluation:", e)
        evaluation = "Evaluation failed."
    
    #shows feature importance of the logistic regression model
    plot_feature = plot_feature_importance(model, expanded_col_names, nr_features=20)
    plot_feature.savefig(str(path) +'feature_importance.png', bbox_inches='tight')
    #Create a csv submission file with the predictions on the test set

    #x_test_sub = load_data_col('../data/dataset/x_test.csv', cols=col_index)
    x_test_extended, header_test_expanded = expand_dataset_col(col_list, '../data/dataset/x_test.csv' )
    x_test_extended = fill_missing_column(expanded_col_names, header_test_expanded, x_test_extended)
    y_pred_probabilities = model.predict_proba(X=x_test_extended)

    y_pred = model.predict(X=x_test_extended, threshold=threshold)


    ids = load_data_col('../data/dataset/x_test.csv', cols=(0,), has_header=True).astype(int)

    create_csv_submission(ids, y_pred, str(path)+"submission.csv")
    #pack the x,y,ypred values
    values = {
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "y_pred_probabilities": y_pred_probabilities
    }
    save_report(model, evaluation, values, path)
    #calculate accuracy
    y_val_pred = model.predict(X=x_val, threshold=threshold)
    acuracy = np.mean(y_val_pred == y_val)
    return best_f1, acuracy

#do a grid search only over class_weights [1.0,5.0,10.0,20.0,30.0,40.0,50.0], then plot f1 score and accuracy vs class weight
class_weights = [1.0,5.0,10.0,20.0,30.0,40.0,50.0]
f1_scores = []
acuracies = []
for i, cw in enumerate(class_weights):
    f1, acuracy = run(class_weight=cw, penalty="l2", with_missing=False, nr_run=str(i)+"_classweight_"+str(cw))
    f1_scores.append(f1)
    acuracies.append(acuracy)
#plot f1 scores vs class weights
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(class_weights, f1_scores, marker='o', color='blue', linewidth=2, markersize=8, label='F1 Score')
plt.plot(class_weights, acuracies, marker='x', color='red', linewidth=2, markersize=8, label='Accuracy')
plt.xlabel("Class Weight", fontsize=12)
plt.ylabel("Score", fontsize=12)
plt.title("F1 Score and Accuracy vs Class Weight", fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.ylim([0, 1])  # Scores are between 0 and 1
plt.tight_layout()
plt.savefig(path+'metrics_vs_class_weight.png', bbox_inches='tight', dpi=300)
plt.close()

#do a grid search only over penalty ['l1', 'l2','elasticnet'], then plot (f1 score and accuracy) vs penalty
penalties = ['l1', 'l2','elasticnet']
f1_scores_penalty = []
acuracies_penalty = []
for i, pen in enumerate(penalties):
    f1, acuracy = run(class_weight=30.0, penalty=pen, with_missing=False, nr_run=str(i)+"_penalty_"+pen)
    f1_scores_penalty.append(f1)
    acuracies_penalty.append(acuracy)
#plot f1 scores vs penalties
plt.figure(figsize=(10, 6))
plt.plot(penalties, f1_scores_penalty, marker='o', color='blue', linewidth=2, markersize=8, label='F1 Score')
plt.plot(penalties, acuracies_penalty, marker='x', color='red', linewidth=2, markersize=8, label='Accuracy')
plt.xlabel("Penalty Type", fontsize=12)
plt.ylabel("Score", fontsize=12)
plt.title("F1 Score and Accuracy vs Penalty Type", fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.ylim([0, 1])  # Scores are between 0 and 1
plt.tight_layout()
plt.savefig(path+'metrics_vs_penalty.png', bbox_inches='tight', dpi=300)
plt.close()

#do a grid search only over with_missing [True, False], then plot (f1 score and accuracy) vs with_missing
with_missing_options = [True, False]
f1_scores_missing = []
acuracies_missing = []
for i, with_missing in enumerate(with_missing_options):
    f1, acuracy = run(class_weight=30.0, penalty='l2', with_missing=with_missing, nr_run=str(i)+"_with_missing_"+str(with_missing))
    f1_scores_missing.append(f1)
    acuracies_missing.append(acuracy)
#plot f1 scores vs with_missing
plt.figure(figsize=(10, 6))
plt.plot(with_missing_options, f1_scores_missing, marker='o', color='blue', linewidth=2, markersize=8, label='F1 Score')
plt.plot(with_missing_options, acuracies_missing, marker='x', color='red', linewidth=2, markersize=8, label='Accuracy')
plt.xlabel("With Missing", fontsize=12)
plt.ylabel("Score", fontsize=12)
plt.title("F1 Score and Accuracy vs With Missing", fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.ylim([0, 1])  # Scores are between 0 and 1
plt.tight_layout()
plt.savefig(path+'metrics_vs_with_missing.png', bbox_inches='tight', dpi=300)
plt.close()