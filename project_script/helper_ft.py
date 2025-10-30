import sys
import numpy as np
import csv
import matplotlib.pyplot as plt
import os

#todo
#rng = np.random.default_rng()
#sub and over sampling


np.random.seed(1)

def load_data_col(path, cols=(0), has_header=True):
    """Load numeric CSV with missing values -> np.nan using only numpy.
    path: path to CSV file
    cols: list of column indices to load (default: 0)
    has_header: whether the CSV file has a header row (default: True)

    returns data with nan for missing values
    """
    skip = 1 if has_header else 0
    data = np.genfromtxt(
        path,
        delimiter=",",
        skip_header=1 if has_header else 0,
        usecols=(cols),  # adjust based on relevant columns]       # force float to accommodate np.nan
        missing_values=["", "NA", "NaN"],       # treat empty strings as missing missing_values=["", "NA", "NaN"]
        filling_values=np.nan,
        autostrip=True,
        invalid_raise=False
    )
    return data

def load_header(path):
    """Load column names from CSV header.
    path: path to CSV file
    return: list of column names in the header
    """
    with open(path, 'r') as f:
        header = f.readline().strip()
    return header.split(',')

def get_col_index(cols, header):
    """Get the index of a column given its name from a list of column names.
    cols: list of column names to find
    header: list of all column names
    """
    for name in cols:
        if name not in header:
            print(f"Column '{name}' not found in the provided list.")
    return [name for name in cols if name in header], [header.index(name) for name in cols if name in header]

def expand_column(col, col_name, with_missing=True):
    '''
    Expand a 1D column vector into a N array where N is the number of unique values in col.
    Each column in the output array is a binary indicator (0 or 1) of whether the corresponding
    entry in col matches the unique value for that column.
    if the number of unique values is greater than 10, the column is standardized instead.
    col: 1D numpy array of categorical values
    col_name: name of the column
    with_missing: whether to handle missing values
    #Missing values are handled by adding an additional binary indicator for nans column and setting them to 0.
    return: expanded 2D numpy array and list of new column names
    '''

    is_missing = np.isnan(col).astype(np.int8)
    has_missing = np.any(is_missing)
    unique_values = np.unique(col)[~np.isnan(np.unique(col))]

    if with_missing:
        if unique_values.size > 10:
            if not has_missing:
                col = (col - col.mean()) / col.std()
                return col.reshape(-1, 1).astype(np.float16), [col_name]
            col = np.where(np.isnan(col), np.nanmean(col), col)
            col = (col - col.mean()) / col.std()
            expanded = np.column_stack([col, is_missing])
            col_names = np.append([col_name], [f"{col_name}_is_missing"])
        else:
            expanded = np.zeros((col.size, unique_values.size), dtype=np.int8)
            for i, val in enumerate(unique_values):
                expanded[:, i] = (col == val).astype(np.int8)
            col_names = [f"{col_name}_{val}" for val in unique_values]
            if has_missing:
                expanded = np.column_stack([expanded, is_missing])
                col_names.append(f"{col_name}_is_missing")
        return expanded, col_names
    else:
        col = np.where(np.isnan(col), np.nanmean(col), col)
        if unique_values.size > 10:
            col = (col - col.mean()) / col.std()
            return col.reshape(-1, 1).astype(np.float16), [col_name]
        else:
            expanded = np.zeros((col.size, unique_values.size), dtype=np.int8)
            for i, val in enumerate(unique_values):
                expanded[:, i] = (col == val).astype(np.int8)
            col_names = [f"{col_name}_{val}" for val in unique_values]
        return expanded, col_names

def match_col_names(partial_names, header):
    """
    Match partial column names with full column names from the header.
    partial_names: list of partial column names to match.
    Needed to match the columns present in the training set with the ones in the test set.
    header: list of all column names
    return: list of matched full column names
    """
    matched_names = []
    for pname in partial_names:
        for hname in header:
            if hname == pname:
                matched_names.append(hname)
                break
            elif hname.startswith(pname):
                matched_names.append(hname)
                break
            elif hname == f"_{pname}":
                matched_names.append(hname)
                break
    return matched_names

def expand_dataset_col(col_list, path, with_missing=True):
    '''
    Expand a dataset by selecting specific columns and expanding categorical columns into binary indicators.
    col_list: list of column names to select and expand
    header: list of all column names in the dataset
    dataset: numpy array of the dataset to expand or path to the dataset
    with_missing: whether to handle missing values
    return: expanded dataset as a numpy array and list of new column names
    '''
    path_dataset = path
    header = load_header(path_dataset)
    col_list = match_col_names(col_list, header)

    col_list, col_index = get_col_index(col_list, header)
    dataset = load_data_col(path_dataset, cols=col_index)
    print("number of columns accepted: " + str(len(col_index)) + " out of " + str(len(col_list)))
    
    x_train_subset = dataset
    expanded_cols = []
    expanded_col_names = []

    for i, col_name in enumerate(col_list):
        expanded_col, col_names = expand_column(x_train_subset[:, i], col_name, with_missing=with_missing)
        if expanded_col is None:
            print(f"Skipping column '{col_name}' due to too many unique values.")
            continue
        expanded_cols.append(expanded_col)
        expanded_col_names.extend(col_names)
    return np.hstack(expanded_cols), expanded_col_names

def check_saved_extended_dataset(col_list, file_path, col_missing=False):
    # Change file extensions to .npy
    npy_path = file_path.rsplit('.', 1)[0] + '.npy'
    col_path = file_path.rsplit('.', 1)[0] + '_columns.txt'
    
    if os.path.isfile(npy_path) and os.path.isfile(col_path):
        print("Extended dataset found.")
        with open(col_path, 'r') as f:
            #read first line
            stripped_header = f.readline().strip().split(',')
            #read second line
            extended_col_names = f.readline().strip().split(',')
            
        
        if len(stripped_header) == len(col_list):
            print("Extended dataset has the correct number of columns.")
            x_train_expanded = np.load(npy_path)
            if not col_missing:
                # Create a boolean mask for columns NOT ending with '_is_missing'
                mask = np.array([not name.endswith('_is_missing') for name in extended_col_names])
                # Apply mask to columns (axis=1)
                x_train_expanded = x_train_expanded[:, mask]
                # Apply mask to column names (convert to numpy array first)
                extended_col_names = np.array(extended_col_names)[mask].tolist()
            return x_train_expanded, extended_col_names
        else:
            difference = set(col_list).symmetric_difference(set(stripped_header))
            print("Column mismatch. Difference:", difference)
            raise ValueError(f"Expected {len(col_list)} columns, got {len(stripped_header)}")
    else:
        print("Extended dataset not found. Generating...")
        x_train_expanded, expanded_col_names = expand_dataset_col(col_list, '../data/dataset/x_train.csv', True)
        
        # Save as binary .npy (10-100x smaller than CSV)
        np.save(npy_path, x_train_expanded)
        print(f"Extended dataset saved to {npy_path}")
        
        # Save column names as text
        with open(col_path, 'w') as f:
            f.write(','.join(col_list))
            f.write('\n')
            f.write(','.join(expanded_col_names))
        print(f"Column names saved to {col_path}")
        
        return x_train_expanded, expanded_col_names

def build_k_indices(y, k_fold, seed):
    """module taken frome the lecture:
    
    build k indices for k-fold.

    Args:
        y:      shape=(N,)
        k_fold: K in K-fold, i.e. the fold num
        seed:   the random seed

    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
    return np.array(object=k_indices, dtype=int)

def split_data_train_test_80(x_full, y_full, seed=1):
    """split the dataset into training set and test set (80%-20%)

    Args:
        x_full: shape=(N,D)
        y_full: shape=(N,1)
        seed:   random seed

    Returns:
        x_train: shape=(0.8N,D)
        y_train: shape=(0.8N,1)
        x_test:  shape=(0.2N,D)
        y_test:  shape=(0.2N,1)
    """
    k_fold = 5
    seed = 1
    k_indices = build_k_indices(y_full, k_fold, seed)
    ks = [1]
    x_test = []
    y_test = []
    x_train = []
    y_train = []
    for k in ks:
        x_test = x_full[k_indices[k]]
        y_test = y_full[k_indices[k]]
        x_train = np.delete(x_full, k_indices[k], axis=0)
        y_train = np.delete(y_full, k_indices[k], axis=0)
    return x_train, y_train, x_test, y_test
   

def resample_by_ratio(X, y, percentage=1.0, sampling="normal", seed=1):
    """
    Resample X,y so that count(-1)/count(1) ~= percentage.
    sampling: 'normal' | 'over' | 'under'
    Returns X_res, y_res.
    """
    if sampling == "normal":
        return X, y
    if percentage <= 0:
        raise ValueError("percentage must be > 0")

    rng = np.random.RandomState(seed)
    y1d = np.asarray(y).reshape(-1)

    idx_neg = np.where(y1d == -1)[0]
    idx_pos = np.where(y1d == 1)[0]
    n_neg, n_pos = len(idx_neg), len(idx_pos)

    if n_neg == 0 or n_pos == 0:
        # cannot rebalance if a class is missing
        return X, y

    r_cur = n_neg / n_pos
    r_tgt = float(percentage)

    if sampling == "over":
        if r_tgt >= r_cur:
            # need more negatives
            n_pos_tgt = n_pos
            n_neg_tgt = int(np.ceil(r_tgt * n_pos_tgt))
            n_neg_tgt = max(n_neg_tgt, n_neg)  # only oversample
            sel_neg = rng.choice(idx_neg, size=n_neg_tgt, replace=True)
            sel_pos = idx_pos
        else:
            # need more positives
            n_neg_tgt = n_neg
            n_pos_tgt = int(np.ceil(n_neg_tgt / r_tgt))
            n_pos_tgt = max(n_pos_tgt, n_pos)
            sel_pos = rng.choice(idx_pos, size=n_pos_tgt, replace=True)
            sel_neg = idx_neg

    elif sampling == "under":
        if r_tgt >= r_cur:
            # reduce positives
            n_neg_tgt = n_neg
            n_pos_tgt = int(np.floor(n_neg_tgt / r_tgt))
            n_pos_tgt = min(n_pos_tgt, n_pos)  # only undersample
            n_pos_tgt = max(n_pos_tgt, 1)
            sel_pos = rng.choice(idx_pos, size=n_pos_tgt, replace=False)
            sel_neg = idx_neg
        else:
            # reduce negatives
            n_pos_tgt = n_pos
            n_neg_tgt = int(np.floor(r_tgt * n_pos_tgt))
            n_neg_tgt = min(n_neg_tgt, n_neg)
            n_neg_tgt = max(n_neg_tgt, 1)
            sel_neg = rng.choice(idx_neg, size=n_neg_tgt, replace=False)
            sel_pos = idx_pos
    else:
        raise ValueError("sampling must be one of {'normal','over','under'}")

    new_idx = np.concatenate([sel_neg, sel_pos])
    rng.shuffle(new_idx)
    return X[new_idx], y[new_idx]

def split_data_train_test_80_new(x_full, y_full, seed=1, sampling="normal", percentage=1.0):
    """split 80/20 using 5-fold (use fold 1 as test). Optionally re-sample train.
    sampling: 'normal' | 'over' | 'under'
    percentage: target ratio #-1/#1 on the TRAIN set (e.g., 1.0 => balanced)
    """
    k_fold = 5
    rng = np.random.RandomState(seed)
    k_indices = build_k_indices(y_full, k_fold, seed)
    k = 1

    x_test = x_full[k_indices[k]]
    y_test = y_full[k_indices[k]]
    x_train = np.delete(x_full, k_indices[k], axis=0)
    y_train = np.delete(y_full, k_indices[k], axis=0)

    # apply optional sampling on train only
    x_train, y_train = resample_by_ratio(x_train, y_train, percentage=percentage, sampling=sampling, seed=seed)

    return x_train, y_train, x_test, y_test


def create_csv_submission( y_pred, name):
    """ Module taken from the lectures helper functions.
    This function creates a csv file named 'name' in the format required for a submission in Kaggle or AIcrowd.
    The file will contain two columns the first with 'ids' and the second with 'y_pred'.
    y_pred must be a list or np.array of 1 and -1 otherwise the function will raise a ValueError.

    Args:
        ids (list,np.array): indices
        y_pred (list,np.array): predictions on data correspondent to indices
        name (str): name of the file to be created
    """
    ids = load_data_col('../data/dataset/x_test.csv', cols=(0,), has_header=True).astype(int)
    # Check that y_pred only contains -1 and 1
    if not all(i in [-1, 1] for i in y_pred):
        raise ValueError("y_pred can only contain values -1, 1")

    with open(name, "w", newline="") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})

def fill_missing_column(header_train, header_test, dataset_test):
    '''
    the test dataset has columns that the train dataset don't have,
    this function fills the missing columns with zeros and removes extra columns form the test dataset.
    Args:
        header_train: list of column names in the training set
        header_test: list of column names in the test set
        dataset_test: numpy array of the test dataset
    returns:
        aligned: numpy array of the test dataset with missing columns filled with zeros

    '''
    idx_map = {name: i for i, name in enumerate(header_test)}
    n_rows = dataset_test.shape[0]
    n_cols_target = len(header_train)

    # allocate result once with same dtype as dataset_test (zeros for missing cols)
    aligned = np.zeros((n_rows, n_cols_target), dtype=dataset_test.dtype)
    # copy columns that exist in the test set into the right positions
    for j, col in enumerate(header_train):
        i = idx_map.get(col)
        if i is not None and i < dataset_test.shape[1]:
            aligned[:, j] = dataset_test[:, i]

    return aligned

def plot_feature_importance(model, expanded_col_names,nr_features=20):
    feature_importance = (model.weights_[1:])  # Exclude intercept
    feature_names = np.asarray(expanded_col_names)
    sorted_indices = np.concatenate([np.argsort(feature_importance)[::-1][:nr_features], np.argsort(feature_importance)[::-1][-nr_features:]])

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(sorted_indices)), feature_importance[sorted_indices], align="center")
    plt.xticks(range(len(sorted_indices)), feature_names[sorted_indices], rotation=90)
    plt.title("Feature Importance (Sklearn Logistic Regression)")
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.tight_layout()
    return plt

def save_report(model, evaluation, values, timestamp):
    """
    Saves the shape of the datasets x_train, y_train, x_val, y_val,
    saves the roc curve report,
    saves the evaluate report,
    saves the feature importance plot.
    Args:
        model: trained model
        plot_loss: loss plot
        plot_roc: roc curve plot
        evaluation: evaluation report
        plot_feature: feature importance plot
        values: dictionary containing x_train, y_train, x_val, y_val, y_pred_probabilities
    Returns:
        None but saves the report to a text file with timestamp
    """

    
    filename =  timestamp + '_report.txt'
    with open(filename, 'w') as f:
        f.write("Loss plot saved as " + str(timestamp) +'_loss_plot.png\n')
        f.write("Feature importance plot saved as " + str(timestamp) +'_feature_importance.png\n')
        f.write("ROC curve plot saved as " + str(timestamp) +'_roc_curve.png\n')

        f.write("Model parameters:\n")
        for param, value in model.__dict__.items():
            f.write(f"{param}: {value}\n")
      
        f.write("\nEvaluation report:\n")
        f.write(evaluation + '\n')
        original_stdout = sys.stdout

    with open(timestamp + "_output_log" + ".txt", 'w') as f:
        sys.stdout = f
        # Repeat all the print statements here
        print("x_train shape:", values["x_train"].shape," y_train shape:", values["y_train"].shape)
        print("x_val shape:", values["x_val"].shape," y_val shape:", values["y_val"].shape)
        print(evaluation)
        print("Predicted probabilities from our model:", values["y_pred_probabilities"].flatten()[:100])
        sys.stdout = original_stdout

class LogisticRegression_costum:
    def __init__(
        self,
        lr=0.01,
        max_iter=1000,
        penalty=None,
        alpha=0.0,
        l1_ratio=0.5,
        method="",           # '' => full batch; 'batch' => mini-batch
        batch_size=512,
        fit_intercept=True,
        verbose=False,
        early_stopping=False,
        patience=5,
        tol=1e-4,
        bias_init=0.0,
        class_weight=None,
        plot_path=None,
        solver="gd",         # NEW: 'gd' | 'momentum' | 'rmsprop' | 'adam'
        momentum=0.9,        # for momentum
        beta1=0.9,           # for adam
        beta2=0.999,         # for adam/rmsprop
        epsilon=1e-8         # for adam/rmsprop
    ):
        self.lr = lr
        self.max_iter = max_iter
        self.penalty = penalty
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.method = method
        self.batch_size = batch_size
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.early_stopping = early_stopping
        self.patience = patience
        self.tol = tol
        self.bias_init = bias_init
        self.class_weight = class_weight
        self.plot_path = plot_path
        self.solver = solver
        self.momentum = momentum
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.weights_ = None
        self.best_weights_ = None
        self.train_losses_ = []
        self.val_losses_ = []

    def _add_intercept(self, X):
        if not self.fit_intercept:
            return X
        return np.hstack((np.ones((X.shape[0], 1)), X))

    def _sigmoid(self, z):
        #overflow-safe sigmoid for np.float16
        z = np.clip(z, -500, 500)  # clip to avoid overflow
        return 1 / (1 + np.exp(-z))

    def _compute_class_weights(self, y):
        if self.class_weight is None:
            return np.ones_like(y, dtype=float)
        if self.class_weight == "balanced":
            classes, counts = np.unique(y, return_counts=True)
            total = y.shape[0]
            w = {cls: total / (len(classes) * cnt) for cls, cnt in zip(classes, counts)}
            return np.array([w[val] for val in y])
        if isinstance(self.class_weight, dict):
            return np.array([self.class_weight.get(val, 1.0) for val in y])
        raise ValueError("Invalid class_weight parameter")

    def _loss(self, X, y, sample_weight=None):
        m = X.shape[0]
        y_pred = self._sigmoid(X @ self.weights_)
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)

        if sample_weight is None:
            sample_weight = np.ones_like(y, dtype=float)

        loss = -np.average(
            y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred),
            weights=sample_weight
        )

        if self.penalty == "l2":
            loss += self.alpha * np.sum(self.weights_[1:] ** 2) / (2 * m)
        elif self.penalty == "l1":
            loss += self.alpha * np.sum(np.abs(self.weights_[1:])) / m
        elif self.penalty == "elasticnet":
            l1 = self.l1_ratio * np.sum(np.abs(self.weights_[1:]))
            l2 = (1 - self.l1_ratio) * np.sum(self.weights_[1:] ** 2) / 2
            loss += self.alpha * (l1 + l2) / m
        return loss

    def _gradient(self, X, y, sample_weight=None):
        m = X.shape[0]
        y_pred = self._sigmoid(X @ self.weights_)
        if sample_weight is None:
            sample_weight = np.ones_like(y, dtype=float)
        error = (y_pred - y) * sample_weight
        grad = X.T @ error / np.sum(sample_weight)

        if self.penalty in ("l2", "elasticnet"):
            l2_term = self.alpha * (1 - (self.l1_ratio if self.penalty == "elasticnet" else 0)) * np.r_[[0], self.weights_[1:]] / m
            grad += l2_term
        if self.penalty in ("l1", "elasticnet"):
            l1_term = self.alpha * (self.l1_ratio if self.penalty == "elasticnet" else 1) * np.sign(self.weights_) / m
            l1_term[0] = 0
            grad += l1_term
        return grad

    def _plot_loss(self):
        plt.figure(figsize=(7, 5))
        plt.plot(self.train_losses_, label="Train Loss")
        if self.val_losses_:
            plt.plot(self.val_losses_, label="Validation Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True)
        if self.plot_path:
            plt.savefig(self.plot_path, bbox_inches="tight")
        else:
           return plt

    def _soft_threshold(self, w, lam):
        # Prox for L1 (do not regularize intercept outside)
        return np.sign(w) * np.maximum(0.0, np.abs(w) - lam)

    def fit(self, X, y, X_val=None, y_val=None):
        X = X.astype(np.float32)
        y = y.astype(np.float32)
        X = self._add_intercept(X)
        self.weights_ = np.zeros(X.shape[1])
        self.weights_[0] = self.bias_init
        best_loss = np.inf
        no_improve_count = 0
        sample_weight = self._compute_class_weights(y)

        # --- optimizer state (local) ---
        v_mom = np.zeros_like(self.weights_)              # for momentum
        rms_cache = np.zeros_like(self.weights_)          # for rmsprop
        m_adam = np.zeros_like(self.weights_)             # for adam
        v_adam = np.zeros_like(self.weights_)             # for adam
        t_adam = 0                                        # for adam bias correction

        # --- SAGA state (if used) ---
        if self.solver == "saga":
            n, d = X.shape
            den = float(np.sum(sample_weight))
            # a_mem[i] = w_i * (p_i - y_i) for last time sample i seen
            p0 = self._sigmoid(X @ self.weights_)  # initially 0.5 with zero weights
            a_mem = (p0 - y) * sample_weight       # shape (n,)
            grad_avg = (X.T @ a_mem) / den         # averaged gradient of data-fit term
            m_full = n                             

        for i in range(self.max_iter):
            if self.solver == "saga":
                # One-sample SAGA update (fast and memory-light)
                idx = np.random.randint(0, X.shape[0])
                x_i = X[idx]             # shape (d,)
                y_i = y[idx]
                w_i = sample_weight[idx]

                p_i =  self._sigmoid(x_i @ self.weights_)
                a_old = a_mem[idx]
                a_new = w_i * (p_i - y_i)
                corr = (a_new - a_old) / den

                # variance-reduced gradient of data-fit term
                vr_grad = grad_avg + corr * x_i

                # L2 part (and ElasticNet's L2) as smooth gradient (no intercept)
                if self.penalty in ("l2", "elasticnet"):
                    l2_coeff = self.alpha * (1.0 - (self.l1_ratio if self.penalty == "elasticnet" else 0.0)) / m_full
                    vr_grad = vr_grad + np.r_[0.0, l2_coeff * self.weights_[1:]]

                # gradient step
                w_new = self.weights_ - self.lr * vr_grad

                # Prox for L1 part (L1 or ElasticNet)
                if self.penalty in ("l1", "elasticnet"):
                    lam = self.lr * self.alpha * (self.l1_ratio if self.penalty == "elasticnet" else 1.0) / m_full
                    w_new[1:] = self._soft_threshold(w_new[1:], lam)  # don't regularize intercept

                self.weights_ = w_new

                # Update memory and averaged gradient
                grad_avg = grad_avg + corr * x_i
                a_mem[idx] = a_new

                # for logging, compute a lightweight loss on the sampled point
                X_batch = x_i[None, :]
                y_batch = np.array([y_i], dtype=np.float32)
                w_batch = np.array([w_i], dtype=np.float32)

            else:
                # Original batching for other solvers
                if self.method == "batch":
                    idxs = np.random.randint(0, X.shape[0], size=self.batch_size)
                    X_batch = X[idxs]
                    y_batch = y[idxs]
                    w_batch = sample_weight[idxs]
                else:
                    X_batch, y_batch, w_batch = X, y, sample_weight

                grad = self._gradient(X_batch, y_batch, w_batch)

                # --- weight update per solver ---
                if self.solver in ("gd", "sgd"):
                    self.weights_ -= self.lr * grad

                elif self.solver == "momentum":
                    v_mom = self.momentum * v_mom - self.lr * grad
                    self.weights_ += v_mom

                elif self.solver == "rmsprop":
                    rms_cache = self.beta2 * rms_cache + (1.0 - self.beta2) * (grad ** 2)
                    self.weights_ -= self.lr * grad / (np.sqrt(rms_cache) + self.epsilon)

                elif self.solver == "adam":
                    t_adam += 1
                    m_adam = self.beta1 * m_adam + (1.0 - self.beta1) * grad
                    v_adam = self.beta2 * v_adam + (1.0 - self.beta2) * (grad ** 2)
                    m_hat = m_adam / (1.0 - self.beta1 ** t_adam)
                    v_hat = v_adam / (1.0 - self.beta2 ** t_adam)
                    self.weights_ -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
                else:
                    raise ValueError("solver must be one of {'gd','sgd','momentum','rmsprop','adam','saga'}")

            # logging
            train_loss = self._loss(X_batch, y_batch, w_batch)
            self.train_losses_.append(train_loss)
            val_loss = None

            if X_val is not None:
                val_loss = self._loss(self._add_intercept(X_val), y_val)
                self.val_losses_.append(val_loss)

            if self.verbose and i % 10 == 0:
                msg = f"Iter {i}: train_loss={train_loss:.4f}"
                if val_loss is not None:
                    msg += f", val_loss={val_loss:.4f}"
                print(msg)

            if self.early_stopping and val_loss is not None:
                if val_loss is None:
                    pass
                elif val_loss + self.tol < best_loss:
                    best_loss = val_loss
                    self.best_weights_ = self.weights_.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                if no_improve_count >= self.patience:
                    if self.verbose:
                        print("Early stopping triggered.")
                    break

        if self.best_weights_ is not None:
            self.weights_ = self.best_weights_

        if self.verbose:
            plt_loss = self._plot_loss()
            return plt_loss

    def predict_proba(self, X):
        X = self._add_intercept(X)
        return self._sigmoid(X @ self.weights_)

    def predict(self, X, threshold=0.5):
        temp = (self.predict_proba(X) >= threshold)
        temp = (temp.astype(np.int8) * 2) - 1
        return temp

    def evaluate(self, X, y, threshold=0.5):
        y_pred = self.predict(X, threshold)
        return (self.classification_report(y, y_pred))
    
    def roc_curve(self, X, y, plot=False):
        '''
        Plot ROC curve, implemented using only NumPy and Matplotlib.
        shows the best f1 score on the curve
        Args:
            X: numpy array of shape (N, D)
            y: numpy array of shape (N, 1)
            plot: whether to plot the ROC curve
        Returns: 
            best_f1, best_thresh
        '''
    
        y_scores = self.predict_proba(X)
        thresholds = np.arange(0, 1.01, 0.01)
        tpr = []
        fpr = []
        f1_scores = []
    
        for thresh in thresholds:
            y_pred = ((y_scores >= thresh).astype(int)).astype(np.int8) * 2 - 1
            tp = np.sum((y_pred == 1) & (y == 1))
            tn = np.sum((y_pred == -1) & (y == -1))
            fp = np.sum((y_pred == 1) & (y == -1))
            fn = np.sum((y_pred == -1) & (y == 1))
            
            tpr_val = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tpr_val
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            tpr.append(tpr_val)
            fpr.append(fpr_val)
            f1_scores.append(f1)
    
        # Find best F1 score
        best_f1_idx = np.argmax(f1_scores)
        best_f1 = f1_scores[best_f1_idx]
        best_thresh = thresholds[best_f1_idx]
        best_fpr = fpr[best_f1_idx]
        best_tpr = tpr[best_f1_idx]
        
        if plot:
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label="ROC Curve")
            plt.plot(best_fpr, best_tpr, 'ro', markersize=10, 
                    label=f'Best F1={best_f1:.3f} (thresh={best_thresh:.2f})')
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("Receiver Operating Characteristic")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
        print(f"Best F1 Score: {best_f1:.4f} at threshold {best_thresh:.2f}")
        print(f"TPR: {best_tpr:.4f}, FPR: {best_fpr:.4f}")
        if plot:
            return best_f1, best_thresh, plt
        else:
            return best_f1, best_thresh

    def classification_report(y_true, y_pred, digits=3):
        """NumPy-only version of sklearn.metrics.classification_report."""
        classes = np.unique(np.concatenate((y_true, y_pred)))
        report = []
        total = len(y_true)

        accuracy = np.sum(y_true == y_pred) / total
        report.append(f"\nAccuracy: {accuracy:.{digits}f}\n")

        metrics = []
        for cls in classes:
            tp = np.sum((y_true == cls) & (y_pred == cls))
            fp = np.sum((y_true != cls) & (y_pred == cls))
            fn = np.sum((y_true == cls) & (y_pred != cls))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            support = np.sum(y_true == cls)
            metrics.append((precision, recall, f1, support))
            report.append(f"Class {cls}: precision={precision:.{digits}f}, recall={recall:.{digits}f}, f1={f1:.{digits}f}, support={support}")

        precisions, recalls, f1s, supports = zip(*metrics)
        macro_avg = (np.mean(precisions), np.mean(recalls), np.mean(f1s))
        weighted_avg = (
            np.average(precisions, weights=supports),
            np.average(recalls, weights=supports),
            np.average(f1s, weights=supports),
        )

        report.append("\nMacro avg:   precision={:.{d}f}, recall={:.{d}f}, f1={:.{d}f}".format(*macro_avg, d=digits))
        report.append("Weighted avg: precision={:.{d}f}, recall={:.{d}f}, f1={:.{d}f}".format(*weighted_avg, d=digits))
        return "\n".join(report)

