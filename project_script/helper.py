import numpy as np
import csv
import matplotlib.pyplot as plt
#add a seed for np
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
    #check name for name, if not found print an error message with the name not found and continue adding the found columns
    for name in cols:
        if name not in header:
            print(f"Column '{name}' not found in the provided list.")
    return [name for name in cols if name in header], [header.index(name) for name in cols if name in header]

def expand_column(col, col_name, with_missing=True):
    '''
    Expand a 1D column vector into a N array where N is the number of unique values in col.
    Each column in the output array is a binary indicator (0 or 1) of whether the corresponding
    entry in col matches the unique value for that column.
    col: 1D numpy array of categorical values

    #Missing values are handled by adding an additional binary indicator column and set to 0.
    '''

    is_missing = np.isnan(col).astype(np.int8)
    has_missing = np.any(is_missing)
    unique_values = np.unique(col)[~np.isnan(np.unique(col))]

    if with_missing:
        if unique_values.size > 10:
            if not has_missing:
                #normalize col 
                col = (col - col.mean()) / col.std()
                return col.reshape(-1, 1), [col_name]
            col = np.where(np.isnan(col), np.nanmean(col), col)
            #normalize col 
            col = (col - col.mean()) / col.std()
            expanded = np.column_stack([col, is_missing])
            col_names = np.append([col_name], [f"{col_name}_is_missing"])
        else:
            #print(f"Unique values in column '{col_name}': {unique_values}")
            expanded = np.zeros((col.size, unique_values.size), dtype=np.int16)
            for i, val in enumerate(unique_values):
                expanded[:, i] = (col == val).astype(int)
            col_names = [f"{col_name}_{val}" for val in unique_values]

            if has_missing:
                expanded = np.column_stack([expanded, is_missing])
                col_names.append(f"{col_name}_is_missing")

        return expanded, col_names
    else:
        col = np.where(np.isnan(col), np.nanmean(col), col)
        if unique_values.size > 10:
            #normalize col 
            col = (col - col.mean()) / col.std()
            return col.reshape(-1, 1), [col_name]
        else:
            expanded = np.zeros((col.size, unique_values.size), dtype=np.int16)
            for i, val in enumerate(unique_values):
                expanded[:, i] = (col == val).astype(int)
            col_names = [f"{col_name}_{val}" for val in unique_values]
        return expanded, col_names

def match_col_names(partial_names, header):
    """
    Match partial column names with full column names from the header.
    partial_names: list of partial column names to match
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
    return: expanded dataset as a numpy array and list of new column names
    '''
    path_dataset = path
    header = load_header(path_dataset)
    print("Original col_list:", col_list)
    print("Header:", header)
    col_list = match_col_names(col_list, header)

    col_name, col_index = get_col_index(col_list, header)
    print(col_index)
    dataset = load_data_col(path_dataset, cols=col_index)
    header = load_header(path_dataset)
    x_train_subset = dataset
   
    col_list = match_col_names(col_list, header)
    print("Matched col_list:", col_list)
    col_list, col_index = get_col_index(col_list, header)
    print(len(col_index))
    #load x_train with only the columns in col_indices
    
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

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold.

    Args:
        y:      shape=(N,)
        k_fold: K in K-fold, i.e. the fold num
        seed:   the random seed

    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold

    >>> build_k_indices(np.array([1., 2., 3., 4.]), 2, 1)
    array([[3, 2],
           [0, 1]])
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
    #print shape of x_train and y_train and x_test and y_test
    print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")
    return x_train, y_train, x_test, y_test


def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a non-negative loss (scalar)
    """
    '''

    # compute the loss: negative log likelihood
    y_hat = sigmoid(tx @ w)
    loss = -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    '''
    y = np.asarray(y).reshape(-1)              # ensure 1D
    t = (tx @ w).reshape(-1)
    # stable loss: mean( log(1 + exp(t)) - y * t )  implemented via np.logaddexp(0, t)
    loss = -np.mean(np.logaddexp(0, t) - y * t)
    return float(loss)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent.

    Args:
        y: numpy array of shape (N, 1)
        tx: numpy array of shape (N, D)
        initial_w: numpy array of shape (D, 1)
        max_iters: scalar
        gamma: scalar

    Returns:
        losses: list of loss values
        ws: list of weights
    """
    ws = [initial_w]
    w = initial_w
    losses = [calculate_loss(y, tx, w)]
    for n_iter in range(max_iters):
        if n_iter % 2 == 0:
            print(f"Iteration {n_iter}/{max_iters}: loss={losses[-1]}")
        gradient = calculate_gradient(y, tx, w)
        w = w - gamma * gradient
        loss = calculate_loss(y, tx, w)
        ws.append(w)
        losses.append(loss)

    return ws[-1], np.asarray(losses[-1])


def sigmoid(t):
    """apply sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array
    """

    #return 1 / (1 + np.exp(-t))
    #clipped
    t = np.asarray(t, dtype=np.float64)
    t = np.clip(t, -500, 500)
    return 1.0 / (1.0 + np.exp(-t))


def calculate_gradient(y, tx, w):
    """compute the gradient of loss.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a vector of shape (D, 1)
    """
    y = np.asarray(y).reshape(-1, 1)        # (N,1)
    t = tx @ w                             # (N,1) if w is (D,1)
    y_hat = sigmoid(t).reshape(-1, 1)      # (N,1)
    error = y_hat - y                      # (N,1) (no broadcasting)
    gradient = tx.T @ error / y.shape[0]   # (D,1)
    return gradient


#def penalized_logistic_regression(y, tx, w, lambda_):
#    """return the loss and gradient.
#
#    Args:
#        y:  shape=(N, 1)
#        tx: shape=(N, D)
#        w:  shape=(D, 1)
#        lambda_: scalar
#
#    Returns:
#        loss: scalar number
#        gradient: shape=(D, 1)
#    """
#    gradient = calculate_gradient(y, tx, w) + lambda_ * 2 * w
#    loss = calculate_loss(y, tx, w)
#
#    return float(loss), gradient

def penalized_logistic_regression(y, tx, w, lambda_):
    """Return L2-regularized logistic loss and gradient.

    Loss = mean( log(1+exp(t)) - y * t ) + lambda_ * ||w||^2
    Gradient = tx.T @ (sigma(t) - y) / N + 2 * lambda_ * w
    """
    w=np.int16(w)
    y = np.asarray(y).reshape(-1)              # ensure 1D
    t = (np.int16(tx) @ w).reshape(-1)                            # (N,1)
    y_hat = sigmoid(t)                      # (N,1)

    # stable data loss
    loss_data = np.mean(np.logaddexp(0, t.reshape(-1)) - y.reshape(-1) * t.reshape(-1))
    loss_reg = lambda_ * np.sum(w ** 2)     # L2 penalty
    loss = float(loss_data + loss_reg)
    #cast everything to int16 to avoid overflow
    y_hat = y_hat.astype(np.int16)
    y = y.astype(np.int16)
    gradient = tx.T @ (y_hat - y) / y.shape[0] + 2 * np.int16(lambda_) * w
    return loss, gradient


def logistic_regression_penalized(y, tx, initial_w, max_iters, gamma, lambda_):
    """Gradient descent training for L2-regularized logistic regression.

    Returns:
        w: final weights (D,1)
        loss: final scalar loss
    """
    w = initial_w
    loss, _ = penalized_logistic_regression(y, tx, w, lambda_)
    for n_iter in range(max_iters):
        if n_iter % 2 == 0:
            print(f"Iteration {n_iter}/{max_iters}: loss={loss}")
        loss, grad = penalized_logistic_regression(y, tx, w, lambda_)
        w = w - gamma * grad

    loss, _ = penalized_logistic_regression(y, tx, w, lambda_)
    return w, loss

# ...existing code...
def _sigmoid_f32(t):
    # small float32 sigmoid for intermediate calculations
    t = np.asarray(t, dtype=np.float32)
    t = np.clip(t, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-t))


def penalized_logistic_regression_int16(y, tx, w, lambda_, pos_weight=50, neg_weight=1):
    """
    Memory-compact variant: store inputs as int16, perform computations in float32
    and return (loss float, gradient float32).
    Labels expected in {-1, 1}. Bias (w[0]) is NOT regularized.
    """
    # store/compress inputs as int16
    tx_i = np.asarray(tx, dtype=np.int16)
    w_i = np.asarray(w, dtype=np.int16).reshape(-1, 1)
    y_i = np.asarray(y, dtype=np.int16).reshape(-1, 1)

    # minimal working copies in float32 for computations
    tx_f = tx_i.astype(np.float32)
    w_f = w_i.astype(np.float32)
    y01 = (y_i == 1).astype(np.float32)

    sample_weights = np.where(y_i == 1, float(pos_weight), float(neg_weight)).astype(np.float32).reshape(-1, 1)
    sum_w = float(sample_weights.sum())

    # linear scores & predictions (float32)
    t = tx_f @ w_f                              # (N,1) float32
    y_hat = _sigmoid_f32(t)                     # (N,1) float32

    # stable per-sample loss (float32)
    per_sample_loss = np.logaddexp(0.0, t.reshape(-1)) - (y01.reshape(-1) * t.reshape(-1))
    loss_data = float((sample_weights.reshape(-1) @ per_sample_loss) / sum_w)

    # do not regularize bias term
    w_reg = w_f.copy()
    if w_reg.shape[0] > 0:
        w_reg[0, 0] = 0.0
    loss_reg = float(lambda_ * np.sum(w_reg ** 2))
    loss = float(loss_data + loss_reg)

    # weighted gradient (float32)
    residual = (y_hat - y01) * sample_weights        # (N,1)
    gradient = (tx_f.T @ residual) / sum_w + 2.0 * float(lambda_) * w_reg  # (D,1) float32

    return loss, gradient.astype(np.float32)


def logistic_regression_penalized_int16(y, tx, initial_w, max_iters, gamma, lambda_, pos_weight=50, neg_weight=1):
    """
    Gradient descent using compressed int16 storage for weights/data.
    Updates are computed in float32; weights are quantized back to int16 after each step
    to keep memory footprint low.
    Returns final weights as int16 array (D,1) and final loss (float).
    """
    # compress initial data
    tx_i = np.asarray(tx, dtype=np.int16)
    y_i = np.asarray(y, dtype=np.int16).reshape(-1, 1)
    w_i = np.asarray(initial_w, dtype=np.int16).reshape(-1, 1)

    # keep small float working vector for update
    w_f = w_i.astype(np.float32)

    loss = None
    for n_iter in range(max_iters):
        # compute loss & gradient using float32 intermediates
        loss, grad = penalized_logistic_regression_int16(y_i, tx_i, w_i, lambda_, pos_weight=pos_weight, neg_weight=neg_weight)
        if n_iter % 2 == 0:
            print(f"Iteration {n_iter}/{max_iters}: loss={loss}")
        # gradient is float32; update in float32 then quantize back to int16
        w_f = w_f - gamma * grad
        # quantize to int16 storage (rounding) to reduce memory footprint
        w_i = np.round(w_f).astype(np.int16).reshape(-1, 1)
        # keep w_f synced to quantized value to avoid drift of unused precision
        w_f = w_i.astype(np.float32)

    # final recompute loss
    loss, _ = penalized_logistic_regression_int16(y_i, tx_i, w_i, lambda_, pos_weight=pos_weight, neg_weight=neg_weight)
    return w_i, float(loss)
# ...existing code...

def prediction(x,w,threshold=0.5):
    """make a prediction given new data x and weights w.

    Args:
        x: numpy array of shape (N, D)
        w: numpy array of shape (D, 1)

    Returns:
        y_pred: numpy array of shape (N, 1)
    """
    y_pred = sigmoid(np.dot(x, w))
    y_pred = np.sign(y_pred - threshold)    
    return y_pred

def calculate_accuracy(y, y_pred):
    """calculate the accuracy of the prediction.

    Args:
        y: numpy array of shape (N, 1)
        y_pred: numpy array of shape (N, 1)

    Returns:
        accuracy: scalar
    """
    accuracy = np.mean(y == y_pred)
    return accuracy

def f1_score(y, y_pred):
    tp = np.sum((y == 1) & (y_pred == 1))
    fp = np.sum((y == -1) & (y_pred == 1))
    fn = np.sum((y == 1) & (y_pred == -1))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1


import numpy as np

class WeightedLogisticRegression:
    def __init__(self, lr=0.01, n_iter=1000):
        self.lr = lr
        self.n_iter = n_iter
        self.w = None
        self.b = 0
        self.class_weights = {}

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _compute_class_weights(self, y):
        n_samples = len(y)
        classes, counts = np.unique(y, return_counts=True)
        n_classes = len(classes)
        weights = {}
        for c, count in zip(classes, counts):
            weights[c] = n_samples / (n_classes * count)
        sweights = {-1.0: 1.0, 1.0: 50.0}
        return weights

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        # Compute class weights
        self.class_weights = self._compute_class_weights(y)

        for _ in range(self.n_iter):
            if _ % 10 == 0:
                print(f"Iteration {_+1}/{self.n_iter}")
            # Linear model
            linear_output = np.dot(X, self.w) + self.b
            y_pred = self._sigmoid(linear_output)

            # Compute sample weights depending on class
            sample_weights = np.vectorize(self.class_weights.get)(y)

            # Gradient with weights
            dw = (1 / n_samples) * np.dot(X.T, sample_weights * (y_pred - y))
            db = (1 / n_samples) * np.sum(sample_weights * (y_pred - y))

            # Update parameters
            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict_proba(self, X):
        linear_output = np.dot(X, self.w) + self.b
        return self._sigmoid(linear_output)

    def predict(self, X, threshold=0.5):
        y_prob = self.predict_proba(X)
        return (y_prob >= threshold).astype(int)
    
def create_csv_submission(ids, y_pred, name):
    """
    This function creates a csv file named 'name' in the format required for a submission in Kaggle or AIcrowd.
    The file will contain two columns the first with 'ids' and the second with 'y_pred'.
    y_pred must be a list or np.array of 1 and -1 otherwise the function will raise a ValueError.

    Args:
        ids (list,np.array): indices
        y_pred (list,np.array): predictions on data correspondent to indices
        name (str): name of the file to be created
    """
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
    the test dataset has columns that the train dataset don't have, and vice versa
    to solve this we need to traverse both headers and remove columns that are only present in the header_train
    and add columns that are only present in header_test with 0 values
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


# file: logistic_regression_numpy.py
import numpy as np
import matplotlib.pyplot as plt

# file: logistic_regression_numpy.py
import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression_costum:
    def __init__(
        self,
        lr=0.01,
        max_iter=1000,
        penalty=None,
        alpha=0.0,
        l1_ratio=0.5,
        method="",
        batch_size=512,
        fit_intercept=True,
        verbose=False,
        early_stopping=False,
        patience=5,
        tol=1e-4,
        bias_init=0.0,
        class_weight=None,
        plot_path=None
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

        self.weights_ = None
        self.best_weights_ = None
        self.train_losses_ = []
        self.val_losses_ = []

    def _add_intercept(self, X):
        if not self.fit_intercept:
            return X
        return np.hstack((np.ones((X.shape[0], 1)), X))

    def _sigmoid(self, z):
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


    def _hessian(self, X):
        y_pred = self._sigmoid(X @ self.weights_)
        s = y_pred * (1 - y_pred)
        # Use broadcasting instead of np.diag
        return (X.T * s) @ X / X.shape[0]

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
            plt.show()

    def fit(self, X, y, X_val=None, y_val=None):
        X = X.astype(np.float32)
        y = y.astype(np.float32)
        X = self._add_intercept(X)
        self.weights_ = np.zeros(X.shape[1])
        self.weights_[0] = self.bias_init
        best_loss = np.inf
        no_improve_count = 0
        sample_weight = self._compute_class_weights(y)
        # existing gradient-descent/Newton code below unchanged
        for i in range(self.max_iter):
            if self.method == "stochastic":
                idxs = np.random.randint(0, X.shape[0], size=self.batch_size)
                X_batch = X[idxs]
                y_batch = y[idxs]
                w_batch = sample_weight[idxs]
            else:
                X_batch, y_batch, w_batch = X, y, sample_weight

            grad = self._gradient(X_batch, y_batch, w_batch)
            if self.method == "newton":
                H = self._hessian(X_batch)
                try:
                    self.weights_ -= np.linalg.inv(H) @ grad
                except np.linalg.LinAlgError:
                    self.weights_ -= self.lr * grad
            else:
                self.weights_ -= self.lr * grad

            #train_loss = self._loss(X, y, sample_weight)
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
            self._plot_loss()

    def predict_proba(self, X):
        X = self._add_intercept(X)
        return self._sigmoid(X @ self.weights_)

    def predict(self, X, threshold=0.5):
        temp = (self.predict_proba(X) >= threshold)
        temp = (temp.astype(np.int8) * 2) - 1
        return temp

    def evaluate(self, X, y, threshold=0.5, digits=3):
        y_pred = self.predict(X, threshold)
        print(classification_report(y, y_pred, digits=digits))
    
    def roc_curve(self, X, y, plot=False):
        '''
        Plot ROC curve, implemented using only NumPy and Matplotlib.
        shows the best f1 score on the curve
        X: numpy array of shape (N, D)
        y: numpy array of shape (N, 1)
        Returns: best_f1, best_thresh
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
            plt.show()
            
        print(f"Best F1 Score: {best_f1:.4f} at threshold {best_thresh:.2f}")
        print(f"TPR: {best_tpr:.4f}, FPR: {best_fpr:.4f}")
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

