#imports
import numpy as np
import csv


#Variables
col_list = ['_RFHLTH',
'_HCVU651', #Respondents aged 18-64 who have any form of health care coverage
'_RFHYPE5', #Adults who have been told they have high blood pressure by a doctor, nurse, or other health professional
'_CHOLCHK', #Cholesterol check within past five years
'_RFCHOL', #adults who have had their cholesterol checked and have been told by a doctor, nurse, or other health professional that it was high 59k nan
'_MICHD', #Respondents that have ever reported having coronary heart disease (CHD) or myocardial infarction (MI) 3k nan
'_ASTHMS1', #Computed asthma status
'_DRDXAR1', #Respondents who have had a doctor diagnose them as having some form of arthritis
'_PRACE1', #Preferred race category
'_MRACE1', #Calculated multiracial race categorization
'_HISPANC', #Hispanic, Latino/a, or Spanish origin calculated variable
'_RACE', #Race/ethnicity categories
'_RACEG21', #White non-Hispanic race group
'_RACEGR3', #Five-level race/ethnicity category
'_RACE_G1', #Race groups used for internet prevalence tables 7k nan
'_AGEG5YR', #Fourteen-level age category
'_AGE65YR', #Two-level age category
'_AGE80', #Imputed Age value collapsed above 80
'_AGE_G', #Six-level imputed age category
'HTIN4', #Reported height in inches #17k nan
'HTM4', #Reported height in meters 15k nan
'WTKG3', #Reported weight in kilograms
'BMI5', #Body Mass Index (BMI) 36k nan
'BMI5CAT', #Four-categories of Body Mass Index (BMI) 36k nan
'RFBMI5', #Adults who have a body mass index greater than 25.00 (Overweight or Obese) 36k nan
'CHLDCNT', #Number of children in household
'EDUCAG', #Level of education completed
'INCOMG', #Income categories
'SMOKER3', #Four-level smoker status: Everyday smoker, Someday smoker, Former smoker, Non-smoker
'RFSMOK3', #Adults who are current smokers
'DRNKANY5', #Adults who reported having had at least one drink of alcohol in the past 30 days.
'DROCDY3_', #Drink-occasions-per-day
'_RFBING5', #Binge drinkers (males having five or more drinks on one occasion, females having four or more drinks on one occasion)
'_DRNKWEK', #Calculated total number of alcoholic beverages consumed per week
'_RFDRHV5', #Heavy drinkers (adult men having more than 14 drinks per week and adult women having more than 7 drinks per
'FTJUDA1_', #Fruit juice intake in times per day 38k nan
'FRUTDA1_', #Fruit intake in times per day 36k nan
'BEANDAY_', #Bean intake in times per day 39k nan
'GRENDAY_', #Dark green vegetable intake in times per day 38k nan
'ORNGDAY_', #Orange-colored vegetable intake in times per day 39k nan
'VEGEDA1_', #Other vegetable intake in times per day 41k nan
'_MISFRTN', #The number of missing fruit responses
'_MISVEGN', #The number of missing vegetable responses
'_FRTRESP', #Missing any fruit responses
'VEGRES', #Missing any vegetable responses
'_FRUTSUM', #Total fruits consumed per day 43k nan
'_VEGESUM', #Total vegetables consumed per day 51k nan
'FRTLT1', #Consume Fruit 1 or more times per day
'VEGLT1', #Consume Vegetables 1 or more times per day
'FRT16', #Reported consuming Fruit >16 per day
'VEG23', #Reported consuming Vegetables >23 per day
'_FRUITEX', #Fruit Exclusion from analyses
'_VEGETEX', #Vegetable Exclusion from analyses
'_TOTINDA', #Adults who reported doing physical activity or exercise during the past 30 days other than their regular job
'METVL11_', #Activity MET Value for First Activity 146k nan
'METVL21_', #Activity MET Value for Second Activity 151k nan
'MAXVO2_', #Estimated Age-Gender Specific Maximum Oxygen Consumption
'FC60_', #Estimated Functional Capacity
'ACTIN11_', #Estimated Activity Intensity for First Activity 150k nan
'ACTIN21_', #Estimated Activity Intensity for Second Activity 154k nan
'PADUR1_', #Estimated Duration in Minutes for First Activity 154k nan
'PADUR2_', #Minutes of Second Activity 249k nan
'PAFREQ1_', #Physical Activity Frequency per Week for First Activity 150k nan
'PAFREQ2_', #Physical Activity Frequency per Week for Second Activity 246k nan
'MINAC11_', #Minutes of Physical Activity per week for First Activity 155k nan
'MINAC21_', #Minutes of Physical Activity per week for Second Activity 157k nan
'STRFREQ_', #Strength Activity Frequency per Week 44k nan
'PAMISS1_', #Missing Physical Activity Data
'PAMIN11_', #Minutes of Physical Activity per week for First Activity 158k nan
'PAMIN21_', #Minutes of Physical Activity per week for Second Activity 160k nan
'PA1MIN_', #Minutes of total Physical Activity per week 152k nan
'PAVIG11_', #Minutes of Vigorous Physical Activity per week for First Activity 153k nan
'PAVIG21_', #Minutes of Vigorous Physical Activity per week for Second Activity 158k nan
'PA1VIGM_', #Minutes of total Vigorous Physical Activity per week 150k nan
'PACAT1_', #Physical Activity Categories
'PAINDX1_', #Physical Activity Index
'PA150R2_', #Adults that participated in 150 minutes (or vigorous equivalent minutes) of physical activity per week.
'PA300R2_', #Adults that participated in 300 minutes (or vigorous equivalent minutes) of physical activity per week.
'PA30021_', #Adults that participated in 300 minutes (or vigorous equivalent minutes) of physical activity per week (2-levels).
'PASTRNG_', #Muscle Strengthening Recommendation
'PAREC1_', #Aerobic and Strengthening Guideline
'PASTAE1_', #Aerobic and Strengthening (2-level)
'LMTACT1_', #Limited usual activities 3k nan
'LMTWRK1_', #Limited work activities 3k nan
'LMTSCL1_', #Limited social activities 3k nan
'RFSEAT2_', #Always or Nearly Always Wear Seat Belts Calculated Variable
'RFSEAT3_', #Always Wear Seat Belts Calculated Variable
'FLSHOT6_', #Adults aged 65+ who have had a flu shot within the past year, 283k nan
'PNEUMO2_', #Adults aged 65+ who have ever had a pneumonia vaccination 283k nan
'AIDTST3_' #Adults who have ever been tested for HIV 43k nan
]





#Helper functions

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


def expand_column(col, col_name):
    '''
    Expand a 1D column vector into a N array where N is the number of unique values in col.
    Each column in the output array is a binary indicator (0 or 1) of whether the corresponding
    entry in col matches the unique value for that column.
    col: 1D numpy array of categorical values

    #Missing values are handled by adding an additional binary indicator column and set to 0.
    '''
    is_missing = np.isnan(col).astype(np.int8)
    has_missing = np.any(is_missing)

    col = np.where(is_missing, 0, col)

    unique_values = np.unique(col)

    if unique_values.size > 10 and not has_missing:
        return None, None
    elif unique_values.size > 10 and has_missing:
        expanded = np.column_stack([col, is_missing])
        col_names = [f"{col_name}_is_missing"]
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

def expand_dataset_col(col_list, path):
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
        expanded_col, col_names = expand_column(x_train_subset[:, i], col_name)
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