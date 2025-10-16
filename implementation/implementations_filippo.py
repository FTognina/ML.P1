import numpy as np


def compute_loss(y, tx, w):
    """Calculate the loss using either MSE or MAE.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    (n,) = np.shape(y)

    MSE = (1.0 / (2 * n)) * np.sum(np.square(y - np.dot(tx, w)))
    return MSE


def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.

    Returns:
        An numpy array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    (n,) = np.shape(y)
    e = y - np.dot(tx, w)
    grad = -1.0 / n * np.dot(tx.T, e)
    return grad


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
        ws: a list of length max_iters + 1 containing the model parameters as numpy arrays of shape (2, ),
            for each iteration of GD (as well as the final weights)
    """
    # Define parameters to store w and loss
    ws = [initial_w]
    w = initial_w
    losses = [compute_loss(y, tx, w)]
    for n_iter in range(max_iters):

        grad = compute_gradient(y, tx, w)
        w = w - gamma * grad
        loss = compute_loss(y, tx, w)
        # store w and loss
        ws.append(w)
        losses.append(loss)
    print(losses)
    return ws[-1], losses[-1]


def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient at w from a data sample batch of size B, where B < N, and their corresponding labels.

    Args:
        y: numpy array of shape=(B, )
        tx: numpy array of shape=(B,2)
        w: numpy array of shape=(2, ). The vector of model parameters.

    Returns:
        A numpy array of shape (2, ) (same shape as w), containing the stochastic gradient of the loss at w.
    """
    (n,) = np.shape(y)
    stoch_grad = np.zeros(w.shape)
    stoch_grad = compute_gradient(y, tx, w)
    return stoch_grad


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    """
    data_size = len(y)  # NUmber of data points.
    batch_size = min(data_size, batch_size)  # Limit the possible size of the batch.
    max_batches = int(
        data_size / batch_size
    )  # The maximum amount of non-overlapping batches that can be extracted from the data.
    remainder = (
        data_size - max_batches * batch_size
    )  # Points that would be excluded if no overlap is allowed.

    if shuffle:
        # Generate an array of indexes indicating the start of each batch
        idxs = np.random.randint(max_batches, size=num_batches) * batch_size
        if remainder != 0:
            # Add an random offset to the start of each batch to eventually consider the remainder points
            idxs += np.random.randint(remainder + 1, size=num_batches)
    else:
        # If no shuffle is done, the array of indexes is circular.
        idxs = np.array([i % max_batches for i in range(num_batches)]) * batch_size

    for start in idxs:
        start_index = start  # The first data point of the batch
        end_index = (
            start_index + batch_size
        )  # The first data point of the following batch
        yield y[start_index:end_index], tx[start_index:end_index]


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD).

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of SGD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of SGD
    """
    batch_size = 1
    # Define parameters to store w and loss
    ws = [initial_w]
    y_b, tx_b = next(batch_iter(y, tx, batch_size))
    w = initial_w
    losses = [compute_loss(y_b, tx_b, w)]

    for n_iter in range(max_iters):
        # batch_iter returns an iterator, so get the first batch using next()
        y_b, tx_b = next(batch_iter(y, tx, batch_size))
        stoch_grad = compute_stoch_gradient(y_b, tx_b, w)
        w = w - gamma * stoch_grad
        loss = compute_loss(y_b, tx_b, w)
        ws.append(w)
        losses.append(loss)
    return ws[-1], losses[-1]


def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.
    """
    a = np.dot(tx.T, tx)
    b = np.dot(tx.T, y)
    w = np.linalg.solve(a, b)
    e = y - np.dot(tx, w)
    # mse = 1/(2*len(y)) * np.dot(e.T, e)
    mse = compute_loss(y, tx, w)
    return w, mse


def ridge_regression(y, tx, lambda_):
    """implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
    """
    a = np.dot(tx.T, tx) + 2 * tx.shape[0] * lambda_ * np.eye(tx.shape[1])
    b = np.dot(tx.T, y)
    w = np.linalg.solve(a, b)
    e = y - np.dot(tx, w)
    mse = 1 / (2 * len(y)) * np.dot(e.T, e)
    return w, mse


def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a non-negative loss (scalar)
    """

    # compute the loss: negative log likelihood
    y_hat = sigmoid(tx @ w)
    loss = -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
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

    return 1 / (1 + np.exp(-t))


def calculate_gradient(y, tx, w):
    """compute the gradient of loss.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a vector of shape (D, 1)
    """

    y_hat = sigmoid(tx @ w)
    gradient = tx.T @ (y_hat - y) / y.shape[0]
    return gradient


def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss and gradient.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        lambda_: scalar

    Returns:
        loss: scalar number
        gradient: shape=(D, 1)
    """
    gradient = calculate_gradient(y, tx, w) + lambda_ * 2 * w
    loss = calculate_loss(y, tx, w)

    return float(loss), gradient


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent.

    Args:
        y: numpy array of shape (N, 1)
        tx: numpy array of shape (N, D)
        lambda_: scalar
        initial_w: numpy array of shape (D, 1)
        max_iters: scalar
        gamma: scalar

    Returns:
        ws: last of weights
        losses: last of loss values
    """
    ws = [initial_w]
    w = initial_w
    loss, gradient = penalized_logistic_regression(y, tx, w, lambda_)
    losses = []
    for n_iter in range(max_iters):
        loss, gradient = penalized_logistic_regression(y, tx, w, lambda_)
        w = w - gamma * gradient
        ws.append(w)
        losses.append(loss)
    loss, gradient = penalized_logistic_regression(y, tx, w, lambda_)
    w = w - gamma * gradient
    losses.append(loss)

    return ws[-1], np.asarray(losses[-1])