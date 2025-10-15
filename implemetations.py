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
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: compute loss by MSE
    # ***************************************************
    #raise NotImplementedError   
    n, = np.shape(y)
    
    MSE = (1.0/(2*n)) * np.sum(np.square(y-np.dot(tx,w)))
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
    n, = np.shape(y)
    e = y - np.dot(tx,w)
    grad = -1.0/n * np.dot(tx.T,e)
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
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        
        grad = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)
        w = w - gamma * grad
        # store w and loss
        ws.append(w)
        losses.append(loss)

    return losses[-1], ws[-1]

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    pass

def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.

    >>> least_squares(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]))
    (array([ 0.21212121, -0.12121212]), 8.666684749742561e-33)
    """
    a = np.dot(tx.T, tx)
    b = np.dot(tx.T, y)
    w = np.linalg.solve(a, b)
    e = y - np.dot(tx, w)
    #mse = 1/(2*len(y)) * np.dot(e.T, e)
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
    mse = 1/(2*len(y)) * np.dot(e.T, e)
    return w, mse

def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a non-negative loss

    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(4).reshape(2, 2)
    >>> w = np.c_[[2., 3.]]
    >>> round(calculate_loss(y, tx, w), 8)
    1.52429481
    """
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]
    assert y.shape[1] == 1
    assert w.shape[1] == 1

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
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        loss, gradient = calculate_loss(y, tx, w), calculate_gradient(y, tx, w)
        w = w - gamma * gradient
        ws.append(w)
        losses.append(loss)

    return losses[-1], ws[-1]

def sigmoid(t):
    """apply sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array

    >>> sigmoid(np.array([0.1]))
    array([0.52497919])
    >>> sigmoid(np.array([0.1, 0.1]))
    array([0.52497919, 0.52497919])
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

    >>> np.set_printoptions(8)
    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(6).reshape(2, 3)
    >>> w = np.array([[0.1], [0.2], [0.3]])
    >>> calculate_gradient(y, tx, w)
    array([[-0.10370763],
           [ 0.2067104 ],
           [ 0.51712843]])
    """
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]
    assert y.shape[1] == 1
    assert w.shape[1] == 1
    
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

    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(6).reshape(2, 3)
    >>> w = np.array([[0.1], [0.2], [0.3]])
    >>> lambda_ = 0.1
    >>> loss, gradient = penalized_logistic_regression(y, tx, w, lambda_)
    >>> round(loss, 8)
    0.62137268
    >>> gradient
    array([[-0.08370763],
           [ 0.2467104 ],
           [ 0.57712843]])
    """
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]
    assert y.shape[1] == 1
    assert w.shape[1] == 1

    loss = calculate_loss(y, tx, w)
    gradient = calculate_gradient(y, tx, w) + lambda_*2 * w
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
        losses: list of loss values
        ws: list of weights
    """
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        loss, gradient = penalized_logistic_regression(y, tx, w, lambda_)
        w = w - gamma * gradient
        ws.append(w)
        losses.append(loss)

    return losses[-1], ws[-1]
