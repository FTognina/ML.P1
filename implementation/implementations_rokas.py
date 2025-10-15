import numpy as np

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        w: last weight vector of the method
        loss: loss value of last weights used
    """

    w = initial_w

    for n_iter in range(max_iters):

        error = y - tx @ w
        gradient = - (tx.T @ error) / len(y)
        w = w - gamma * gradient
        loss = 0.5 * np.mean((y - tx @ w)**2)

        print(f"GD iter. {n_iter}/{max_iters - 1}: loss={loss:.6f}")
        
    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, batch_size, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD).

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        w: last weight vector of the method
        loss: loss value of last weights used
    """
    w = initial_w

    for n_iter in range(max_iters):

        idx = np.random.choice(tx.shape[0], size=batch_size, replace=False)
        X = tx[idx]
        Y = y[idx]
        error = Y - X @ w
        gradient = - (X.T @ error) / len(idx)
        w = w - gamma * gradient
        loss = 0.5 * np.mean((Y - X @ w)**2)

        print(f"SGD iter. {n_iter}/{max_iters - 1}: loss={loss:.6f}")
        
    return w, loss


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
    w = np.linalg.solve(tx.T @ tx, tx.T @ y) 
       
    MSE = np.mean((y - tx @ w)**2)
    return w, MSE

def ridge_regression(y, tx, lambda_):
    """implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.

    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 0)
    array([ 0.21212121, -0.12121212])
    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 1)
    array([0.03947092, 0.00319628])
    """
    D = tx.shape[1]
    I = np.eye(D)
    return np.linalg.solve(tx.T @ tx + (2 * lambda_ * I * len(y)), tx.T @ y) 


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    

    w = initial_w

    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]
    
    N = y.shape[0]

    for n_iter in range(max_iters):
        z = tx @ w
    
        loss = (np.sum(np.logaddexp(0, z) - y * z)) / N
        
        gradient = (tx.T @ ((1/(1 + np.exp(-z))) - y)) / N
        
        w = w - gradient * gamma

        print(f"LR iter. {n_iter}/{max_iters - 1}: loss={loss:.6f}")
        
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    
    w = initial_w

    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]
    
    N = y.shape[0]

    for n_iter in range(max_iters):
        z = tx @ w
    
        loss = (np.sum(np.logaddexp(0, z) - y * z)) / N
        
        gradient = ((tx.T @ ((1/(1 + np.exp(-z))) - y)) / N ) + 2 * lambda_ * w
        
        w = w - gradient * gamma

        print(f"regLR iter. {n_iter}/{max_iters - 1}: loss={loss:.6f}")
        
    return w, loss



