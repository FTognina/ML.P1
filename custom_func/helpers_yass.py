import numpy as np

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



def mse(y, tx, w):
    """compute the loss by mse.
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        w: weights, numpy array of shape(D,), D is the number of features.

    Returns:
        mse: scalar corresponding to the mse 

    """
    f = tx @ (w)
    l = (y-f)**2
    return np.mean(l)
   
    
def mae(y,tx,w):
    """compute the loss by mae.
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        w: weights, numpy array of shape(D,), D is the number of features.

    Returns:
        mae: scalar corresponding to the mae 

    """
    f = tx @ (w)
    l = np.abs(y-f)
    return np.mean(l)
    

def rmse(y, tx, w):
    """compute the loss by rmse.
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        w: weights, numpy array of shape(D,), D is the number of features.

    Returns:
        rmse: scalar corresponding to the rmse 

    """
    f = tx @ (w)
    l = (y-f)**2
    mse = np.mean(l)
    return np.sqrt( 2 * mse )
   

def lin_reg_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.

    Returns:
        An numpy array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    e =  y - tx@w
    grad =(tx.T @ e)/len(y)
    
    return grad
    

def sigmoid(t):
    """apply sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array
    """
    return 1 / (1 + np.exp( -t ))

def neg_log_likelihood(y, tx, w):
    """compute the cost by negative log likelihood.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a non-negative loss
    """
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]

    z = tx @ w
    return np.mean(np.log(1 + np.exp(z)) - y * z)

def log_reg_gradient(y, tx, w):
    """compute the gradient of loss.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a vector of shape (D, 1)
    """
    return ( tx.T @ ( sigmoid( tx @ w ) - y ) ) / len(y)

def reg_term(lambda_,w):
    """Calculates the regularization term needed for penalization

    Args:
        lambda_:scalar
        w: weights of shape(D,)

    Returns:
        regularization term
    """
    return 2*lambda_*w


def reg_log_reg_gradient(y, tx, w, lambda_):
    """compute the gradient of loss.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a vector of shape (D, 1)
    """
    return ( tx.T @ ( sigmoid( tx @ w ) - y ) ) / len(y) + reg_term(lambda_,w)

def least_squares(y, tx):
    """Calculate the least squares solution.
       returns optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.

    """
    w = np.linalg.solve((tx.T @ tx) ,( tx.T @ y))
    return w

def ridge_regression(y, tx, lambda_):
    """implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
    """
    l = 2*tx.shape[1] *  lambda_
    w = np.linalg.solve(  tx.T @ tx  +  l * np.eye( tx.shape[1] )  , tx.T @ y )
    
    return w

def gradient_descent(y, tx, initial_w, max_iters, gamma, grad_func):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
        grad_func: a function that calculates the gradient

    Returns:
        ws: a list of length max_iters + 1 containing the model parameters as numpy arrays of shape (2, ),
            for each iteration of GD (as well as the final weights)
    """
    ws = [initial_w]
    w = initial_w

    for n_iter in range(max_iters):
            
        grad = grad_func(y,tx,w)    
        w = w + gamma*grad
        ws.append(w)
        
    return ws


def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma,grad_func):
    """The Stochastic Gradient Descent algorithm (SGD).

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize
        grad_func: a function that calculates the gradient

    Returns:
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of SGD
    """

    ws = [initial_w]
    w = initial_w

    for _ in range(max_iters):
   
        for i in batch_iter(y,tx,batch_size):

            y1,tx1 = i
            grad = grad_func(y1,tx1,w)
            w = w+ gamma*grad
            ws.append(w)
        
    return ws



