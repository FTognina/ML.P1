import numpy as np
from custom_func import helpers_yass as hp
from functools import partial

def  mean_squared_error_gd(y,tx,initial_w, max_iters,gamma):
    """Linear regression using gradient descent
    Args: 
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        initial_w: initial guess of weights, numpy array of shape(D,), D is the number of features.
        max_iters: a scalar representing the number of iteration
        gamma: a scalar representing the stepsize 
    
    Returns:
        loss: scalar , the final MSE for GD
        w_star: numpy array of shape (D, ), the resulting model parameters
    """

    w_star = hp.gradient_descent(y ,tx ,initial_w ,max_iters ,gamma ,hp.lin_reg_gradient )
    loss = hp.mse(y ,tx ,w_star )
    return loss,w_star

def mean_squared_error_sgd(y,tx,initial_w, max_iters,gamma):
    """Linear regression using stochastic gradient descent
    Args: 
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        initial_w: initial guess of weights, numpy array of shape(D,), D is the number of features.
        max_iters: a scalar representing the number of iteration
        gamma: a scalar representing the stepsize 
    
    Returns:
        loss: scalar , the final MSE for SGD
        w_star: numpy array of shape (D, ), the optimal model parameters
    """

    w_star = hp.stochastic_gradient_descent(y ,tx ,initial_w ,max_iters ,gamma ,hp.lin_reg_gradient )
    loss = hp.mse(y ,tx ,w_star )
    return loss,w_star

def least_squares(y,tx):
    """Least squares regression using normal equations

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        loss: a scalar denoting the computed loss
        w: optimal weights, numpy array of shape(D,), D is the number of features.
    """
    w = hp.least_squares(y,tx)
    loss = hp.mse(y,tx,w) 

    return loss,w

def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        loss: a scalar denoting the computed loss
        w: optimal weights, numpy array of shape(D,), D is the number of features.
    """

    w = hp.ridge_regression(y,tx,lambda_)
    loss = hp.mse(y,tx,w) 

    return loss,w
    
def logistic_regression(y ,tx ,initial_w ,max_iters ,gamma ):
    """Logistic regression using gradient descent
    Args: 
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        initial_w: initial guess of weights, numpy array of shape(D,), D is the number of features.
        max_iters: a scalar representing the number of iteration
        gamma: a scalar representing the stepsize 
    
    Returns:
        loss: a scalar denoting the computed loss ( negative log likelihood )
        w_star: numpy array of shape (D, ), the resulting model parameters
    """
    w_star = hp.gradient_descent(y ,tx ,initial_w ,max_iters ,gamma ,hp.log_reg_gradient )
    loss = hp.neg_log_likelihood(y ,tx ,w_star )
    return loss,w_star

def reg_logistic_regression(y ,tx ,lambda_ ,initial_w ,max_iters ,gamma ):
    """Logistic regression using gradient descent
    Args: 
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: a scalar, used for regulazing
        initial_w: initial guess of weights, numpy array of shape(D,), D is the number of features.
        max_iters: a scalar representing the number of iteration
        gamma: a scalar representing the stepsize 
    
    Returns:
        loss: a scalar denoting the computed loss ( negative log likelihood )
        w_star: numpy array of shape (D, ), the resulting model parameters
    """
    grad_func = partial(hp.reg_log_reg_gradient, lambda_=lambda_)
    w_star = hp.gradient_descent(y ,tx ,initial_w ,max_iters ,gamma , grad_func)
    loss = hp.neg_log_likelihood(y ,tx ,w_star )
    return loss,w_star