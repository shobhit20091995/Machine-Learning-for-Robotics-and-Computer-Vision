import numpy as np
from typing import Optional

def rmse(y: np.ndarray, y_pred: np.ndarray) -> float:
    """
    This function should calculate the root mean squared error given target y and prediction y_pred

    Args:
        - y(np.array): target data
        - y_pred(np.array): predicted data

    Returns:
        - err (float): root mean squared error between y and y_pred

    """
    
    N = len(y)
    squared_err = np.square(y - y_pred)
    mean_squared_err = np.sum(squared_err)/ N
    err = np.sqrt(mean_squared_err)   



    return err

def split_data(x: np.ndarray, y: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    This function should split the X and Y data in training, validation

    Args:
        - x: input data
        - y: target data

    Returns:
        - x_train: input data used for training
        - y_train: target data used for training
        - x_val: input data used for validation
        - y_val: target data used for validation
       
     
 

    """
    sample_size = int(len(x) * 0.8) # we take the 80% of data as trainning set
    
   
    
    x_train = x[:sample_size]
    y_train = y[:sample_size]
    x_val = x[sample_size:]
    y_val = y[sample_size:]

    return x_train, y_train, x_val, y_val