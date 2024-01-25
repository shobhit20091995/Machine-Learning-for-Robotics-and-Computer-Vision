import numpy as np
from typing import Optional
from mlrcv.utils import *

class LinearRegression:
    def __init__(self):
        self.theta_0 = None
        self.theta_1 = None

    def calculate_theta(self, x: np.ndarray, y: np.ndarray):
        """
        This function should calculate the parameters theta0 and theta1 for the regression line

        Args:
            - x (np.array): input data
            - y (np.array): target data

        """
        
        
        z = np.column_stack((np.ones(len(x)), x)) # adding x_0 = 1 as discussed in lecture
        
        # Maximum Likelihood Estimator , calculating theta using normal equation
      

        X_T = np.transpose(z)
        X_T_X = np.dot(X_T, z)
        X_T_X_inv = np.linalg.inv(X_T_X)
        X_T_y = np.dot(X_T, y)
        theta = np.dot(X_T_X_inv, X_T_y)       
        
        self.theta0 = theta[0]
        self.theta1 = theta[1]
        pass

    def predict_y(self, x: np.ndarray) -> np.ndarray:
        """
        This function should use the parameters theta0 and theta1 to predict the y value given an input x

        Args:
            - x: input data

        Returns:
            - y_pred: y computed w.r.t. to input x and model theta0 and theta1

        """
         
        x = np.column_stack((np.ones(len(x)), x)) # adding 1 sothat we can have intercept       
       
        y_pred = np.dot(x, np.array([self.theta0, self.theta1]))

      

        return y_pred

class NonLinearRegression:
    def __init__(self):
        self.theta = None

    def calculate_theta(self, x: np.ndarray, y: np.ndarray, degree: Optional[int] = 2):
        """
        This function should calculate the parameters theta for the regression curve.
        In this case there should be a vector with the theta parameters (len(parameters)=degree + 1).

        Args:
            - x: input data
            - y: target data
            - degree (int): degree of the polynomial curve

        Returns:

        """
        # polynomial transformation
        X = np.ones((len(x), 1))
        for i in range(1, degree + 1):
            X = np.hstack((X, np.power(x, i).reshape((-1, 1))))

       
        # Maximum Likelihood Estimator , calculating theta using normal equation
        X_T = np.transpose(X)
        X_T_X = np.dot(X_T, X)
        X_T_X_inv = np.linalg.inv(X_T_X)
        X_T_y = np.dot(X_T, y)
        theta = np.dot(X_T_X_inv, X_T_y)       
        
        self.theta = theta
       
        
        pass

    def predict_y(self, x: np.ndarray) -> np.ndarray:
        """
        This function should use the parameters theta to predict the y value given an input x

        Args:
            - x: input data

        Returns:
            - y: y computed w.r.t. to input x and model theta parameters
        """
        
        X = np.ones((len(x), 1))
        for i in range(1, len(self.theta)):
            X = np.hstack((X, np.power(x, i).reshape((-1, 1))))
            
        
        y_pred = np.dot(X,(self.theta))


        return y_pred
