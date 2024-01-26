import numpy as np
import matplotlib.pyplot as plt
from mlrcv.core import *
from mlrcv.utils import *
import typing

class LogisticRegression:
    def __init__(self, learning_rate: float, epochs: int):
        """
        This function should initialize the model parameters

        Args:
            - learning_rate (float): the lambda value to multiply the gradients during the training parameters update
            - epochs (int): number of epochs to train the model

        Returns:
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.theta = None

    def predict_y(self, x: np.ndarray) -> np.ndarray:
        """
        This function should use the parameters theta to predict the y class given an input x

        Args:
            - x (np.ndarray): input data to predict y classes

        Returns:
            - y_pred (np.ndarray): the model prediction of the input x
            
        """
        
        
        if self.theta is None:
            raise ValueError("The model has not been trained yet, theta is None")
            
        #x = np.insert(x, 0, 1, axis=1)
                
        z = np.dot(x, self.theta)

        #y_pred = 1 / (1 + np.exp(-z))
        
        #y_pred = np.round(y_pred)

        y_pred = sigmoid(z)  
        prob = sigmoid(z)    




        # return 1 if the prob  is greater than 0.5 and otherwise 0
        y_pred = (prob >= 0.5).astype(int)
        

        return y_pred

    def first_derivative(self, x: np.ndarray, y_pred: np.ndarray, y: np.ndarray) -> np.ndarray:
        
        """
        This function should calculate the first derivative w.r.t. input x, predicted y and true labels y

        Args:
            - x (np.ndarray): input data
            - y_pred (np.ndarray): predictions of x
            - y (np.ndarray): true labels of x

        Returns:
            - der (np.ndarray): first derivative value      """
      
        
        
        m = len(y)
        
        der = 1 / m * np.dot(x.transpose(), (y_pred - y))        



        return der

   

    def train_model(self, x: np.ndarray, y: np.ndarray):
        """
        This function should use train the model to find theta parameters that best fit the data

        Args:
            - x (np.ndarray): input data to predict y classes
            - y (np.ndarray): true labels of x
        """
        rows = x.shape[0]
        cols = x.shape[1]
        self.theta = np.zeros((cols+1, 1))
        #self.theta = np.random.uniform(0, 0, (3, 3))
        x = np.append(np.ones((rows, 1)), x, axis=1)
        
        for i in range(self.epochs):
            y_pred = self.predict_y(x)
            der = self.first_derivative(x, y_pred, y) 
            self.theta  -= (self.learning_rate * der)     

            
 

    def eval(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        This function should use evaluate the model and output the accuracy of the model
        (accuracy function already implemented)

        Args:
            - x (np.ndarray): input data to predict y classes
            - y (np.ndarray): true labels of x

        Returns:
            - acc (float): accuracy of the model (accuracy(y,y_pred)) note: accuracy function already implemented in core.py
        """
        
       # acc = None
        rows = x.shape[0]
        cols = x.shape[1]
        x = np.append(np.ones((rows, 1)), x, axis=1)
        
        y_pred = self.predict_y(x)
        acc = accuracy(y, y_pred)

        return acc

class MultiClassLogisticRegression:
    def __init__(self, learning_rate: float, epochs: int):
        """
        This function should initialize the model parameters

        Args:
            - learning_rate (float): the lambda value to multiply the gradients during the training parameters update
            - epochs (int): number of epochs to train the model

        Returns:
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.theta_class = None

    def predict_y(self, x: np.ndarray) -> np.ndarray:
        """
        This function should use the parameters theta to predict the y class given an input x

        Args:
            - x (np.ndarray): input data to predict y classes

        Returns:
            - y_pred (np.ndarray): the model prediction of the input x
        """
        if self.theta is None:
            raise ValueError("The model has not been trained yet, theta is None")
            
        
                
        z = np.dot(x, self.theta)
        prob = softmax(z)
        
     
        #y_pred = np.argmax(prob, axis=1)
        
        y_pred = prob

        return y_pred

    def first_derivative(self, x: np.ndarray, y_pred: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        This function should calculate the first derivative w.r.t. input x, predicted y and true labels y,
        for each possible class.

        Args:
            - x (np.ndarray): input data
            - y_pred (np.ndarray): predictions of x
            - y (np.ndarray): true labels of x

        Returns:
            - der: first derivative value
        """
        m = len(y)
        
        der = 1 / m * np.dot(x.transpose(), (y_pred - y))

        return der

    def train_model(self, x: np.ndarray, y: np.ndarray):
        """
        This function should use train the model to find theta_class parameters (multiclass) that best fit the data

        Args:
            - x (np.ndarray): input data to predict y classes
            - y (np.ndarray): true labels of x
        """
        
        rows = x.shape[0]
        cols = x.shape[1]
        
        self.theta = np.zeros((3, 3))
        x = np.append(np.ones((rows, 1)), x, axis=1)
        
        for i in range(self.epochs):
            y_pred = self.predict_y(x)
            
            der = self.first_derivative(x, y_pred, y) 
            self.theta  -= (self.learning_rate * der)

    def eval(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        This function should use evaluate the model and output the accuracy of the model
        (accuracy function already implemented)

        Args:
            - x (np.ndarray): input data to predict y classes
            - y (np.ndarray): true labels of x

        Returns:
            - acc (float): accuracy of the model (accuracy(y,y_pred)) note: accuracy function already implemented in core.py
        """

        rows = x.shape[0]
        cols = x.shape[1]
        x = np.append(np.ones((rows, 1)), x, axis=1)
        
        y_pred = self.predict_y(x)
        acc = accuracy(y, y_pred)

        return acc