import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlrcv.core import *
from mlrcv.decision_tree import *
from typing import Optional

class RandomTreeNode:
    def __init__(self, node_id: str, max_degree: int):
        """
        This function initializes the TreeNode class (already implemented):

        Args:
            - node_id (str): node id to identiy the current node
            - max_degree (int): max degree of the tree

        Returns:
        """
        self.max_attr_gain = -np.inf
        self.attr_split = None
        self.attr_split_val = None
        self.node_class = None
        self.children = {}
        self.leaf = False
        self.node_id = node_id
        self.max_degree = max_degree

    def infer_node(self, x: np.ndarray) -> float:
        """
        This function goes over the tree given the input data x and return the respective leaf class:

        Args:
            - x (np.ndarray): input data x to be checked over the tree

        Returns:
            - node_class (float): respective leaf class given input x
        """
        node_class = None

        return node_class

    def split_node(self, x: np.ndarray, y: np.ndarray, degree: int):
        """
        This function uses the current x and y data to split the tree nodes (left and right) given the information_gain
        calculated over the possible splits. Recursion stop condition will be when the current degree arrives at
        maximum degree (setting it as leaf):

        Args:
            - x (np.ndarray): input data x to be splited
            - y (np.ndarray): class labels of the input data x
            - degree (int): current node degree

        Returns:
        """

        return

    def attr_gain(self, x_attr: np.ndarray, y: np.ndarray) -> (float, float):
        """
        This function calculates the attribute gain. For the random tree case, the attr splits should be divided
        as in the decision tree, but then a subset should be random selected from it:

        Args:
            - x_attr (np.ndarray): input data x[attr] to be splitted
            - y (np.ndarray): labels of the input data x

        Returns:
            - split_gain (float): highest gain from the possible attributes splits
            - split_value (float): split value selected for x_attr attribute
        """
        split_gain = None
        split_value = None

        return split_gain, split_value

    def information_gain(self, y: np.ndarray, y_l: np.ndarray, y_r: np.ndarray) -> float:
        """
        This function calculates the attribute gain from the candidate splits y_l and y_r:

        Args:
            - y (np.ndarray): the full labels of the current node
            - y_l (np.ndarray): labels of the candidate left node split
            - y_r (np.ndarray): labels of the candidate right node split

        Returns:
            - I (float): information gain from the candidate splits y_l and y_r
        """
        I = None

        return I

    def entropy(self, y: np.ndarray) -> float:
        """
        This function calculates the entropy from the input labels set y:

        Args:
            - y (np.ndarray): the labels set to calculate the entropy

        Returns:
            - H (float): the entropy of the input labels set y
        """
        H = None

        return H

class RandomTree:
    def __init__(self, num_class: int, max_degree: int):
        """
        This function initializes the RandomTree class (already implemented):

        Args:
            - num_class (int): number of class from your data
            - max_degree (int): max degree of the tree

        Returns:
        """
        self.root = None
        self.max_degree = max_degree
        self.num_class = num_class
    
    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        This function fits the random tree with the training data x and labels y. Starting from root tree node,
        and iterate recursively over the nodes, split on left and right nodes:

        Args:
            - x (np.ndarray): the input data x
            - y (np.ndarray): labels of the input data x

        Returns:

        """
        pass

    def predict_y(self, x: np.ndarray) -> np.ndarray:
        """
        This function predicts y_pred class from the input x:

        Args:
            - x (np.ndarray): input data to be predicted by the tree

        Returns:
            - y_pred (np.ndarray): tree predictions over input x
        """
        y_pred = None

        return y_pred

    def eval(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        This function evaluate the model predicting y_pred from input x and calculating teh accuracy between y_pred and y:

        Args:
            - x (np.ndarray): input data to be predicted by the tree
            - y (np.ndarray): input class labels

        Returns:
            - acc (float): accuracy of the model
        """
        return accuracy(y, self.predict_y(x))

class RandomForest:
    def __init__(self, num_class: int, max_degree: int, trees_num: Optional[int] = 10, random_rho: Optional[float] = 1.0):
        """
        This function initializes the RandomForest class (already implemented):

        Args:
            - num_class (int): number of class from your data
            - max_degree (int): max degree of the tree
            - trees_num (int): number of random trees to be generated
            - random_rho (float): rho attribute to generate the random subset from the input data

        Returns:
        """
        self.max_degree = max_degree
        self.trees_num = trees_num
        self.random_rho = random_rho
        self.d_trees = [ RandomTree(num_class, self.max_degree) for _ in range(trees_num) ]
        self.num_class = num_class

    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        This function fits the random forest with the training data x and labels y. For each random tree fits the 
        data x and y:

        Args:
            - x (np.ndarray): the input data x
            - y (np.ndarray): labels of the input data x

        Returns:

        """
        pass

    def predict_y(self, x: np.ndarray) -> np.ndarray:
        """
        This function predicts y_pred from input data x. For each random tree get the predicted class,
        then, sum the predicts over a voting matrix, returning a data distribution:

        Args:
            - x (np.ndarray): the input data x to be predicted
        Returns:
            - y_pred (np.ndarray): the prediction y_pred from the input x
        """
        
        y_pred = None

        return y_pred

    def random_tree_data(self, x: np.ndarray, y: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        This function generates a random subset x_s of samples from the input data x and y where
        len(x_s) / len(x) = self.rho:

        Args:
            - x (np.ndarray): the input data x
            - y (np.ndarray): the labels of the input data x
        Returns:
            - x_s (np.ndarray): the subset of the input data x
            - y_s (np.ndarray): the correspondent subset of labels y
        """
        x_s = x
        y_s = y

        return x_s, y_s

    def eval(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        This function evaluate the model predicting y_pred from input x and calculating teh accuracy between y_pred and y:

        Args:
            - x (np.ndarray): input data to be predicted by the tree
            - y (np.ndarray): input class labels

        Returns:
            - acc (float): accuracy of the model
        """
        return accuracy(y, self.predict_y(x))
