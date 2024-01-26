import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def plot_clusters(x, model):
    y_pred = model.predict_c(x)
    classes = np.unique(y_pred)

    for cls_ in classes:
        cls_ind = y_pred == cls_
    
        plt.scatter(x[cls_ind,0], x[cls_ind,1], color=cm.brg(cls_ / classes.max()))

    for center in model.k_centers:
        plt.scatter(center[0], center[1], s=150., color=(.9, .9, 0), marker='*')

    plt.show()

def plot_dataset(x):
    plt.scatter(x[:,0], x[:,1])


def accuracy(y, y_pred):
    correct = np.sum(y == y_pred)

    return (correct / y.shape[0]) * 100.