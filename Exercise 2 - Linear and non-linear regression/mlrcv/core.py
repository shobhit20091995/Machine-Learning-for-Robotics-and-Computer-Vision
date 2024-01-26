import numpy as np
import matplotlib.pyplot as plt

def plot_regression(x, y, y_pred=None):
    assert len(x.shape) == 1, f"y shape should be ({x.shape[0]},) but it is {x.shape}"
    assert len(y.shape) == 1, f"y shape should be ({y.shape[0]},) but it is {y.shape}"

    plt.scatter(x, y)

    if y_pred is not None:
        assert len(y_pred.shape) == 1, f"y_pred shape should be ({y_pred.shape[0]},) but it is {y_pred.shape}"
        x_y = np.concatenate((x[:,np.newaxis], y_pred[:,np.newaxis]), axis=-1)
        x_y = x_y[x_y[:,0].argsort()]

        plt.plot(x_y[:,0], x_y[:,1], color='r')
    plt.show()