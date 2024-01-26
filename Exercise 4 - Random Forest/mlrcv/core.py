import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def accuracy(y, y_pred):
    if len(y_pred.shape) == 2:
        correct = np.sum(y == np.argmax(y_pred, axis=-1))
    else:
        correct = np.sum(y == y_pred)

    return (correct / y.shape[0]) * 100.

def plot_model(x, y, model):
    y = np.expand_dims(y, axis=-1)

    grid = np.ones((30,30, 4))
    grid[:,:,-1] = .5

    x1max, x1min = 2., -1.
    x2max, x2min = 2., -1.

    x1_step = (x1max - x1min) / 30.
    x2_step = (x2max - x2min) / 30.

    if model is not None:
        for i in range(30):
            for j in range(30):
                grid_x1, grid_x2 = x1min + i * x1_step, x2min + j * x2_step
                grid_pred_ = model.predict_y(np.array([[grid_x1, grid_x2]]))
                grid_pred = np.zeros((1,3))

                if grid_pred_.shape[-1] != 1:
                    grid_pred[0] += np.array([1., 0., 0.]) * grid_pred_[0][0]
                    grid_pred[0] += np.array([0., 1., 0.]) * grid_pred_[0][1]
                    grid_pred[0] += np.array([0., 0., 1.]) * grid_pred_[0][2]
                    grid_pred[0] += np.array([1., 1., 0.]) * grid_pred_[0][3]
                else:
                    if grid_pred_[0] != 3:
                        grid_pred[0, int(grid_pred_[0])] = 1.
                    else:
                        grid_pred[0, 0] = 1.
                        grid_pred[0, 1] = 1.

                
                grid[i,j,:3] = grid_pred

    fig = plt.figure()  
    ax = fig.add_subplot(111)
    ax.set_xlim([-1.,2.])
    ax.set_ylim([-1.,2.])

    cls1 = y[:,0] == 0
    cls2 = y[:,0] == 1
    cls3 = y[:,0] == 2
    cls4 = y[:,0] == 3

    plt.scatter(x[cls1,0], x[cls1,1], color='r')
    plt.scatter(x[cls2,0], x[cls2,1], color='g')
    plt.scatter(x[cls3,0], x[cls3,1], color='b')
    plt.scatter(x[cls4,0], x[cls4,1], color=(1.,1.,0.,1.))
    	
    plotlim = plt.xlim() + plt.ylim()

    if model is not None:
        ax.imshow(np.rot90(grid), interpolation='gaussian', extent=plotlim)  

    plt.show()

def split_data(x, y):
    """
    This function should split the X and Y data in training and validation sets

    Args:
        - x: input data
        - y: target data

    Returns:
        - x_train: input data used for training
        - y_train: target data used for training
        - x_val: input data used for validation
        - y_val: target data used for validation

    """
    # split the train data
    train_index = np.random.choice(len(x), int(len(x) * .7), replace=False)
    train_set = np.zeros((x.shape[0])).astype(bool)
    train_set[train_index] = True

    x_train = x[train_set]
    x_val = x[~train_set]
    y_train = y[train_set]
    y_val = y[~train_set]

    return x_train, y_train, x_val, y_val
