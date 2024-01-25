import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_data_multi_class(x, y):
    fig = plt.figure()  
    ax = fig.add_subplot(111)
    ax.set_xlim([-1.,2.])
    ax.set_ylim([-1.,2.])

    cls1 = y[:,0] == 0
    cls2 = y[:,0] == 1
    cls3 = y[:,2] == 1

    plt.scatter(x[cls1,0], x[cls1,1], color='r', marker='*')
    plt.scatter(x[cls2,0], x[cls2,1], color='g', marker='^')
    plt.scatter(x[cls3,0], x[cls3,1], color='b', marker='o')

    plt.show()

def plot_data_binary(x, y):
    fig = plt.figure()  
    ax = fig.add_subplot(111)
    ax.set_xlim([-1.,2.])
    ax.set_ylim([-1.,2.])

    cls1 = y[:,0] == 0
    cls2 = y[:,0] == 1

    plt.scatter(x[cls1,0], x[cls1,1], color='r', marker='*')
    plt.scatter(x[cls2,0], x[cls2,1], color='g', marker='^')

    plt.show()

def reg_line(x, theta):
    return -(theta[0] + theta[1]*x)/theta[2]

def plot_regression_binary(x, y, model):
    if len(y.shape) == 1:
        y = y[:, np.newaxis]

    assert y.shape[-1] == 1, f"y shape should be ({y.shape[0]},1) or ({y.shape[0]},) but it is {y.shape}"

    grid = np.zeros((100,100, 4))
    grid[:,:,-1] = .5

    x1max, x1min = 2., -1.
    x2max, x2min = 2., -1.

    x1_step = (x1max - x1min) / 100.
    x2_step = (x2max - x2min) / 100.

    for i in range(100):
        for j in range(100):
            grid_x1, grid_x2 = x1min + i * x1_step, x2min + j * x2_step
            grid_pred = model.predict_y(np.array([[1., grid_x1, grid_x2]]))
            grid[i,j,1] = grid_pred
            grid[i,j,0] = -grid_pred + 1.

    fig = plt.figure()  
    ax = fig.add_subplot(111)
    ax.set_xlim([-1.,2.])
    ax.set_ylim([-1.,2.])

    cls1 = y[:,0] == 0
    cls2 = y[:,0] == 1

    plt.plot([-1., 2.], [reg_line(-1., model.theta), reg_line(2., model.theta)], color='r')
    plt.scatter(x[cls1,0], x[cls1,1], color='r', marker='*')
    plt.scatter(x[cls2,0], x[cls2,1], color='g', marker='^')
    plotlim = plt.xlim() + plt.ylim()

    ax.imshow(np.rot90(grid), interpolation='bilinear', extent=plotlim)  

    plt.show()

def plot_regression_multi_class(x, y, model):
    assert y.shape[-1] == 3, f"y shape should be ({y.shape[0]}, 3) but it is {y.shape}"

    grid = np.ones((100,100, 4))
    grid[:,:,-1] = .5

    x1max, x1min = 2., -1.
    x2max, x2min = 2., -1.

    x1_step = (x1max - x1min) / 100.
    x2_step = (x2max - x2min) / 100.

    for i in range(100):
        for j in range(100):
            grid_x1, grid_x2 = x1min + i * x1_step, x2min + j * x2_step
            grid[i,j,:3] = model.predict_y(np.array([[1., grid_x1, grid_x2]]))

    fig = plt.figure()  
    ax = fig.add_subplot(111)
    ax.set_xlim([-1.,2.])
    ax.set_ylim([-1.,2.])

    cls1 = y[:,0] == 1
    cls2 = y[:,1] == 1
    cls3 = y[:,2] == 1

    plt.plot([-1., 2.], [reg_line(-1., model.theta_class[0]), reg_line(2., model.theta_class[0])], color='r')
    plt.plot([-1., 2.], [reg_line(-1., model.theta_class[1]), reg_line(2., model.theta_class[1])], color='g')
    plt.plot([-1., 2.], [reg_line(-1., model.theta_class[2]), reg_line(2., model.theta_class[2])], color='b')

    plt.scatter(x[cls1,0], x[cls1,1], color='r', marker='*')
    plt.scatter(x[cls2,0], x[cls2,1], color='g', marker='^')
    plt.scatter(x[cls3,0], x[cls3,1], color='b', marker='o')
    plotlim = plt.xlim() + plt.ylim()

    ax.imshow(np.rot90(grid), interpolation='bilinear', extent=plotlim)  

    plt.show()

def accuracy(y, y_pred):
    if y.shape[-1] > 1:
        y = np.argmax(y, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
    else:
        y_pred = (y_pred > .5).astype(np.float32)

    correct = np.sum(y == y_pred).astype(float)

    return (correct / y.shape[0]) * 100.