from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

def plot_compare(mag_check, check=None):
    if check == 'magnitude':
        GT = np.array([[0.,0.,0.,0.,0.,0.,0.,0.],
                        [0.,0.,360.62445841,0.,0., 0.,0.,0.],
                        [0.,360.62445841,0.,360.62445841,0.,0.,0.,0.],
                        [0.,0.,360.62445841,0.,360.62445841,0.,0.,0.],
                        [0.,0.,0.,360.62445841,0.,360.62445841,0.,0.],
                        [0.,0.,0.,0.,360.62445841,0.,360.62445841,0.],
                        [0.,0.,0.,0.,0.,360.62445841,0.,0.],
                        [0.,0.,0.,0.,0.,0.,0.,0.]])
    elif check == 'angle':
        GT = np.array([[0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 135., 0., 0., 0., 0., 0.],
                        [0., 45., 0., 135., 0., 0., 0., 0.],
                        [0., 0., 45., 0., 135., 0., 0., 0.],
                        [0., 0., 0., 45., 0., 135., 0., 0.],
                        [0., 0., 0., 0.,45., 0., 135., 0.],
                        [0., 0., 0., 0., 0., 45., 0., 0.],
                        [0., 0., 0., 0., 0.,  0., 0., 0.]])
    elif check == 'hog':
        GT = np.array([[255., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 255., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 255., 0., 0., 0., 0., 255.],
                        [0., 0., 0., 255., 0., 255., 255., 0.],
                        [0., 0., 0., 255., 0., 0., 0., 0.],
                        [0., 255., 255., 0., 0., 255., 0., 0.],
                        [255., 0., 0., 0., 0., 0., 255., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 255.]])
        

    plt.imshow(np.concatenate((mag_check, GT.max()*np.ones((GT.shape[0],1)), GT), axis=1), cmap='gray')

def bins_to_feats(cells_bins):
    feats = []

    for i in range(cells_bins.shape[0]-1):
        for j in range(cells_bins.shape[1]-1):
            tmp = []
            tmp.append(cells_bins[i,j])
            tmp.append(cells_bins[i+1,j])
            tmp.append(cells_bins[i,j+1])
            tmp.append(cells_bins[i+1,j+1])

            tmp -= np.mean(tmp)
            feats.append(tmp.flatten())

    return np.array(feats).flatten()

def img_to_cell(img, cell_size=8):
    cells_x = int(img.shape[0] / cell_size)
    cells_y = int(img.shape[1] / cell_size)

    img_cells = np.zeros((cells_x, cells_y, cell_size, cell_size))

    for i, x_cell in enumerate(np.split(img, cells_x, axis=0)):
        for j, y_cell in enumerate(np.split(x_cell, cells_y, axis=1)):
            img_cells[i][j] = y_cell

    return img_cells

def build_hog_image(hog_img, cells_bins, cell_size=8, step_size=20.):
    max_mag = cells_bins.max()
    for x in range(cells_bins.shape[0]):
        for y in range(cells_bins.shape[1]):
            cell_mag = cells_bins[x][y] / max_mag

            angle = 0.
            for magnitude in cell_mag:
                radian = np.radians(angle)
                x1 = int(x * cell_size + cell_size / 2 + magnitude * 2*cell_size * np.cos(radian))
                y1 = int(y * cell_size + cell_size / 2 + magnitude * 2*cell_size * np.sin(radian))
                x2 = int(x * cell_size + cell_size / 2 - magnitude * 2*cell_size * np.cos(radian))
                y2 = int(y * cell_size + cell_size / 2 - magnitude * 2*cell_size * np.sin(radian))

                cv2.line(hog_img, (y1, x1), (y2, x2), int(255 * np.sqrt(magnitude)))
                angle += step_size

    return hog_img

def load_image(img_file):
    img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 256))

    img = np.power(img / 255., 1) * 255

    return img

def save_image(image, fname):
    if not os.path.isdir('./data/hog'):
        os.makedirs('./data/hog')

    cv2.imwrite(os.path.join('./data/hog/', fname.split('/')[-1]), image)

def mags_diag(shape_size):
    mags = np.zeros((shape_size,shape_size))

    for i in range(mags.shape[0]):
        for j in range(mags.shape[1]):
            if i == j:
                mags[i,j] = 255.

    return mags

CHECK = mags_diag(8)

def load_data(datapath):
    person_data = np.asarray([os.path.join(datapath, 'person', fname) for fname in os.listdir(os.path.join(datapath, 'person'))])
    non_person_data = np.asarray([os.path.join(datapath, 'non_person', fname) for fname in os.listdir(os.path.join(datapath, 'non_person'))])

    label_person = np.ones((person_data.shape[0],))
    label_non_person = np.zeros((non_person_data.shape[0],))

    labels = np.concatenate((label_person, label_non_person), axis=0)
    data = np.concatenate((person_data, non_person_data), axis=0)

    np.random.seed(42)
    val_data = np.random.choice(len(labels), 36, replace=False)

    return (data[~val_data], labels[~val_data]), (data[val_data], labels[val_data])


def train_classifier(train_data, train_labels):
    classifier = make_pipeline(StandardScaler(), RandomForestClassifier(max_depth=10, n_estimators=10, max_features=1))
    classifier.fit(normalize(train_data), train_labels)

    return classifier

def eval_classifier(classifier, val_data, val_labels):
    print(f'accuracy: {(val_labels==classifier.predict(normalize(val_data))).sum() / len(val_data)}')