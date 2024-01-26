import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms

transform = transforms.Compose([
    transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05)
])

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0

    return h

def imshow(img, text=None):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if text is not None:
        plt.title(text)
    plt.show()

def plot_img_bb_test(img, lbl, map_):
    fig, axs = plt.subplots(2,2)
    axs[0][0].imshow(img[0])
    axs[1][0].imshow(img[1])
    axs[0][1].imshow(map_[0][:,:,0], cmap=plt.get_cmap('viridis'))
    axs[1][1].imshow(map_[1][:,:,0], cmap=plt.get_cmap('viridis'))
    for i, (im, lbl_) in enumerate(zip(img, lbl)):
        for l in lbl_:
            axs[i][0].plot((l['bndbox']['xmin']*4., l['bndbox']['xmin']*4.), (l['bndbox']['ymin']*4., l['bndbox']['ymax']*4.), color='green')
            axs[i][0].plot((l['bndbox']['xmax']*4., l['bndbox']['xmax']*4.), (l['bndbox']['ymin']*4., l['bndbox']['ymax']*4.), color='green')
            axs[i][0].plot((l['bndbox']['xmin']*4., l['bndbox']['xmax']*4.), (l['bndbox']['ymin']*4., l['bndbox']['ymin']*4.), color='green')
            axs[i][0].plot((l['bndbox']['xmin']*4., l['bndbox']['xmax']*4.), (l['bndbox']['ymax']*4., l['bndbox']['ymax']*4.), color='green')
            axs[i][0].scatter(l['center'][0]*4., l['center'][1]*4., color='green', s=50)

    plt.show()