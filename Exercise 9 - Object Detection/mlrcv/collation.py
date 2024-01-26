import torch
import numpy as np
from mlrcv.pre_process import *

class VOCollation:
    def __init__(self, R=4, test=None):
        self.R = R
        self.test = test

    def __call__(self, list_data):
        img_, lbl_ = list(zip(*list_data))

        lbl = []
        img = []

        for i,l in zip(img_,lbl_):
            s_x = 512. / i.size[0]
            s_y = 512. / i.size[1]

            i = i.resize((512,512))
            img.append(np.asarray(i))
            obj = []

            for o in l['annotation']['object']:
                center_x = (int(o['bndbox']['xmin']) + int(o['bndbox']['xmax'])) / 2
                center_y = (int(o['bndbox']['ymin']) + int(o['bndbox']['ymax'])) / 2
                bndbox = {'xmax': int(s_x*int(o['bndbox']['xmax']))//self.R, 'xmin': int(s_x*int(o['bndbox']['xmin']))//self.R,
                            'ymax': int(s_y*int(o['bndbox']['ymax']))//self.R, 'ymin': int(s_y*int(o['bndbox']['ymin']))//self.R}
                obj_ = {'bndbox': bndbox, 'center': (int(s_x*center_x)//self.R, int(s_y*center_y)//self.R)}
                obj.append(obj_)

            lbl.append(obj)

        img = np.asarray(img).astype(np.float)

        if self.test == 'heatmap':
            return self.test_heatmap(img, lbl)
        elif self.test == 'sizemap':
            return self.test_sizemap(img, lbl)

        htm, szm = self.centers_to_maps(img, lbl)
        szm_mask = np.where(szm > 0)

        # normalization
        img = self.normalize_img(img)

        img, htm, szm = torch.from_numpy(img), torch.from_numpy(htm), torch.from_numpy(szm)

        img = img.permute(0, 3, 1, 2).float()
        htm = htm.permute(0, 3, 1, 2).float()
        szm = szm.permute(0, 3, 1, 2).float()

        return img, htm, szm, szm_mask

    def test_heatmap(self, image, label):
        hm = []

        for img, lbl in zip(image, label):
            heatmap = np.zeros((img.shape[0]//self.R,img.shape[1]//self.R))

            for l in lbl:
                heatmap = heatmap_object(img, l, heatmap)

            hm.append(heatmap)

        return image/255., label, np.asarray(hm)[:,:,:,np.newaxis]

    def test_sizemap(self, image, label):
        sz = []

        for img, lbl in zip(image, label):
            sizemap = np.zeros((img.shape[0]//self.R,img.shape[1]//self.R, 3))

            for l in lbl:
                sizemap = sizemap_object(img, l, sizemap)

            sz.append(sizemap)

        sz = np.asarray(sz)
        sz[...,0] /= sz[...,0].max()
        sz[...,1] /= sz[...,1].max()
        

        return image/255., label, sz

    def centers_to_maps(self, image, label):
        hm, sz = [], []

        for img, lbl in zip(image, label):
            heatmap = np.zeros((img.shape[0]//self.R,img.shape[1]//self.R))
            sizemap = np.zeros((img.shape[0]//self.R,img.shape[1]//self.R,2))
            
            for l in lbl:
                heatmap = heatmap_object(img, l, heatmap)
                sizemap = sizemap_object(img, l, sizemap)

            hm.append(heatmap)
            sz.append(sizemap)

        return np.asarray(hm)[:,:,:,np.newaxis], np.asarray(sz)

    def normalize_img(self, img):
        img[...,0] = (img[...,0] - 123.675) / 58.395
        img[...,1] = (img[...,1] - 116.28) / 57.12
        img[...,2] = (img[...,2] - 103.53) / 57.375

        return img

    