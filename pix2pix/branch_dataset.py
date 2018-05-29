import os

import numpy
from PIL import Image
import six

import numpy as np

from io import BytesIO
import os
import pickle
import json
import numpy as np

import skimage.io as io

from chainer.dataset import dataset_mixin

# download `BASE` dataset from http://cmp.felk.cvut.cz/~tylecr1/facade/
class BranchDataset(dataset_mixin.DatasetMixin):
    def __init__(self, dataDir='./facade/base', data_range=(1,300)):
        print("load dataset start")
        print("    from: %s"%dataDir)
        print("    range: [%d, %d)"%(data_range[0], data_range[1]))
        self.dataDir = dataDir
        self.dataset = []
        for i in range(data_range[0],data_range[1]):
            print("file_name", dataDir+"/Input/%04d.jpg"%i)
            img = Image.open(dataDir + "/Input/%04d.jpg"%i)
            label = Image.open(dataDir+ "Label/%04d.jpg"%i)
            w,h = img.size
            height = 129
            # width = int(w*(float(height)/float(h)))
            width = 129

            img = img.resize((width, height), Image.NEAREST)
            label = label.resize((width, height), Image.NEAREST)
            #debug
            img = np.asarray(img).astype("f")
            img = img/128.0 - 1.0

            print("img.shape = ", img.shape, np.max(img), np.min(img))

            label_ = np.asarray(label)-1  # [0, 12)
            # print type(label)
            label = np.zeros((256, img.shape[0], img.shape[1])).astype("i")
            # print(img.shape)
            for j in range(256):
                label[j,:] = label_==j

            self.dataset.append((img,label))
            # print self.dataset[-1][0].shape, self.dataset[-1][1].shape

        print("load dataset done")

    def __len__(self):
        return len(self.dataset)

    # return (label, img)
    def get_example(self, i, crop_width=128):
        # print("get_example", self.dataset[i][0].shape)
        h,w = self.dataset[i][0].shape
        x_l = np.random.randint(0,w-crop_width)
        x_r = x_l+crop_width
        y_l = np.random.randint(0,h-crop_width)
        y_r = y_l+crop_width
        # print("np.max(self.dataset[i][1][:,y_l:y_r,x_l:x_r]) = ", np.max(self.dataset[i][1][:,y_l:y_r,x_l:x_r]))
        # print("np.max(self.dataset[i][0][:,y_l:y_r,x_l:x_r]) = ", np.max(self.dataset[i][0][:,y_l:y_r,x_l:x_r]))
        # print("np.min(self.dataset[i][1][:,y_l:y_r,x_l:x_r]) = ", np.min(self.dataset[i][1][:,y_l:y_r,x_l:x_r]))
        # print("np.min(self.dataset[i][0][:,y_l:y_r,x_l:x_r]) = ", np.min(self.dataset[i][0][:,y_l:y_r,x_l:x_r]))
        # print("Input", self.dataset[i][0].shape)
        # print("Output", self.dataset[i][1].shape)
        #return self.dataset[i][1][:, y_l:y_r, x_l:x_r], self.dataset[i][0][:, y_l:y_r,x_l:x_r]
        # return self.dataset[i][1][:, y_l:y_r, x_l:x_r], self.dataset[i][0][y_l:y_r,x_l:x_r]
        # X , y
        return self.dataset[i][1][:, y_l:y_r, x_l:x_r], self.dataset[i][0][y_l:y_r,x_l:x_r][None, :, :]
