#!/usr/bin/env python

import os

import numpy as np
from PIL import Image

import chainer
import chainer.cuda
from chainer import Variable

def out_image(updater, enc, dec, rows, cols, seed, dst):
    @chainer.training.make_extension()
    def make_image(trainer):
        np.random.seed(seed)
        n_images = rows * cols
        xp = enc.xp

        w_in = 128
        w_out = 128
        in_ch = 256
        out_ch = 1

        in_all = np.zeros((n_images, in_ch, w_in, w_out)).astype("i")
        gt_all = np.zeros((n_images, out_ch, w_in, w_out)).astype("f")
        gen_all = np.zeros((n_images, out_ch, w_in, w_out)).astype("f")

        for it in range(n_images):
            batch = updater.get_iterator('test').next()
            batchsize = len(batch)

            x_in = xp.zeros((batchsize, in_ch, w_in, w_out)).astype("f")
            t_out = xp.zeros((batchsize, out_ch, w_in, w_out)).astype("f")

            # print("batch 0 0 shape", batch[0][0].shape)
            # print("batch 0 1 shape", batch[0][1].shape)

            for i in range(batchsize):
                x_in[i,:] = xp.asarray(batch[i][0])
                t_out[i,:] = xp.asarray(batch[i][1])
            x_in = Variable(x_in)

            z = enc(x_in)
            x_out = dec(z)

            # print("x_in.data shape",  x_in.data[0,:].shape)
            # print("x_out.data shape",  x_out.data[0,:].shape)

            in_all[it,:] = x_in.data[0,:]
            gt_all[it,:] = t_out[0,:]
            gen_all[it,:] = x_out.data[0,:]


        def save_image(x, name, mode=None):
            _, C, H, W = x.shape
            x = x.reshape((rows, cols, C, H, W))
            x = x.transpose(0, 3, 1, 4, 2)
            if C==1:
                x = x.reshape((rows*H, cols*W))
            else:
                x = x.reshape((rows*H, cols*W, C))

            preview_dir = '{}/preview'.format(dst)
            preview_path = preview_dir +\
                '/image_{}_{:0>8}.png'.format(name, trainer.updater.iteration)
            if not os.path.exists(preview_dir):
                os.makedirs(preview_dir)

            Image.fromarray(x, mode=mode).convert('RGB').save(preview_path)

        x = np.asarray(np.clip(gen_all * 128 + 128, 0.0, 255.0), dtype=np.uint8)
        print("save image gen", x.shape)
        save_image(x, "gen")

        # x = np.ones((n_images, 1, 256, 256)).astype(np.uint8)*255
        # x[:,0,:,:] = 0
        # for i in range(128):
        #     x[:,0,:,:] += np.uint8(255*i*in_all[:,i,:,:])
        # save_image(x, "in", mode='HSV')

        x = np.asarray(np.clip(gt_all * 128+128, 0.0, 255.0), dtype=np.uint8)
        print("save image target", x.shape)
        save_image(x, "gt")

    return make_image
