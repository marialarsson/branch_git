#!/usr/bin/env python

# python train_facade.py -g 0 -i ./facade/base --out result_facade --snapshot_interval 10000

from __future__ import print_function
import argparse
import os

import chainer
from chainer import training
from chainer.training import extensions
from chainer import serializers

from net import Discriminator
from net import Encoder
from net import Decoder
from updater import FacadeUpdater


from branch_dataset import BranchDataset
from branch_visualizer import out_image

import numpy as np
from chainer import Variable
from PIL import Image


# def main():
parser = argparse.ArgumentParser(description='chainer implementation of pix2pix')
parser.add_argument('--batchsize', '-b', type=int, default=100,
                    help='Number of images in each mini-batch')
parser.add_argument('--epoch', '-e', type=int, default=200,
                    help='Number of sweeps over the dataset to train')
parser.add_argument('--gpu', '-g', type=int, default=-1,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--dataset', '-i', default="../Dataset/",
                    help='Directory of image files.')
parser.add_argument('--out', '-o', default='results',
                    help='Directory to output the results')
parser.add_argument('--resume', '-r', default='',
                    help='Resume the training from snapshot')
parser.add_argument('--seed', type=int, default=0,
                    help='Random seed')
parser.add_argument('--snapshot_interval', type=int, default=10000,
                    help='Interval of snapshot')
parser.add_argument('--display_interval', type=int, default=1,
                    help='Interval of displaying log to console')
args = parser.parse_args()

# Set up a neural network to train
enc = Encoder(in_ch=256)
dec = Decoder(out_ch=1)
dis = Discriminator(in_ch=256, out_ch=1)

if args.gpu >= 0:
    chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
    enc.to_gpu()  # Copy the model to the GPU
    dec.to_gpu()
    dis.to_gpu()

# Setup an optimizer
def make_optimizer(model, alpha=0.0002, beta1=0.5):
    optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.00001), 'hook_dec')
    return optimizer
opt_enc = make_optimizer(enc)
opt_dec = make_optimizer(dec)
opt_dis = make_optimizer(dis)

train_d = BranchDataset(args.dataset, data_range=(1,2))
test_d = BranchDataset(args.dataset, data_range=(66,67))

train_iter = chainer.iterators.SerialIterator(train_d, args.batchsize)
test_iter = chainer.iterators.SerialIterator(test_d, args.batchsize)

# Set up a trainer
updater = FacadeUpdater(
    models=(enc, dec, dis),
    iterator={
        'main': train_iter,
        'test': test_iter},
    optimizer={
        'enc': opt_enc,
        'dec': opt_dec,
        'dis': opt_dis},
    device=args.gpu)

trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

print('load models')
chainer.serializers.load_npz('/home/hyoshida/git/branch_git/results/126922/enc_iter_4000.npz', enc)
chainer.serializers.load_npz('/home/hyoshida/git/branch_git/results/126922/dec_iter_4000.npz', dec)
chainer.serializers.load_npz('/home/hyoshida/git/branch_git/results/126922/dis_iter_4000.npz', dis)


print('save image')

batch = updater.get_iterator('test').next()
# print("get iterator" , batch[0][0].shape)

test_data  = test_d.get_example(0)[0]
test_in = np.zeros((1, 256, 128, 128)).astype("f")
test_in[0, :] = np.asarray(batch[0][0])
# print("get_example" , test_in.shape, test_in)

test_in = Variable(test_in)
# print("after variable" , test_in.shape, test_in)

z = enc(test_in)
out = dec(z)
out = out.data[0, :]*128 + 128
out = np.asarray(out, dtype=np.uint8)
out = out.transpose(1,2,0)
out = out.reshape((128,128))
# print(out.shape)
Image.fromarray(out).show()

# test_in[:, 0, :, :] = 0
# for i in range(12):
#     test_in[:, 0, :, :] += np.uint8(15*i*in_all[:,i,:,:])
#
# # data_in = x_in.transpose(0,2,3,1)
# Image.fromarray(test_in.transpose(0,2,3,1)[0], 'HSV').convert('RGB').show()
