#!/usr/local/anaconda3/envs/torch-1.1-py3/bin/python
#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

ml100k = '/data2/gpxu/popular_debias/pid/data/ml100k/'
music = '/data2/gpxu/popular_debias/pid/data/music/'
epinion = '/data2/gpxu/popular_debias/pid/data/epinion/'

coo_record = 'coo_record.npz'
train_coo_record = 'train_coo_record.npz'
val_coo_record = 'val_coo_record.npz'
test_coo_record = 'test_coo_record.npz'

train_skew_coo_record = 'val_1_coo_record.npz' # 'train_skew_coo_record.npz'

popularity = 'popularity_train.npy'
blend_popularity = 'popularity_all.npy' # 'popularity_blend.npy'

ckpt = 'ckpt/'
