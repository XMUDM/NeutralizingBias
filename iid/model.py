#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

import math
import numpy as np

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.autograd import Variable
import dgl.function as fn

import utils

import json

# from deprecated import deprecated
from tqdm import tqdm

import random

class PearsonCorrelation(nn.Module):
    def forward(self,tensor_1,tensor_2):
        x = tensor_1
        y = tensor_2

        vx = x - torch.mean(x)
        vy = y - torch.mean(y)

        cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        return cost

class MF(nn.Module):

    def __init__(self, num_users, num_items, embedding_size):

        super(MF, self).__init__()

        self.users = Parameter(torch.FloatTensor(num_users, embedding_size))
        self.items = Parameter(torch.FloatTensor(num_items, embedding_size))
        self.item_pop = torch.FloatTensor(num_items)
        self.num_items = num_items
        self.init_params()

    def init_params(self):

        stdv = 1. / math.sqrt(self.users.size(1))
        self.users.data.uniform_(-stdv, stdv)
        self.items.data.uniform_(-stdv, stdv)

        self.item_pop_dict = np.load("/data2/gpxu/popular_debias/iid/data/ml100k/item_train_pop_dic.npy",
                                     allow_pickle=True).item()

        self.item_pop_max = max(self.item_pop_dict.values())
        for ii in range(0, self.num_items):
            if ii not in self.item_pop_dict.keys():
                self.item_pop_dict[ii] = 0
            self.item_pop[ii] = (self.item_pop_dict[ii] / self.item_pop_max) # self.item_pop_dict[ii] / self.item_pop_max # (self.item_pop_dict[ii] / self.item_pop_max) *2 - 1

    def pair_forward(self, user, item_p, item_n):

        user = self.users[user]
        item_p = self.items[item_p].cuda()
        item_n = self.items[item_n].cuda()

        p_score = torch.sum(user * item_p, 2)
        n_score = torch.sum(user * item_n, 2)

        return p_score, n_score

    def point_forward(self, user, item):

        user = self.users[user]
        item = self.items[item]
        score = torch.sum(user * item, 2)

        return score

    def get_item_embeddings(self):
        # a = self.items.detach().cpu().numpy().astype('float32')
        # b = dict()
        # for i in range(self.num_items):
        #     b[i] = a[i].ravel().tolist()
        # json_item = json.dumps(b)
        # with open('/data2/gpxu/popular_debias/iid/MF_item_ml100k.json', 'w') as json_file:
        #     json_file.write(json_item)
        return self.items.detach().cpu().numpy().astype('float32')

    def get_user_embeddings(self):

        return self.users.detach().cpu().numpy().astype('float32')

class IIDMF(nn.Module):

    def __init__(self, num_users, num_items, embedding_size):

        super(IIDMF, self).__init__()
        # id part
        self.num_items = num_items
        self.users = Parameter(torch.FloatTensor(num_users, embedding_size))
        self.items = Parameter(torch.FloatTensor(num_items, embedding_size))  # _id

        self.init_params()
        self.lr1 = nn.Linear(embedding_size-1, 1)  # 63 -> 1

    def init_params(self):
        stdv = 1. / math.sqrt(self.users.size(1))
        self.users.data.uniform_(-stdv, stdv)
        self.items.data.uniform_(-stdv, stdv)

        # popular part 
        self.item_pop_dict = np.load("/data2/gpxu/popular_debias/iid/data/ml100k/item_train_pop_dic.npy",
                                     allow_pickle=True).item()

        self.item_pop_max = max(self.item_pop_dict.values())

        item_pop = np.zeros((self.num_items, 1))
        for ii in range(0, self.num_items):
            if ii not in self.item_pop_dict.keys():
                self.item_pop_dict[ii] = 0
            item_pop[ii] = self.item_pop_dict[ii] / self.item_pop_max

        self.item_pop = Parameter(torch.from_numpy(item_pop).float()).requires_grad_(requires_grad=False)

    def pair_forward(self, user, item_p, item_n):
        user = self.users[user]

        item_p = self.items[item_p]
        item_n = self.items[item_n]

        p_score = torch.sum(user * item_p, 2)
        n_score = torch.sum(user * item_n, 2)

        return p_score, n_score

    def point_forward(self, user, item):

        user = self.users[user]
        item = self.items[item]

        score = torch.sum(user * item, 2)

        return score

    def get_item_embeddings(self):

        return self.items.detach().cpu().numpy().astype('float32')

    def get_user_embeddings(self):

        return self.users.detach().cpu().numpy().astype('float32')
