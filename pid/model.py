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

        self.item_pop_dict = np.load("/data2/gpxu/popular_debias/pid/data/ml100k/item_train_pop_dic.npy",
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
        # with open('/data2/gpxu/popular_debias/pid/MF_item_ml100k.json', 'w') as json_file:
        #     json_file.write(json_item)
        return self.items.detach().cpu().numpy().astype('float32')

    def get_user_embeddings(self):

        return self.users.detach().cpu().numpy().astype('float32')

class PIDMF(nn.Module):

    def __init__(self, num_users, num_items, embedding_size):

        super(PIDMF, self).__init__()
        # id part
        self.num_items = num_items
        self.users = Parameter(torch.FloatTensor(num_users, embedding_size))
        self.items_id = Parameter(torch.FloatTensor(num_items, embedding_size-1))  # _id

        self.init_params()
        self.lr1 = nn.Linear(embedding_size-1, 1)  # 63 -> 1

    def init_params(self):
        stdv = 1. / math.sqrt(self.users.size(1))
        self.users.data.uniform_(-stdv, stdv)
        self.items_id.data.uniform_(-stdv, stdv)

        # popular part 
        self.item_pop_dict = np.load("/data2/gpxu/popular_debias/pid/data/ml100k/item_train_pop_dic.npy",
                                     allow_pickle=True).item()

        self.item_pop_max = max(self.item_pop_dict.values())
        # self.item_pop_min = min(self.item_pop_dict.values())

        item_pop = np.zeros((self.num_items, 1))
        for ii in range(0, self.num_items):
            if ii not in self.item_pop_dict.keys():
                self.item_pop_dict[ii] = 0
            item_pop[ii] = self.item_pop_dict[ii] / self.item_pop_max

        # print(min(item_pop[ii]))

        self.item_pop = Parameter(torch.from_numpy(item_pop).float()).requires_grad_(requires_grad=False)
        self.item_pop_true = Parameter(torch.from_numpy(item_pop).float()).requires_grad_(requires_grad=False)
        # self.item_pop_grad = torch.FloatTensor(self.num_items, 1)

    def pair_forward(self, user, item_p, item_n):
        user = self.users[user]

        # 先将63 -> 1，只更新lr1
        for p in self.lr1.parameters():
            p.requires_grad = True

        optimizer1 = torch.optim.Adam(self.lr1.parameters(), lr=1e-4, weight_decay=1e-5)
        optimizer1.zero_grad()

        pop_loss = 0
        item_p_emb = self.items_id[item_p]
        item_n_emb = self.items_id[item_n]
        pop_p_predict = self.lr1(item_p_emb).to(torch.device('cuda:0'))
        pop_n_predict = self.lr1(item_n_emb).to(torch.device('cuda:0'))

        # 这里的true 应该保持不变的，是真实的结果
        pop_p_true = self.item_pop_true[item_p].to(torch.device('cuda:0'))
        pop_n_true = self.item_pop_true[item_n].to(torch.device('cuda:0'))

        loss_func = nn.MSELoss()

        pop_loss += loss_func(pop_p_predict, pop_p_true)
        pop_loss += loss_func(pop_n_predict, pop_n_true)
        pop_loss.backward()

        optimizer1.step()

        self.items = torch.cat([self.items_id, self.item_pop.to(torch.device('cuda:0'))], dim=1).to(torch.device('cuda:0'))
        self.items.retain_grad()

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

        # # 在 test 的时候，讲流行度方向置0
        # item_pop = np.zeros((self.num_items, 1))
        # self.item_pop = Parameter(torch.from_numpy(item_pop).float()).requires_grad_(requires_grad=False)
        # self.items = torch.cat([self.items_id, self.item_pop.cuda()], dim=1).to(torch.device('cuda:0'))

        # train 
        self.items = torch.cat([self.items_id, self.item_pop_true.cuda()], dim=1).to(torch.device('cuda:0'))

        return self.items.detach().cpu().numpy().astype('float32')

    def get_user_embeddings(self):

        return self.users.detach().cpu().numpy().astype('float32')
