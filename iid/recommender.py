#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

import torch
import torch.optim as optim

import data
import model
import utils
import candidate_generator as cg
import config.const as const_util
import data_utils.loader as loader
import os
import dgl
import numpy as np


class Recommender(object):

    def __init__(self, flags_obj, workspace, dm):

        self.dm = dm
        self.model_name = flags_obj.model
        self.flags_obj = flags_obj
        self.set_device()
        self.set_model()
        self.workspace = workspace

    def set_device(self):

        self.device = utils.ContextManager.set_device(self.flags_obj)

    def set_model(self):

        raise NotImplementedError

    def transfer_model(self):

        self.model = self.model.to(self.device)

    def save_ckpt(self, epoch):

        ckpt_path = os.path.join(self.workspace, const_util.ckpt)
        if not os.path.exists(ckpt_path):
            os.mkdir(ckpt_path)

        model_path = os.path.join(ckpt_path, 'epoch_' + str(epoch) + '.pth')
        torch.save(self.model.state_dict(), model_path)

    def load_ckpt(self, epoch):

        ckpt_path = os.path.join(self.workspace, const_util.ckpt)
        model_path = os.path.join(ckpt_path, 'epoch_' + str(epoch) + '.pth')
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    def get_dataloader(self):

        raise NotImplementedError

    def get_pair_dataloader(self):

        raise NotImplementedError

    def get_point_dataloader(self):

        raise NotImplementedError

    def get_optimizer(self):

        return optim.Adam(self.model.parameters(), lr=self.flags_obj.lr, weight_decay=self.flags_obj.weight_decay, betas=(0.5, 0.99), amsgrad=True)

    def inference(self, sample):

        raise NotImplementedError

    def make_cg(self):

        raise NotImplementedError

    def cg(self, users, topk):

        raise NotImplementedError


class MFRecommender(Recommender):

    def __init__(self, flags_obj, workspace, dm):

        super(MFRecommender, self).__init__(flags_obj, workspace, dm)

    def set_model(self):

        self.model = model.MF(self.dm.n_user, self.dm.n_item, self.flags_obj.embedding_size)

    def get_pair_dataloader(self):

        return data.FactorizationDataProcessor.get_blend_pair_dataloader(self.flags_obj, self.dm)

    def get_point_dataloader(self):

        return data.FactorizationDataProcessor.get_blend_point_dataloader(self.flags_obj, self.dm)

    def pair_inference(self, sample):

        user, item_p, item_n = sample
        p_score, n_score = self.model.pair_forward(user, item_p, item_n)
        return p_score, n_score

    def point_inference(self, sample):
        user, item, label = sample
        score = self.model.point_forward(user, item)
        label = label.to(self.device)
        score.to(self.device)
        return score, label

    def make_cg(self):

        self.item_embeddings = self.model.get_item_embeddings()
        self.generator = cg.FaissInnerProductMaximumSearchGenerator(self.flags_obj, self.item_embeddings)
        self.user_embeddings = self.model.get_user_embeddings()

    def cg(self, users, topk):
        # 返回与embedding内积最大的item_id
        return self.generator.generate(self.user_embeddings[users], topk)


class IIDRecommender(MFRecommender):

    def __init__(self, flags_obj, workspace, dm):

        super(IIDRecommender, self).__init__(flags_obj, workspace, dm)

    def get_ips_pair_dataloader(self):

        return data.FactorizationDataProcessor.get_ips_blend_pair_dataloader(self.flags_obj, self.dm)

    def pair_inference(self, sample):

        user, item_p, item_n, weight = sample
        sample_wrap = (user, item_p, item_n)
        p_score, n_score = super(IIDRecommender, self).pair_inference(sample_wrap)
        weight = weight.to(self.device)
        return p_score, n_score, weight
    
    def set_model(self):

        self.model = model.IIDMF(self.dm.n_user, self.dm.n_item, self.flags_obj.embedding_size)
