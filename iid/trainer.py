#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

from absl import logging

import time
from tqdm import tqdm

import utils
from tester import Tester
from model import PearsonCorrelation

import numpy as np
import torch
import torch.optim as optim


#  parent class
class Trainer(object):

    def __init__(self, flags_obj, cm, dm):

        self.name = flags_obj.name + '_trainer'
        self.cm = cm
        self.dm = dm
        self.flags_obj = flags_obj
        self.lr = flags_obj.lr
        self.set_recommender(flags_obj, cm.workspace, dm)  # create model
        self.recommender.transfer_model()  # to device
        self.tester = Tester(flags_obj, self.recommender)

    def set_recommender(self, flags_obj, workspace, dm):

        self.recommender = utils.ContextManager.set_recommender(flags_obj, workspace, dm)

    def train(self):

        self.set_dataloader()  # train data
        self.tester.set_dataloader('val')  # val data
        self.tester.set_metrics(self.flags_obj.val_metrics)
        self.set_optimizer()
        self.set_scheduler()
        self.set_esm()  # early stop
        self.set_leaderboard()  # vizdom

        for epoch in range(self.flags_obj.epochs):

            self.train_one_epoch(epoch)  # train

            # update hybrid parameter
            watch_metric_value = self.validate(epoch)
            self.recommender.save_ckpt(epoch)
            self.scheduler.step(watch_metric_value)
            self.update_leaderboard(epoch, watch_metric_value)

            stop = self.esm.step(self.lr, watch_metric_value)
            if stop:
                break

            if self.flags_obj.adaptive:
                self.adapt_hyperparameters(epoch)

    def test(self):

        self.cm.set_test_logging()
        self.tester.set_dataloader('test')
        self.tester.set_metrics(self.flags_obj.metrics)

        # self.max_epoch = 49

        if self.flags_obj.test_model == 'best':
            self.recommender.load_ckpt(self.max_epoch)
            logging.info('best epoch: {}'.format(self.max_epoch))

        for topk in self.flags_obj.topk:

            self.tester.max_topk = topk
            results = self.tester.test(self.flags_obj.num_test_users)

            logging.info('TEST results topk = {}:'.format(topk))
            for metric, value in results.items():
                logging.info('{}: {}'.format(metric, value))

        self.tester.set_topk(self.flags_obj)

    def set_dataloader(self):
        # would be rewritten in children class
        raise NotImplementedError

    def set_optimizer(self):

        self.optimizer = self.recommender.get_optimizer()

    def set_scheduler(self):

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', patience=self.flags_obj.patience, min_lr=self.flags_obj.min_lr)

    def set_esm(self):

        self.esm = utils.EarlyStopManager(self.flags_obj)

    def set_leaderboard(self):

        self.max_metric = -1.0
        self.max_epoch = -1

    def update_leaderboard(self, epoch, metric):

        if metric > self.max_metric:

            self.max_metric = metric
            self.max_epoch = epoch

    def adapt_hyperparameters(self, epoch):

        raise NotImplementedError

    def train_one_epoch(self, epoch):

        self.lr = self.train_one_epoch_core(epoch, self.dataloader, self.optimizer, self.lr)
        # return totol_loss

    def train_one_epoch_core(self, epoch, dataloader, optimizer, lr):

        start_time = time.time()
        running_loss = 0.0
        total_loss = 0.0
        num_batch = len(dataloader)
        self.distances = np.zeros(num_batch)

        current_lr = optimizer.param_groups[0]['lr']
        if current_lr < lr:
            lr = current_lr
            logging.info('reducing learning rate!')

        logging.info('learning rate : {}'.format(lr))

        for batch_count, sample in enumerate(tqdm(dataloader)):
            optimizer.zero_grad()

            loss = self.get_loss(sample, batch_count)  # loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total_loss += loss.item()

            if batch_count % (num_batch // 5) == num_batch // 5 - 1:
                logging.info('epoch {}: running loss = {}'.format(epoch, running_loss / (num_batch // 5)))
                running_loss = 0.0

        logging.info('epoch {}: total loss = {}'.format(epoch, total_loss))

        return lr #, total_loss

    def get_loss(self, sample, batch_count):
        # would be rewritten in children class
        raise NotImplementedError

    def validate(self, epoch):

        results = self.tester.test(self.flags_obj.num_val_users)  # test
        logging.info('VALIDATION epoch: {}, results: {}'.format(epoch, results))
        return results[self.flags_obj.watch_metric]

class PairTrainer(Trainer):

    def __init__(self, flags_obj, cm, dm):
        super(PairTrainer, self).__init__(flags_obj, cm, dm)

    def set_dataloader(self):
        self.dataloader = self.recommender.get_pair_dataloader()

    def get_loss(self, sample, batch_count):
        p_score, n_score = self.recommender.pair_inference(sample)
        loss = self.bpr_loss(p_score, n_score)
        return loss

    def bpr_loss(self, p_score, n_score):
        return -torch.mean(torch.log(torch.sigmoid(p_score - n_score)))

class PointTrainer(Trainer):

    def __init__(self, flags_obj, cm, dm):

        super(PointTrainer, self).__init__(flags_obj, cm, dm)
        self.mseloss = torch.nn.MSELoss()

    def set_dataloader(self):

        self.dataloader = self.recommender.get_point_dataloader()

    def get_loss(self, sample, batch_count):

        score, label = self.recommender.point_inference(sample)
        # self.distances[batch_count] = (p_score - n_score).mean().item()
        loss = self.mseloss(score, label)

        return loss

# 改进
class IIDPairTrainer(PairTrainer):

    def __init__(self, flags_obj, cm,  dm):

        super(IIDPairTrainer, self).__init__(flags_obj, cm, dm)

    def set_dataloader(self):

        self.dataloader = self.recommender.get_ips_pair_dataloader()

    def train_one_epoch_core(self, epoch, dataloader, optimizer, lr):

        running_loss = 0.0
        total_loss = 0.0

        num_batch = len(dataloader)
        self.distances = np.zeros(num_batch)

        current_lr = optimizer.param_groups[0]['lr']
        if current_lr < lr:
            lr = current_lr
            logging.info('reducing learning rate!')

        logging.info('learning rate : {}'.format(lr))

        for batch_count, sample in enumerate(tqdm(dataloader)):
            optimizer.zero_grad()

            loss = self.get_loss(sample, batch_count)  # loss

            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            total_loss += loss.item()

            if batch_count % (num_batch // 5) == num_batch // 5 - 1:
                logging.info('epoch {}: running loss = {}'.format(epoch, running_loss / (num_batch // 5)))
                running_loss = 0.0

        logging.info('epoch {}: total loss = {}'.format(epoch, total_loss))

        return lr

    def get_loss(self, sample, batch_count):

        p_score, n_score, weight = self.recommender.pair_inference(sample)

        # print(sample[0].shape)
        user, item_p, item_n, weight = sample

        item_emb = self.recommender.model.items
        # print(item_emb.shape)
        # print(item_emb)
        pop_emb = self.recommender.model.item_pop
        # print(pop_emb.shape)
        
        I = torch.eye(item_emb.shape[1]).to(torch.device('cuda:0'))
        # print(I.shape)
        VV_T = torch.mm(item_emb.T, item_emb).to(torch.device('cuda:0'))
        # print(VV_T.shape)
        
        mseloss = torch.nn.MSELoss()
        
        loss1 = mseloss(VV_T, I)  #  (VV_T - I)
        # print(item_emb[:,0].shape)
        # print(pop_emb.shape)
        
        pearson = PearsonCorrelation()
        # cos
        loss2 = 0
        pop_emb = pop_emb.to(torch.device('cuda:0'))
        for i in range(0, item_emb.shape[1]):
            loss2 += pearson(item_emb[:,i], pop_emb)
        
        # print(loss2)
        loss1 = loss1  # 正
        loss2 = loss2 * 0.0001  # 正 # 0.0001 0.00001 ，降低到同量级，或者低1量级
        loss3 = self.bpr_loss(p_score, n_score, weight)  # self.bpr_loss(p_score, n_score) # 变大
        # print(loss1)
        # print(loss2)
        beta = 0.4
        loss = loss3 + beta * loss1 + (1-beta) * loss2

        return loss

    def bpr_loss(self, p_score, n_score, weight):
        loss = torch.log(torch.sigmoid(p_score - n_score))
        loss = loss*weight.to(torch.device('cuda:0'))
        loss = -loss.mean()
        return loss

