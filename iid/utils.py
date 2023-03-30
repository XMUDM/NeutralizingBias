#coding=utf-8

import os
import datetime
import setproctitle

from absl import logging
from visdom import Visdom

import numpy as np

import numbers

import torch

import config.const as const_util
import trainer
import recommender
import data_utils.sampler as sampler
import data_utils.transformer as transformer
import data_utils.loader as loader

# 推荐模型
class ContextManager(object):

    def __init__(self, flags_obj):

        self.name = flags_obj.name + '_cm'
        self.exp_name = flags_obj.name
        self.output = flags_obj.output
        self.set_load_path(flags_obj)

    @staticmethod
    def set_load_path(flags_obj):

        if flags_obj.dataset == 'ml100k':
            flags_obj.load_path = const_util.ml100k
        elif flags_obj.dataset == 'music':
            flags_obj.load_path = const_util.music
        elif flags_obj.dataset == 'epinion':
            flags_obj.load_path = const_util.epinion

    def set_default_ui(self):

        self.set_workspace()
        self.set_process_name()
        self.set_logging()

    def set_workspace(self):

        date_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        # date_time = '2023-03-14-10-32-30' # '2022' 
        dir_name = self.exp_name + date_time
        if not os.path.exists(self.output):
            os.mkdir(self.output)
        self.workspace = os.path.join(self.output, dir_name)
        if not os.path.exists(self.workspace):
            os.mkdir(self.workspace)


    def set_process_name(self):
        setproctitle.setproctitle(self.exp_name + '@guip')

    def set_logging(self):

        self.log_path = os.path.join(self.workspace, 'log')
        if not os.path.exists(self.log_path):
            os.mkdir(self.log_path)
        logging.flush()
        logging.get_absl_handler().use_absl_log_file(self.exp_name + '.log', self.log_path)

    def set_test_logging(self):

        self.log_path = os.path.join(self.workspace, 'test_log')
        if not os.path.exists(self.log_path):
            os.mkdir(self.log_path)
        logging.flush()
        logging.get_absl_handler().use_absl_log_file(self.exp_name + '.log', self.log_path)

    def logging_flags(self, flags_obj):

        logging.info('FLAGS:')
        for flag, value in flags_obj.flag_values_dict().items():
            logging.info('{}: {}'.format(flag, value))

    @staticmethod
    def set_trainer(flags_obj, cm,  dm):

        if 'IPS' in flags_obj.model:
            return trainer.IPSPairTrainer(flags_obj, cm,  dm)
        elif 'IID' in flags_obj.model:
            return trainer.IIDPairTrainer(flags_obj, cm,  dm)
        else:
            return trainer.PairTrainer(flags_obj, cm,  dm)

    @staticmethod
    def set_recommender(flags_obj, workspace, dm):

        if flags_obj.model == 'MF':
            return recommender.MFRecommender(flags_obj, workspace, dm)
        elif flags_obj.model == 'IIDMF':
            return recommender.IIDRecommender(flags_obj, workspace, dm)
        elif flags_obj.model == 'IPS':
            return recommender.IPSRecommender(flags_obj, workspace, dm)

    @staticmethod
    def set_device(flags_obj):

        if not flags_obj.use_gpu:
            return torch.device('cpu')
        else:
            return torch.device('cuda:{}'.format(flags_obj.gpu_id))


# 数据集
class DatasetManager(object):

    def __init__(self, flags_obj):

        self.name = flags_obj.name + '_dm'
        self.make_coo_loader_transformer(flags_obj)
        self.make_npy_loader(flags_obj)
        self.make_csv_loader(flags_obj)

    def make_coo_loader_transformer(self, flags_obj):

        self.coo_loader = loader.CooLoader(flags_obj)
        self.coo_transformer = transformer.SparseTransformer(flags_obj)

    def make_npy_loader(self, flags_obj):

        self.npy_loader = loader.NpyLoader(flags_obj)

    def make_csv_loader(self, flags_obj):

        self.csv_loader = loader.CsvLoader(flags_obj)

    def get_dataset_info(self, flags_obj):

        coo_record = self.coo_loader.load(flags_obj.load_path + const_util.train_coo_record)

        self.n_user = coo_record.shape[0]
        self.n_item = coo_record.shape[1]
        # print(self.n_user)
        # print(self.n_item)
        self.coo_record = coo_record

    def get_skew_dataset(self, flags_obj):
        # 倾斜数据集
        self.skew_coo_record = self.coo_loader.load(flags_obj.load_path + const_util.train_skew_coo_record)

    def get_popularity(self, flags_obj):
        # 流行度数据
        self.popularity = self.npy_loader.load(flags_obj.load_path + const_util.popularity)
        return self.popularity

    def get_blend_popularity(self, flags_obj):
        # 混合流行度
        self.blend_popularity = self.npy_loader.load(flags_obj.load_path + const_util.blend_popularity)
        return self.blend_popularity


class EarlyStopManager(object):

    def __init__(self, flags_obj):

        self.name = flags_obj.name + '_esm'
        self.min_lr = flags_obj.min_lr
        self.es_patience = flags_obj.es_patience
        self.count = 0
        self.max_metric = 0

    def step(self, lr, metric):

        if lr > self.min_lr:
            if metric > self.max_metric:
                self.max_metric = metric
            return False
        else:
            if metric > self.max_metric:
                self.max_metric = metric
                self.count = 0
                return False
            else:
                self.count = self.count + 1
                if self.count > self.es_patience:
                    return True
                return False

