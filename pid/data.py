#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset, DataLoader

import data_utils.transformer
import data_utils.sampler as sampler
import data_utils.loader

import config.const as const_util
import utils


# 因子分解数据进程
class FactorizationDataProcessor(object):

    def __init__(self, flags_obj):
        self.name = flags_obj.name + '_fdp'

    @staticmethod
    def get_blend_pair_dataloader(flags_obj, dm):
        #  混合因子分解
        dataset = BlendPairFactorizationDataset(flags_obj, dm)
        return DataLoader(dataset, batch_size=flags_obj.batch_size, shuffle=flags_obj.shuffle, num_workers=flags_obj.num_workers, drop_last=True)

    @staticmethod
    def get_blend_point_dataloader(flags_obj, dm):
        #  混合因子分解
        dataset = BlendPointFactorizationDataset(flags_obj, dm)
        return DataLoader(dataset, batch_size=flags_obj.batch_size, shuffle=flags_obj.shuffle, num_workers=flags_obj.num_workers, drop_last=True)

    @staticmethod
    def get_ips_blend_pair_dataloader(flags_obj, dm):
        #  IPS
        dataset = IPSBlendPairFactorizationDataset(flags_obj, dm)
        return DataLoader(dataset, batch_size=flags_obj.batch_size, shuffle=flags_obj.shuffle, num_workers=flags_obj.num_workers, drop_last=True)


# parent class
class FactorizationDataset(Dataset):

    def __init__(self, flags_obj, dm):

        self.name = flags_obj.name + '_dataset'
        self.make_sampler(flags_obj, dm)

    def make_sampler(self, flags_obj, dm):

        train_coo_record = dm.coo_record

        transformer = data_utils.transformer.SparseTransformer(flags_obj)
        train_lil_record = transformer.coo2lil(train_coo_record)
        train_dok_record = transformer.coo2dok(train_coo_record)
        self.make_sampler_core(flags_obj, train_lil_record, train_dok_record)

    def __len__(self):

        return len(self.sampler.record)

    def __getitem__(self, index):

        raise NotImplementedError


class BlendPairFactorizationDataset(FactorizationDataset):

    def __init__(self, flags_obj, dm):

        super(BlendPairFactorizationDataset, self).__init__(flags_obj, dm)

    def make_sampler(self, flags_obj, dm):

        transformer = data_utils.transformer.SparseTransformer(flags_obj)

        train_coo_record = dm.coo_record
        train_lil_record = transformer.coo2lil(train_coo_record)
        train_dok_record = transformer.coo2dok(train_coo_record)

        # pair
        self.sampler = sampler.PairSampler(flags_obj, train_lil_record, train_dok_record, flags_obj.neg_sample_rate)

        # train_skew_coo_record = dm.skew_coo_record
        # train_skew_lil_record = transformer.coo2lil(train_skew_coo_record)
        # train_skew_dok_record = transformer.coo2dok(train_skew_coo_record)
        #
        # self.skew_sampler = sampler.PairSampler(flags_obj, train_skew_lil_record, train_skew_dok_record, flags_obj.neg_sample_rate)

    def __len__(self):

        return len(self.sampler.record)
        # return len(self.sampler.record) + len(self.skew_sampler.record)

    def __getitem__(self, index):

        # if index < len(self.sampler.record):
        users, items_pos, items_neg = self.sampler.sample(index)
        # else:
        #     users, items_pos, items_neg = self.skew_sampler.sample(index - len(self.sampler.record))

        return users, items_pos, items_neg

    def adapt(self, epoch, decay):

        pass


class BlendPointFactorizationDataset(FactorizationDataset):

    def __init__(self, flags_obj, dm):

        super(BlendPointFactorizationDataset, self).__init__(flags_obj, dm)

    def make_sampler(self, flags_obj, dm):

        transformer = data_utils.transformer.SparseTransformer(flags_obj)

        train_coo_record = dm.coo_record
        train_lil_record = transformer.coo2lil(train_coo_record)
        train_dok_record = transformer.coo2dok(train_coo_record)

        # point
        self.sampler = sampler.PointSampler(flags_obj, train_lil_record, train_dok_record, flags_obj.neg_sample_rate)

        # train_skew_coo_record = dm.skew_coo_record
        # train_skew_lil_record = transformer.coo2lil(train_skew_coo_record)
        # train_skew_dok_record = transformer.coo2dok(train_skew_coo_record)
        #
        # self.skew_sampler = sampler.PairSampler(flags_obj, train_skew_lil_record, train_skew_dok_record, flags_obj.neg_sample_rate)

    def __len__(self):

        return len(self.sampler.record)
        # return len(self.sampler.record) + len(self.skew_sampler.record)

    def __getitem__(self, index):

        # if index < len(self.sampler.record):
        users, items_pos, items_neg = self.sampler.sample(index)
        # else:
        #     users, items_pos, items_neg = self.skew_sampler.sample(index - len(self.sampler.record))

        return users, items_pos, items_neg

    def adapt(self, epoch, decay):

        pass


class IPSBlendPairFactorizationDataset(FactorizationDataset):

    def __init__(self, flags_obj, dm):

        super(IPSBlendPairFactorizationDataset, self).__init__(flags_obj, dm)
        self.get_weight(flags_obj, dm)

    def get_weight(self, flags_obj, dm):

        pop = self.get_popularity(dm, flags_obj)
        pop = np.clip(pop, 1, pop.max() + 1)
        pop = pop/np.linalg.norm(pop, ord=np.inf)
        pop = 1/pop

        if 'c' in flags_obj.weighting_mode:
            pop = np.clip(pop, 1, np.median(pop))
        if 'n' in flags_obj.weighting_mode:
            pop = pop/np.linalg.norm(pop, ord=np.inf)

        if flags_obj.weighting_smoothness != 1.0:
            pop = pop**flags_obj.weighting_smoothness
            pop = pop/np.linalg.norm(pop, ord=np.inf)

        self.weight = pop.astype(np.float32)

    def get_popularity(self, dm, flags_obj):

        return dm.get_popularity(flags_obj)  # dm.get_blend_popularity()

    def make_sampler(self, flags_obj, dm):

        transformer = data_utils.transformer.SparseTransformer(flags_obj)

        train_coo_record = dm.coo_record
        train_lil_record = transformer.coo2lil(train_coo_record)
        train_dok_record = transformer.coo2dok(train_coo_record)

        self.sampler = sampler.PairSampler(flags_obj, train_lil_record, train_dok_record, flags_obj.neg_sample_rate)

        # train_skew_coo_record = dm.skew_coo_record
        # train_skew_lil_record = transformer.coo2lil(train_skew_coo_record)
        # train_skew_dok_record = transformer.coo2dok(train_skew_coo_record)
        #
        # self.skew_sampler = sampler.PairSampler(flags_obj, train_skew_lil_record, train_skew_dok_record, flags_obj.neg_sample_rate)

    def __len__(self):

        return len(self.sampler.record) #+ len(self.skew_sampler.record)

    def __getitem__(self, index):

        # if index < len(self.sampler.record):
        users, items_pos, items_neg = self.sampler.sample(index)
        # else:
        #     users, items_pos, items_neg = self.skew_sampler.sample(index - len(self.sampler.record))

        weight = self.weight[items_pos-1]

        return users, items_pos, items_neg, weight


class CGDataProcessor(object):

    def __init__(self, flags_obj):

        self.name = flags_obj.name + '_cdp'

    @staticmethod
    def get_dataloader(flags_obj, test_data_source):

        dataset = CGDataset(flags_obj, test_data_source)

        return DataLoader(dataset, batch_size=flags_obj.batch_size, shuffle=True, num_workers=flags_obj.num_workers, drop_last=False), dataset.max_train_interaction


class CGDataset(Dataset):

    def __init__(self, flags_obj, test_data_source):

        self.name = flags_obj.name + '_dataset'
        self.test_data_source = test_data_source
        self.sort_users(flags_obj)

    def sort_users(self, flags_obj):

        loader = data_utils.loader.CooLoader(flags_obj)
        if self.test_data_source == 'val':
            coo_record = loader.load(const_util.val_coo_record)
        elif self.test_data_source == 'test':
            coo_record = loader.load(const_util.test_coo_record)
        transformer = data_utils.transformer.SparseTransformer(flags_obj)
        self.lil_record = transformer.coo2lil(coo_record)

        train_coo_record = loader.load(const_util.train_coo_record)
        # train_skew_coo_record = loader.load(const_util.train_skew_coo_record)

        # blend_user = np.hstack((train_coo_record.row, train_skew_coo_record.row))
        # blend_item = np.hstack((train_coo_record.col, train_skew_coo_record.col))
        # blend_value = np.hstack((train_coo_record.data, train_skew_coo_record.data))
        # blend_coo_record = sp.coo_matrix((blend_value, (blend_user, blend_item)), shape=train_coo_record.shape)

        self.train_lil_record = transformer.coo2lil(train_coo_record)  # transformer.coo2lil(blend_coo_record)

        train_interaction_count = np.array([len(row) for row in self.train_lil_record.rows], dtype=np.int64)
        self.max_train_interaction = int(max(train_interaction_count))

        test_interaction_count = np.array([len(row) for row in self.lil_record.rows], dtype=np.int64)
        self.max_test_interaction = int(max(test_interaction_count))

    def __len__(self):

        return len(self.lil_record.rows)

    def __getitem__(self, index):

        unify_train_pos = np.full(self.max_train_interaction, -1, dtype=np.int64)
        unify_test_pos = np.full(self.max_test_interaction, -1, dtype=np.int64)

        train_pos = self.train_lil_record.rows[index]
        test_pos = self.lil_record.rows[index]

        unify_train_pos[:len(train_pos)] = train_pos
        unify_test_pos[:len(test_pos)] = test_pos

        return torch.LongTensor([index]), torch.LongTensor(unify_train_pos), torch.LongTensor(unify_test_pos), torch.LongTensor([len(test_pos)])

