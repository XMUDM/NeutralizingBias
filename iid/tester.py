#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

import data
import metrics
import recommender as rec

import numpy as np
import torch

from tqdm import tqdm


class Tester(object):

    def __init__(self, flags_obj, recommender, test_iou=False):

        self.name = flags_obj.name + '_tester'
        self.recommender = recommender
        self.flags_obj = flags_obj
        self.set_topk(flags_obj)
        self.set_judger(flags_obj, recommender.dm)
        self.test_iou = test_iou
        if test_iou:
            self.pop_recommender = rec.PopularityRecommender(flags_obj, recommender.workspace, recommender.dm)

    def set_metrics(self, metrics):

        self.judger.metrics = metrics

    def set_dataloader(self, test_data_source):

        self.test_data_source = test_data_source
        self.dataloader, self.topk_margin = data.CGDataProcessor.get_dataloader(self.flags_obj, self.test_data_source)
        self.n_user = self.recommender.dm.n_user
        self.cg_topk = self.max_topk + self.topk_margin

    def set_topk(self, flags_obj):

        self.topk = flags_obj.topk
        self.max_topk = max(flags_obj.topk)

    def set_judger(self, flags_obj, dm):

        self.judger = metrics.Judger(flags_obj, dm, self.max_topk)

    def test(self, num_test_users):

        real_num_test_users = 0

        with torch.no_grad():
            self.init_results()
            self.make_cg()
            if self.test_iou:
                self.pop_recommender.make_cg()

            num_test_batches = num_test_users // self.flags_obj.batch_size + 1
            for batch_count, data in enumerate(tqdm(self.dataloader)):
                if batch_count >= num_test_batches:
                    break
                users, train_pos, test_pos, num_test_pos = data
                users = users.squeeze()
                items = self.recommender.cg(users, self.cg_topk)
                # print(score)
                items = self.filter_history(items, train_pos, test_pos)

                batch_results, valid_num_users = self.judger.judge(items, test_pos, num_test_pos)
                real_num_test_users = real_num_test_users + valid_num_users

                if self.test_iou:
                    pop_items = self.pop_recommender.cg(users, self.cg_topk)
                    pop_items = self.filter_history(pop_items, train_pos)
                    iou = self.compute_iou(items, pop_items)
                    batch_results['iou'] = iou

                self.update_results(batch_results)
        self.average_user(real_num_test_users)
        return self.results

    def init_results(self):

        self.results = {k: 0.0 for k in self.judger.metrics}
        if self.test_iou:
            self.results['iou'] = 0.0

    def make_cg(self):

        self.recommender.make_cg()

    def filter_history(self, items, train_pos, test_pos):
        # return np.stack(
        #     [items[i][np.isin(items[i], train_pos[i], invert=True)][:self.max_topk] for i in range(len(items))], axis=0)

        '''
        # 计算 PRU
        u_i_dict = {}
        pop_dict = {}
        p = '/data2/gpxu/popular_debias/iid/result/' + self.flags_obj.name + '2022/'

        for i in range(len(items)):
            u_i_dict[i] = items[i][np.isin(items[i], train_pos[i], invert=True)]  # 删去 train 中的
        np.save(p + 'result_all.npy', u_i_dict)  #

        for i in range(len(items)):
            u_i_dict[i] = u_i_dict[i][np.isin(u_i_dict[i], test_pos[i], invert=False)] # 只留下 test 中的
        np.save(p + 'result_test.npy', u_i_dict)

        b = np.load("/data2/gpxu/popular_debias/iid/data/" + self.flags_obj.dataset +"/popularity_train.npy", allow_pickle=True)

        rank = np.argsort(-b)
        for i in range(len(items)):
            pop_dict[i] = rank[np.isin(rank, train_pos[i], invert=True)]
        np.save(p + 'pop_all.npy', pop_dict)

        for i in range(len(items)):
            pop_dict[i] = pop_dict[i][np.isin(pop_dict[i], test_pos[i], invert=False)]
        np.save(p + 'pop_test.npy', pop_dict)
        '''

        return np.stack(
                [items[i][np.isin(items[i], train_pos[i], invert=True)][:self.max_topk] for i in range(len(items))],
                axis=0)

    def compute_iou(self, items, pop_items):

        intersection = np.array([len(np.intersect1d(items[i], pop_items[i])) for i in range(len(items))])
        union = np.array([len(np.union1d(items[i], pop_items[i])) for i in range(len(items))])
        iou = np.sum(intersection/union)

        return iou

    def update_results(self, batch_results):

        for metric, value in batch_results.items():
            self.results[metric] = self.results[metric] + value

    def average_user(self, num_test_users):

        if num_test_users > self.n_user:
            num_test_users = self.n_user

        for metric, value in self.results.items():
            if metric in ['recall', 'hit_ratio', 'ndcg', 'iou']:
                self.results[metric] = value/num_test_users

