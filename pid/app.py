#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

from absl import app
from absl import flags

import sys
sys.path.append('/data2/gpxu/popular_debias/pid/')
import os
import utils

FLAGS = flags.FLAGS

flags.DEFINE_string('name', 'ml100k-PIDMF-debug', 'Experiment name.')
flags.DEFINE_enum('model', 'PIDMF', ['PIDMF', 'MF'], 'Model name.')
flags.DEFINE_enum('dataset', 'ml100k', ['ml100k', 'music', 'epinion'], 'Dataset.')

flags.DEFINE_integer('num_layers', 2, 'The number of layers for LGN.')
flags.DEFINE_float('dropout', 0.2, 'Dropout ratio for LGN.')

flags.DEFINE_integer('margin', 40, 'Margin for negative sampling.')
flags.DEFINE_integer('pool', 40, 'Pool for negative sampling.')
flags.DEFINE_bool('adaptive', False, 'Adapt hyper-parameters or not.')
flags.DEFINE_float('margin_decay', 0.9, 'Decay of margin and pool.')
flags.DEFINE_float('loss_decay', 0.9, 'Decay of loss.')

flags.DEFINE_enum('weighting_mode', 'nc', ['n', 'c', 'nc', 'x'], 'Mode of IPS technique.')
flags.DEFINE_float('weighting_smoothness', 0.8, 'IPS weighting smoothness.')

flags.DEFINE_bool('use_gpu', True, 'Use GPU or not.')  #True
flags.DEFINE_integer('gpu_id', 0, 'GPU ID.')

flags.DEFINE_bool('cg_use_gpu', False, 'Use GPU or not for candidate generation.')
flags.DEFINE_integer('cg_gpu_id', 0, 'GPU ID for candidate generation.')

flags.DEFINE_integer('embedding_size', 64, 'Embedding size for embedding based models.')
flags.DEFINE_integer('epochs', 50, 'Max epochs for training.')
flags.DEFINE_float('lr', 0.001, 'Learning rate.')
flags.DEFINE_float('min_lr', 0.00001, 'Minimum learning rate.')
flags.DEFINE_float('weight_decay', 5e-6, 'Weight decay.')  # 5e-6
flags.DEFINE_integer('batch_size', 512, 'Batch Size.')

flags.DEFINE_integer('neg_sample_rate', 4, 'Negative Sampling Ratio.')
flags.DEFINE_bool('shuffle', True, 'Shuffle the training set or not.')

flags.DEFINE_multi_string('metrics', ['recall', 'hit_ratio', 'ndcg'], 'Metrics.')
flags.DEFINE_multi_string('val_metrics', ['recall', 'hit_ratio', 'ndcg'], 'Metrics.')
flags.DEFINE_string('watch_metric', 'recall', 'Which metric to decide learning rate reduction.')
flags.DEFINE_integer('patience', 5, 'Patience for reducing learning rate.')
flags.DEFINE_integer('es_patience', 3, 'Patience for early stop.')

flags.DEFINE_integer('num_val_users', 1000000, 'Number of users for validation.')
flags.DEFINE_integer('num_test_users', 1000000, 'Number of users for test.')
flags.DEFINE_enum('test_model', 'best', ['best', 'last'], 'Which model to test.')
flags.DEFINE_multi_integer('topk', [20], 'Topk for testing recommendation performance.') # 20, 50
flags.DEFINE_integer('num_workers', 16, 'Number of processes for training and testing.')

flags.DEFINE_string('load_path', '', 'Load path.')
flags.DEFINE_string('workspace', './', 'Path to load ckpt.')
flags.DEFINE_string('output', '/data2/gpxu/popular_debias/pid/result/', 'Directory to save model/log/metrics.')


def main(argv):

    flags_obj = FLAGS
    cm = utils.ContextManager(flags_obj)
    dm = utils.DatasetManager(flags_obj)

    dm.get_dataset_info(flags_obj)  # load dataset
    cm.set_default_ui()  # create workplace
    cm.logging_flags(flags_obj)
    trainer = utils.ContextManager.set_trainer(flags_obj, cm, dm)

    trainer.train()
    # 进行测试的时候，把 上一行注释掉
    trainer.test()


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '5'
    app.run(main)

