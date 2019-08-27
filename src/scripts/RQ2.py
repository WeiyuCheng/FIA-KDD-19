from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals
import numpy as np
import tensorflow as tf
import sys
import argparse
import os

sys.path.append("..")
from influence.matrix_factorization import MF
from influence.NCF import NCF
import influence.experiments as experiments
from scripts.load_movielens import load_movielens
from scripts.load_yelp import load_yelp


def parse_args():
    parser = argparse.ArgumentParser(description="Run RQ2 training.")
    parser.add_argument('--avextol', type=float, default=1e-3,
                        help='threshold for optimization')
    parser.add_argument('--damping', type=float, default=1e-6,
                        help='damping term')
    parser.add_argument('--embed_size', type=int, default=16,
                        help='embedding size')
    parser.add_argument('--maxinf', type=int, default=1,
                        help='remove type of train indices')
    return parser.parse_args()


args = parse_args()
data_sets_movielens = load_movielens('../../data')
data_sets_yelp = load_yelp('../../data')

embed_size = args.embed_size
# embed_sizes = [args.embed_size]
#
# for embed_size in embed_sizes:

# Matrix Factorization

for data_sets in [data_sets_movielens, data_sets_yelp]:
    for Model in [MF, NCF]:
        if data_sets is data_sets_movielens:
            test_idx = 59
            batch_size = 3020
            data_name = "movielens"
        else:
            test_idx = 1
            batch_size = 3009
            data_name = "yelp"
        if Model is MF:
            model_name = "MF"
            num_steps = 80000
        else:
            model_name = "NCF"
            num_steps = 120000

        num_users = np.unique(data_sets.train._x[:, 0]).shape[0]
        num_items = np.unique(data_sets.train._x[:, 1]).shape[0]

        weight_decay = 0.001
        initial_learning_rate = 0.001
        print("number of users: %d" % num_users)
        print("number of items: %d" % num_items)
        print("number of training examples: %d" % data_sets.train._x.shape[0])
        print("number of testing examples: %d" % data_sets.test._x.shape[0])
        args = parse_args()
        avextol = args.avextol
        damping = args.damping
        print("Using avextol of %.0e" % avextol)
        print("Using damping of %.0e" % damping)
        print("Using embedding size of %d" % embed_size)

        model = Model(
            num_users=num_users,
            num_items=num_items,
            embedding_size=embed_size,
            weight_decay=weight_decay,
            num_classes=1,
            batch_size=batch_size,
            data_sets=data_sets,
            initial_learning_rate=initial_learning_rate,
            damping=damping,
            decay_epochs=[100000, 200000],
            mini_batch=True,
            train_dir='output',
            log_dir='log',
            avextol=avextol,
            model_name='%s_%s_explicit_damping%.0e_avextol%.0e_embed%d_maxinf%d_wd%.0e' % (data_name, model_name,
                                                                                           damping, avextol, embed_size,
                                                                                           args.maxinf, weight_decay))
        iter_to_load = num_steps - 1
        print("Model name is %s" % model.model_name)
        if os.path.isfile("%s-%s.index" % (model.checkpoint_file, iter_to_load)):
            print('Checkpoint found, loading...')
            model.load_checkpoint(iter_to_load=iter_to_load)
        else:
            print('Checkpoint not found, start training...')
            model.train(num_steps=num_steps)
            model.saver.save(model.sess, model.checkpoint_file, global_step=num_steps - 1)
            print("Training finished")
        experiments.record_time_cost(
            model,
            test_idx=test_idx,
            iter_to_load=iter_to_load,
            force_refresh=True)
        print("This is the time cost of %s on dataset %s for embed %s" % (model_name, data_name, embed_size))
        print("--------------------------------------------------")
        print("--------------------------------------------------")
        print("--------------------------------------------------")
        model.sess.close()
        tf.reset_default_graph()
