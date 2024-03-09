# python3
# Make this standard template for testing and training
from __future__ import division
from __future__ import print_function

import sys
import os
import shutil

sys.path.append('%s/gcn' % os.path.dirname(os.path.realpath(__file__)))
# add the libary path for graph reduction and local search
# sys.path.append( '%s/kernel' % os.path.dirname(os.path.realpath(__file__)) )
import time
import random
import scipy.io as sio
import numpy as np
import scipy.sparse as sp
from multiprocessing import Queue
from copy import deepcopy
import networkx as nx

import tensorflow as tf
from collections import deque
from gcn.models import GCN2_DQN
from gcn.utils import *
# import the libary for graph reduction and local search
# from reduce_lib import reducelib
import warnings

warnings.filterwarnings('ignore')

from runtime_config import flags, FLAGS
# Settings (FLAGS)
from test_utils import *
from heuristics import *

if not hasattr(flags.FLAGS, 'epsilon'):
    flags.DEFINE_float('epsilon', 1.0, 'initial exploration rate')
if not hasattr(flags.FLAGS, 'epsilon_min'):
    flags.DEFINE_float('epsilon_min', 0.001, 'minimal exploration rate')
if not hasattr(flags.FLAGS, 'epsilon_decay'):
    flags.DEFINE_float('epsilon_decay', 0.985, 'exploration rate decay per replay')
if not hasattr(flags.FLAGS, 'gamma'):
    flags.DEFINE_float('gamma', 1.0, 'gamma')

# Some preprocessing
num_supports = 1 + FLAGS.max_degree
model_func = GCN2_DQN
nsr = np.power(10.0, -FLAGS.snr_db / 20.0)


class MWISSolver(object):
    def __init__(self, input_flags, memory_size):
        self.feature_size = input_flags.feature_size
        self.memory = deque(maxlen=memory_size)
        self.reward_mem = deque(maxlen=memory_size)
        self.flags = input_flags
        self.placeholders = {
            'support': [tf.compat.v1.sparse_placeholder(tf.float32) for _ in range(num_supports)],
            'features': tf.compat.v1.sparse_placeholder(tf.float32, shape=(None, self.flags.feature_size)),
            # featureless: #points
            'adj': tf.compat.v1.sparse_placeholder(tf.float32),
            'labels': tf.compat.v1.placeholder(tf.float32, shape=(None, 1)),  # rewards
            'actions': tf.compat.v1.placeholder(tf.float32, shape=(None, 1)),  # real actions, including exploration
            'labels_mask': tf.compat.v1.placeholder(tf.int32),
            'dropout': tf.compat.v1.placeholder_with_default(0., shape=()),
            'num_features_nonzero': tf.compat.v1.placeholder(tf.int32)  # helper variable for sparse dropout
        }
        self.delta = 0.000001  # prevent empty solution
        self.gamma = self.flags.gamma  # discount rate
        self.epsilon = self.flags.epsilon  # exploration rate
        self.epsilon_min = self.flags.epsilon_min
        self.epsilon_decay = self.flags.epsilon_decay
        self.learning_rate = self.flags.learning_rate
        self.sess = None
        # self.writer = tf.summary.create_file_writer('./logs/metrics', max_queue=10000)
        self.saver = None

    def _build_model(self):
        raise NotImplementedError

    def makestate(self, adj, wts_nn):
        reduced_nn = wts_nn.shape[0]
        norm_wts = np.amax(wts_nn) + 1e-9
        if self.flags.predict == 'mwis':
            features = np.ones([reduced_nn, self.flags.feature_size])
        else:
            features = np.multiply(np.ones([reduced_nn, self.flags.feature_size]), wts_nn / norm_wts)
        features_raw = features.copy()
        features = sp.lil_matrix(features)
        if self.flags.predict == 'mwis':
            features = preprocess_features(features)
        else:
            features = sparse_to_tuple(features)
        support = simple_polynomials(adj, self.flags.max_degree)
        state = {"features": features, "support": support, "features_raw": features_raw}
        return state

    def act(self, state, train):
        raise NotImplementedError

    def predict(self, state):
        raise NotImplementedError

    def memorize(self, state, act_vals, solu, next_state, reward):
        self.memory.append((state.copy(), act_vals.copy(), solu.copy(), next_state.copy(), reward))
        # self.reward_mem.append(reward)

    def load(self, name):
        ckpt = tf.train.get_checkpoint_state(name)
        if ckpt:
            with self.sess.as_default():
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print('loaded ' + ckpt.model_checkpoint_path)

    def save(self, name):
        with self.sess.as_default():
            self.saver.save(self.sess, os.path.join(name, "model.ckpt"))

    def copy_model_parameters(self, estimator1, estimator2):
        """
        Copies the model parameters of one estimator to another.
        Args:
          sess: Tensorflow session instance
          estimator1: Estimator to copy the paramters from
          estimator2: Estimator to copy the parameters to
        """
        e1_params = [t for t in tf.compat.v1.trainable_variables() if t.name.startswith(estimator1)]
        e1_params = sorted(e1_params, key=lambda v: v.name)
        e2_params = [t for t in tf.compat.v1.trainable_variables() if t.name.startswith(estimator2)]
        e2_params = sorted(e2_params, key=lambda v: v.name)

        update_ops = []
        for e1_v, e2_v in zip(e1_params, e2_params):
            op = e2_v.assign(e1_v)
            update_ops.append(op)

        self.sess.run(update_ops)

    def mellowmax(self, q_vec, omega, beta):
        c = np.max(q_vec)
        a_size = np.size(q_vec)
        mellow = c + np.log(np.sum(np.exp(omega * (q_vec - c))) / a_size) / omega
        # ans = np.sum(np.exp((q_vec-mellow)*beta)*(q_vec-mellow))
        return mellow

    def utility(self, adj_0, wts_0, train=False):
        """
        GCN for per utility function
        """
        adj = adj_0.copy()
        wts_nn = np.reshape(wts_0, (wts_0.shape[0], self.flags.feature_size))

        # GCN
        state = self.makestate(adj, wts_nn)
        act_vals = self.act(state, train)

        gcn_wts = act_vals

        return gcn_wts, state

    def schedule(self, adj_0, wts_0, train=False):
        """
        GCN followed by LGS
        """
        adj = adj_0.copy()
        wts_nn = np.reshape(wts_0, (wts_0.shape[0], self.flags.feature_size))

        # GCN
        state = self.makestate(adj, wts_nn)
        act_vals = self.act(state, train)

        if self.flags.predict == 'mwis':
            # gcn_wts = np.divide(wts_nn.flatten(), act_vals.flatten()+1e-8)
            gcn_wts = np.multiply(act_vals.flatten(), wts_nn.flatten())
            # gcn_wts = act_vals.flatten()+100
        else:
            gcn_wts = act_vals.flatten()
            # gcn_wts = np.multiply(act_vals.flatten(), wts_nn.flatten())
        # gcn_wts = np.multiply(act_vals.flatten(), wts_nn.flatten()) + wts_nn.flatten()

        mwis, _ = local_greedy_search(adj, gcn_wts)
        # mwis, _ = greedy_search(adj, gcn_wts)
        solu = list(mwis)
        mwis_rt = mwis
        total_wt = np.sum(wts_nn[solu, 0])
        return mwis_rt, total_wt, state, act_vals

    def topology_encode(self, adj_0, wts_0, train=False):
        # buffer = deque(maxlen=20)
        adj = adj_0.copy()
        wts_nn = np.reshape(wts_0, (wts_0.size, 1))

        # GCN
        state = self.makestate(adj, wts_nn)
        act_vals = self.act(state, train)

        return act_vals

    def solve_mwis(self, adj_0, wts_0, train=False, grd=1.0):
        """
        GCN followed by LGS
        """
        adj = adj_0.copy()
        wts_nn = np.reshape(wts_0, (wts_0.shape[0], self.flags.feature_size))

        # GCN
        state = self.makestate(adj, wts_nn)
        act_vals, act = self.act(state, train)

        if self.flags.predict == 'mwis':
            # gcn_wts = np.divide(wts_nn.flatten(), act_vals.flatten()+1e-8)
            gcn_wts = np.multiply(act_vals.flatten(), wts_nn.flatten())
            # gcn_wts = act_vals.flatten()+100
        else:
            gcn_wts = act_vals.flatten()
            # gcn_wts = np.multiply(act_vals.flatten(), wts_nn.flatten())
        # gcn_wts = np.multiply(act_vals.flatten(), wts_nn.flatten()) + wts_nn.flatten()

        mwis, _ = local_greedy_search(adj, gcn_wts)
        # mwis, _ = greedy_search(adj, gcn_wts)
        solu = list(mwis)
        mwis_rt = mwis
        total_wt = np.sum(wts_nn[solu, 0])
        if train:
            # wts_norm = wts_nn[list(sol_gd), :]/greedy_util.flatten()
            # self.memorize(state.copy(), act_vals.copy(), list(sol_gd), wts_norm, 1.0)
            # reward = (total_wt + self.smallconst) / (greedy_util.flatten()[0] + self.smallconst)
            reward = total_wt / (grd + 1e-6)
            # reward = reward if reward > 0 else 0
            wts_norm = wts_nn / np.amax(wts_nn)
            if not np.isnan(reward):
                self.memorize(state.copy(), act_vals.copy(), list(mwis), {}, reward)
            return mwis_rt, total_wt
        return mwis_rt, total_wt

    def solve_mwis_util(self, adj_0, wts_0, wts_u, train=False, grd=1.0):
        """
        This function is to be compatible for utility learning
        """
        # buffer = deque(maxlen=20)
        adj = adj_0.copy()
        wts_nn = np.reshape(wts_0, (wts_0.shape[0], self.flags.feature_size))

        # GCN
        state = self.makestate(adj, wts_nn)
        act_vals, act = self.act(state, train)

        if self.flags.predict == 'mwis':
            # gcn_wts = np.divide(wts_nn.flatten(), act_vals.flatten()+1e-8)
            gcn_wts = np.multiply(act_vals.flatten(), wts_nn.flatten())
            # gcn_wts = act_vals.flatten()+100
        else:
            gcn_wts = act_vals.flatten()
            # gcn_wts = np.multiply(act_vals.flatten(), wts_nn.flatten())
        # gcn_wts = np.multiply(act_vals.flatten(), wts_nn.flatten()) + wts_nn.flatten()

        mwis, _ = local_greedy_search(adj, gcn_wts)
        # mwis, _ = greedy_search(adj, gcn_wts)
        solu = list(mwis)
        mwis_rt = mwis
        total_wt = np.sum(wts_u[solu])
        if train:
            # wts_norm = wts_nn[list(sol_gd), :]/greedy_util.flatten()
            # self.memorize(state.copy(), act_vals.copy(), list(sol_gd), wts_norm, 1.0)
            # reward = (total_wt + self.smallconst) / (greedy_util.flatten()[0] + self.smallconst)
            reward = total_wt / (grd + 1e-6)
            # reward = reward if reward > 0 else 0
            if self.flags.predict == 'mwis':
                wts_norm = wts_nn / grd
            else:
                wts_norm = wts_nn
            if not np.isnan(reward):
                self.memorize(state.copy(), act_vals.copy(), list(mwis), wts_u, reward)
            return mwis_rt, total_wt
        return mwis_rt, total_wt

    def solve_mwis_dit(self, adj_0, wts_0, train=False, grd=1.0):
        """
        GCN embedded into LGS iteration
        """
        wts = np.reshape(wts_0, (wts_0.shape[0], self.flags.feature_size))

        it_cnt = 0
        best_IS_util = np.array([0.0])
        reduced_nn = adj_0.shape[0]
        nIS_vec = -np.ones_like(wts).flatten()
        while reduced_nn > 0:
            it_cnt += 1
            remain_vec = (nIS_vec == -1)
            reverse_mapping = np.argwhere((nIS_vec == -1))
            reverse_mapping = reverse_mapping[:, 0]
            adj_nn = adj_0.copy()
            adj_nn = adj_nn[remain_vec, :]
            adj_nn = adj_nn[:, remain_vec]
            wts_nn = wts.copy()
            wts_nn = wts_nn[remain_vec, :]
            if np.sum(wts_nn) <= 0:
                break
            # GCN
            state = self.makestate(adj_nn, wts_nn)
            act_vals, act = self.act(state, train)

            if self.flags.predict == 'mwis':
                gcn_wts = np.multiply(act_vals.flatten(), wts_nn.flatten())
            else:
                gcn_wts = act_vals.flatten()
            # 1-step LGS
            sol_part, _, nb_is = local_greedy_search_nstep(adj_nn, gcn_wts, nstep=1)
            # post proc
            nIS_vec[reverse_mapping[list(sol_part)]] = 1
            nIS_vec[reverse_mapping[list(nb_is)]] = 0
            best_IS_util = np.dot(nIS_vec, wts)
            reduced_nn = np.sum((nIS_vec == -1))

        solu = np.argwhere((nIS_vec == 1))
        mwis = set(solu.flatten())
        return mwis, best_IS_util

    def solve_mwis_cit_wrap(self, adj_0, wts_0, train=False, grd=1.0):
        wts = np.reshape(wts_0, (wts_0.shape[0], self.flags.feature_size))
        g = nx.from_scipy_sparse_matrix(adj_0)
        subgraphs = list(nx.connected_components(g))
        best_IS_util = np.array([0.0])
        nIS_vec = -np.ones_like(wts).flatten()
        for subgraph in subgraphs:
            subgraph = list(subgraph)
            sub_vec = np.zeros_like(wts, dtype=np.bool).flatten()
            sub_vec[subgraph] = True
            adj_sub = adj_0.copy()
            adj_sub = adj_sub[sub_vec, :]
            adj_sub = adj_sub[:, sub_vec]
            wts_sub = wts.copy()
            wts_sub = wts_sub[sub_vec, :]
            mwis_sub, util_sub = self.solve_mwis_cit(adj_sub, wts_sub, train=train, grd=grd)
            best_IS_util += util_sub
            mwis_idx = list(mwis_sub)
            mwis_map = [subgraph[i] for i in mwis_idx]
            nIS_vec[mwis_map] = 1
        solu = np.argwhere((nIS_vec == 1))
        mwis = set(solu.flatten())
        return mwis, best_IS_util

    def solve_mwis_cit(self, adj_0, wts_0, train=False, grd=1.0):
        """
        GCN in combining with centralized greedy iteration
        """
        wts = np.reshape(wts_0, (wts_0.shape[0], self.flags.feature_size))

        it_cnt = 0
        best_IS_util = np.array([0.0])
        reduced_nn = adj_0.shape[0]
        nIS_vec = -np.ones_like(wts).flatten()
        while reduced_nn > 0:
            it_cnt += 1
            remain_vec = (nIS_vec == -1)
            reverse_mapping = np.argwhere((nIS_vec == -1))
            reverse_mapping = reverse_mapping[:, 0]
            adj_nn = adj_0.copy()
            adj_nn = adj_nn[remain_vec, :]
            adj_nn = adj_nn[:, remain_vec]
            wts_nn = wts.copy()
            wts_nn = wts_nn[remain_vec, :]
            if np.sum(wts_nn) <= 0:
                break
            # GCN
            state = self.makestate(adj_nn, wts_nn)
            act_vals, act = self.act(state, train)

            if self.flags.predict == 'mwis':
                gcn_wts = np.multiply(act_vals.flatten(), wts_nn.flatten())
            else:
                gcn_wts = act_vals.flatten()
            # 1-step CGS
            sol_part = np.argmax(gcn_wts)
            _, nb_v = np.nonzero(adj_nn[sol_part])
            nIS_vec[reverse_mapping[sol_part]] = 1
            nIS_vec[reverse_mapping[nb_v]] = 0
            best_IS_util = np.dot(nIS_vec, wts)
            reduced_nn = np.sum((nIS_vec == -1))

        solu = np.argwhere((nIS_vec == 1))
        mwis = set(solu.flatten())
        return mwis, best_IS_util

    def solve_mwis_rollout_wrap(self, adj_0, wts_0, train=False, grd=1.0, b=16):
        wts = np.reshape(wts_0, (wts_0.shape[0], self.flags.feature_size))
        g = nx.from_scipy_sparse_matrix(adj_0)
        subgraphs = list(nx.connected_components(g))
        best_IS_util = np.array([0.0])
        nIS_vec = -np.ones_like(wts).flatten()
        for subgraph in subgraphs:
            subgraph = list(subgraph)
            sub_vec = np.zeros_like(wts, dtype=np.bool).flatten()
            sub_vec[subgraph] = True
            adj_sub = adj_0.copy()
            adj_sub = adj_sub[sub_vec, :]
            adj_sub = adj_sub[:, sub_vec]
            wts_sub = wts.copy()
            wts_sub = wts_sub[sub_vec, :]
            # GCN-CRS-e
            # mwis_sub, util_sub = self.solve_mwis_rollout1(adj_sub, wts_sub, train=train, grd=grd, b=b)
            # GCN-CRS-v
            mwis_sub, util_sub = self.solve_mwis_rollout(adj_sub, wts_sub, train=train, grd=grd, b=b)
            best_IS_util += util_sub
            mwis_idx = list(mwis_sub)
            mwis_map = [subgraph[i] for i in mwis_idx]
            nIS_vec[mwis_map] = 1
        solu = np.argwhere((nIS_vec == 1))
        mwis = set(solu.flatten())
        return mwis, best_IS_util

    def solve_mwis_rollout00(self, adj_0, wts_0, train=False, grd=1.0, b=16):
        """ Use top b nodes as branches run rollout """
        wts = np.reshape(wts_0, (wts_0.shape[0], self.flags.feature_size))
        it_cnt = 0
        best_IS_util = np.array([0.0])
        reduced_nn = adj_0.shape[0]
        nIS_vec = -np.ones_like(wts).flatten()
        # GCN
        state = self.makestate(adj_0, wts)
        act_vals, act = self.act(state, train)
        while reduced_nn > 0:
            it_cnt += 1
            remain_vec = (nIS_vec == -1)
            reverse_mapping = np.argwhere((nIS_vec == -1))
            reverse_mapping = reverse_mapping[:, 0]
            adj_nn = adj_0.copy()
            adj_nn = adj_nn[remain_vec, :]
            adj_nn = adj_nn[:, remain_vec]
            wts_nn = wts.copy()
            wts_nn = wts_nn[remain_vec, :]
            if np.sum(wts_nn) <= 0:
                break

            if self.flags.predict == 'mwis':
                gcn_wts = np.multiply(act_vals[remain_vec, :], wts_nn)
            else:
                gcn_wts = act_vals[remain_vec, :]
            # # Rollout branches
            ranks = np.argsort(-gcn_wts.flatten())
            # ranks = np.argsort(-act_vals.flatten())
            children = ranks[0:b]
            scores = wts_nn[children]
            if len(scores) > 1:
                for i in range(len(children)):
                    child = children[i]
                    remain_rollout = np.ones((reduced_nn,), dtype=np.bool)
                    remain_rollout[child] = False
                    _, nb_v = np.nonzero(adj_nn[child])
                    remain_rollout[nb_v] = False
                    adj_ro = adj_nn[remain_rollout, :]
                    adj_ro = adj_ro[:, remain_rollout]
                    wts_ro = wts_nn[remain_rollout]
                    gw_ro = gcn_wts[remain_rollout]
                    ps, ss_eval = greedy_search(adj_ro, wts_ro)
                    ss_eval = np.sum(wts_ro[list(ps)])
                    scores[i] += ss_eval

            # 1-step CGS
            i_best = np.random.choice(np.flatnonzero(scores == scores.max()))
            # i_best = np.argmax(scores)
            sol_part = children[i_best]
            _, nb_v = np.nonzero(adj_nn[sol_part])
            nIS_vec[reverse_mapping[sol_part]] = 1
            nIS_vec[reverse_mapping[nb_v]] = 0
            best_IS_util = np.dot(nIS_vec, wts_0)
            reduced_nn = np.sum((nIS_vec == -1))

        solu = np.argwhere((nIS_vec == 1))
        mwis = set(solu.flatten())
        return mwis, best_IS_util

    def solve_mwis_rollout0(self, adj_0, wts_0, train=False, grd=1.0, b=16):
        """ Use top b nodes as branches run rollout """
        wts = np.reshape(wts_0, (wts_0.shape[0], self.flags.feature_size))
        it_cnt = 0
        best_IS_util = np.array([0.0])
        reduced_nn = adj_0.shape[0]
        nIS_vec = -np.ones_like(wts).flatten()
        # GCN
        state = self.makestate(adj_0, wts)
        act_vals, act = self.act(state, train)
        while reduced_nn > 0:
            it_cnt += 1
            remain_vec = (nIS_vec == -1)
            reverse_mapping = np.argwhere((nIS_vec == -1))
            reverse_mapping = reverse_mapping[:, 0]
            adj_nn = adj_0.copy()
            adj_nn = adj_nn[remain_vec, :]
            adj_nn = adj_nn[:, remain_vec]
            wts_nn = wts.copy()
            wts_nn = wts_nn[remain_vec, :]
            if np.sum(wts_nn) <= 0:
                break

            if self.flags.predict == 'mwis':
                gcn_wts = np.multiply(act_vals[remain_vec, :], wts_nn)
            else:
                gcn_wts = act_vals[remain_vec, :]
            # # Rollout branches
            ranks = np.argsort(-gcn_wts.flatten())
            # ranks = np.argsort(-act_vals.flatten())
            children = ranks[0:b]
            scores = wts_nn[children]
            if len(scores) > 1:
                for i in range(len(children)):
                    child = children[i]
                    remain_rollout = np.ones((reduced_nn,), dtype=np.bool)
                    remain_rollout[child] = False
                    _, nb_v = np.nonzero(adj_nn[child])
                    remain_rollout[nb_v] = False
                    adj_ro = adj_nn[remain_rollout, :]
                    adj_ro = adj_ro[:, remain_rollout]
                    wts_ro = wts_nn[remain_rollout]
                    gw_ro = gcn_wts[remain_rollout]
                    ps, ss_eval = greedy_search(adj_ro, gw_ro)
                    ss_eval = np.sum(wts_ro[list(ps)])
                    scores[i] += ss_eval

            # 1-step CGS
            i_best = np.random.choice(np.flatnonzero(scores == scores.max()))
            # i_best = np.argmax(scores)
            sol_part = children[i_best]
            _, nb_v = np.nonzero(adj_nn[sol_part])
            nIS_vec[reverse_mapping[sol_part]] = 1
            nIS_vec[reverse_mapping[nb_v]] = 0
            best_IS_util = np.dot(nIS_vec, wts_0)
            reduced_nn = np.sum((nIS_vec == -1))

        solu = np.argwhere((nIS_vec == 1))
        mwis = set(solu.flatten())
        return mwis, best_IS_util

    def solve_mwis_rollout1(self, adj_0, wts_0, train=False, grd=1.0, b=16):
        """ Use top b nodes as branches run rollout """
        wts = np.reshape(wts_0, (wts_0.shape[0], self.flags.feature_size))
        it_cnt = 0
        best_IS_util = np.array([0.0])
        reduced_nn = adj_0.shape[0]
        nIS_vec = -np.ones_like(wts_0).flatten()
        # GCN
        while reduced_nn > 0:
            it_cnt += 1
            remain_vec = (nIS_vec == -1)
            reverse_mapping = np.argwhere((nIS_vec == -1))
            reverse_mapping = reverse_mapping[:, 0]
            adj_nn = adj_0.copy()
            adj_nn = adj_nn[remain_vec, :]
            adj_nn = adj_nn[:, remain_vec]
            wts_nn = wts.copy()
            wts_nn = wts_nn[remain_vec, :]
            if np.sum(wts_nn) <= 0:
                break

            state = self.makestate(adj_nn, wts_nn)
            act_vals, act = self.act(state, train)
            if self.flags.predict == 'mwis':
                gcn_wts = np.multiply(act_vals, wts_nn)
            else:
                gcn_wts = act_vals
            # # Rollout branches
            ranks = np.argsort(-gcn_wts.flatten())
            # ranks = np.argsort(-act_vals.flatten())
            children = ranks[0:b]
            scores = wts_nn[children]
            if len(scores) > 1:
                for i in range(len(children)):
                    child = children[i]
                    remain_rollout = np.ones((reduced_nn,), dtype=np.bool)
                    remain_rollout[child] = False
                    _, nb_v = np.nonzero(adj_nn[child])
                    remain_rollout[nb_v] = False
                    adj_ro = adj_nn[remain_rollout, :]
                    adj_ro = adj_ro[:, remain_rollout]
                    wts_ro = wts_nn[remain_rollout]
                    gw_ro = gcn_wts[remain_rollout]
                    ps, ss_eval = greedy_search(adj_ro, gw_ro)
                    ss_eval = np.sum(wts_ro[list(ps)])
                    scores[i] += ss_eval

            # 1-step CGS
            i_best = np.random.choice(np.flatnonzero(scores == scores.max()))
            # i_best = np.argmax(scores)
            sol_part = children[i_best]
            _, nb_v = np.nonzero(adj_nn[sol_part])
            nIS_vec[reverse_mapping[sol_part]] = 1
            nIS_vec[reverse_mapping[nb_v]] = 0
            best_IS_util = np.dot(nIS_vec, wts_0)
            reduced_nn = np.sum((nIS_vec == -1))

        solu = np.argwhere((nIS_vec == 1))
        mwis = set(solu.flatten())
        return mwis, best_IS_util

    def solve_mwis_rollout(self, adj_0, wts_0, train=False, grd=1.0, b=16):
        """ Use top b nodes as branches run rollout """
        wts = np.reshape(wts_0, (wts_0.shape[0], self.flags.feature_size))
        it_cnt = 0
        best_IS_util = np.array([0.0])
        reduced_nn = adj_0.shape[0]
        nIS_vec = -np.ones_like(wts).flatten()
        while reduced_nn > 0:
            it_cnt += 1
            remain_vec = (nIS_vec == -1)
            reverse_mapping = np.argwhere((nIS_vec == -1))
            reverse_mapping = reverse_mapping[:, 0]
            adj_nn = adj_0.copy()
            adj_nn = adj_nn[remain_vec, :]
            adj_nn = adj_nn[:, remain_vec]
            wts_nn = wts.copy()
            wts_nn = wts_nn[remain_vec, :]
            if np.sum(wts_nn) <= 0:
                break
            # GCN
            state = self.makestate(adj_nn, wts_nn)
            act_vals, act = self.act(state, train)

            if self.flags.predict == 'mwis':
                gcn_wts = np.multiply(act_vals.flatten(), wts_nn.flatten())
            else:
                gcn_wts = act_vals.flatten()
            # # Rollout branches
            ranks = np.argsort(-gcn_wts.flatten())
            # ranks = np.argsort(-act_vals.flatten())
            children = ranks[0:b]
            scores = wts_nn[children]
            if len(scores) > 1:
                for i in range(len(children)):
                    child = children[i]
                    remain_rollout = np.ones((reduced_nn,), dtype=np.bool)
                    remain_rollout[child] = False
                    _, nb_v = np.nonzero(adj_nn[child])
                    remain_rollout[nb_v] = False
                    adj_ro = adj_nn[remain_rollout, :]
                    adj_ro = adj_ro[:, remain_rollout]
                    wts_ro = wts_nn[remain_rollout]
                    # if np.sum(remain_rollout) > 1:
                    #     _, ss_eval = self.solve_mwis(adj_ro, wts_ro)
                    # elif np.sum(remain_rollout) == 1:
                    #     ss_eval = wts_ro[0]
                    # else:
                    #     ss_eval = 0.0
                    _, ss_eval = greedy_search(adj_ro, wts_ro)
                    scores[i] += ss_eval

            # 1-step CGS
            i_best = np.random.choice(np.flatnonzero(scores == scores.max()))
            # i_best = np.argmax(scores)
            sol_part = children[i_best]
            _, nb_v = np.nonzero(adj_nn[sol_part])
            nIS_vec[reverse_mapping[sol_part]] = 1
            nIS_vec[reverse_mapping[nb_v]] = 0
            best_IS_util = np.dot(nIS_vec, wts)
            reduced_nn = np.sum((nIS_vec == -1))

        solu = np.argwhere((nIS_vec == 1))
        mwis = set(solu.flatten())
        return mwis, best_IS_util


class DQNAgent(MWISSolver):
    def __init__(self, input_flags, memory_size=5000):
        super(DQNAgent, self).__init__(input_flags, memory_size)
        self.sess = tf.compat.v1.Session(config=config)
        with tf.name_scope('model') as scope:
            self.model = self._build_model()
        with tf.name_scope('target') as scope:
            self.target_model = self._build_model()
        self.gamma = 1.0
        self.update_cnt = 0
        self.C = 10
        with self.sess.as_default():
            self.sess.run(tf.compat.v1.global_variables_initializer())
        # self.writer = tf.summary.create_file_writer('./logs/metrics', max_queue=10000)
        self.saver = tf.compat.v1.train.Saver(max_to_keep=1000)

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = model_func(self.placeholders,
                           hidden_dim=self.flags.hidden1,
                           num_layer=self.flags.num_layer,
                           bias=True,
                           learning_rate=self.flags.learning_rate,
                           learning_decay=self.flags.learning_decay,
                           weight_decay=self.flags.weight_decay,
                           logging=True)
        return model

    def predict(self, state):
        feed_dict_val = construct_feed_dict4pred(state["features"], state["support"], self.placeholders)
        with self.sess.as_default():
            act_values, action = self.sess.run([self.model.outputs_softmax, self.model.pred], feed_dict=feed_dict_val)
        return act_values, action

    def act(self, state, train):
        act_values, action = self.predict(state)
        if train:
            if np.random.rand() <= self.epsilon:
                # act_values = np.random.uniform(size=act_values.shape)
                # action = np.argmax(act_values.flatten())
                action = np.random.rand(act_values.size, 1)
            # act_vals = np.random.normal(0,0.5,size=act_values.shape)
            # act_values = np.where(act_rand < self.epsilon, act_vals, act_values)
        return act_values, action  # returns action

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return None
        if self.update_cnt > self.C or self.update_cnt == 0:
            self.update_target_model()
            self.update_cnt = 0
        self.update_cnt += 1
        minibatch = random.sample(self.memory, batch_size)
        losses_act = []
        losses_crt = []
        states, targets_f = [], []
        batch_avg = 0
        batch_len = 0
        batch_var = 0
        batch_tgt = np.array([])
        with self.sess.as_default():
            for state, act_vals, action, next_state, reward in minibatch:
                target = reward
                target_f = act_vals
                # target_f = np.zeros_like(act_vals)
                # feed_dict = construct_feed_dict4pred(state['features'], state['support'], self.placeholders)
                # target_f, = self.sess.run([self.target_model.outputs_softmax], feed_dict=feed_dict)
                # if next_state:
                #     feed_dict = construct_feed_dict4pred(next_state['features'],
                #                                          next_state['support'],
                #                                          self.placeholders)
                #     val_next_state, act = self.sess.run([self.target_model.outputs_softmax, self.target_model.pred], feed_dict=feed_dict)
                #     # mm = mellowmax(val_next_state, 2, 1)
                #     mm = np.amax(val_next_state)
                #     target += self.gamma * mm
                # target += self.gamma * np.amax(val_next_state)
                # act_pred, _ = self.act(state, False)
                # target_f[:, :] += act_pred
                target_f[action, :] += target
                states.append(state.copy())
                targets_f.append(target_f)
                # batch_avg += np.nansum(target_f)
                # batch_len += np.size(target_f)
                # batch_var += np.sum(target_f**2)
                batch_tgt = np.append(batch_tgt, target_f)
                # feed_dict = construct_feed_dict(state['features'], state['support'], target_f, self.placeholders)
                # _, loss = self.sess.run([self.model.opt_op, self.model.loss], feed_dict=feed_dict)
                # losses_crt.append(loss)

            batch_tgt = batch_tgt.flatten()
            batch_avg = np.mean(batch_tgt)
            batch_std = np.std(batch_tgt)
            for i in range(len(targets_f)):
                state = states[i]
                target_f = targets_f[i] / batch_std - batch_avg + 1.0
                feed_dict = construct_feed_dict(state['features'], state['support'], target_f, self.placeholders)
                _, loss = self.sess.run([self.model.opt_op, self.model.loss], feed_dict=feed_dict)
                losses_crt.append(loss)

        self.memory.clear()
        # Keeping track of loss
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        # with self.writer.as_default():
        #     tf.summary.scalar("critic loss", np.nanmean(losses_crt), step=self.step)
        #     tf.summary.scalar("actor loss", np.nanmean(losses_act), step=self.step)
        #     self.step += 1
        return np.nanmean(losses_crt)

    def update_target_model(self):
        # self.target_model.set_weights(self.model.get_weights())
        # for l_tg, l_sr in zip(self.target_model.layers, self.model.layers):
        #     # wk0 = l_sr.get_weights()
        #     l_tg.vars = l_sr.vars
        self.copy_model_parameters('model', 'target')

    def solve_mwis_cgs_train(self, adj_0, wts_0, train=False, grd=1.0):
        """
        GCN in combining with centralized greedy iteration
        """
        buffers = deque(maxlen=500)
        wts = np.reshape(wts_0, (wts_0.shape[0], self.flags.feature_size))

        it_cnt = 0
        best_IS_util = np.array([0.0])
        reduced_nn = adj_0.shape[0]
        nIS_vec = -np.ones_like(wts).flatten()
        while reduced_nn > 0:
            it_cnt += 1
            remain_vec = (nIS_vec == -1)
            reverse_mapping = np.argwhere((nIS_vec == -1))
            reverse_mapping = reverse_mapping[:, 0]
            adj_nn = adj_0.copy()
            adj_nn = adj_nn[remain_vec, :]
            adj_nn = adj_nn[:, remain_vec]
            wts_nn = wts.copy()
            wts_nn = wts_nn[remain_vec, :]
            if np.sum(wts_nn) <= 0:
                break
            # GCN
            state = self.makestate(adj_nn, wts_nn)
            act_vals, act = self.act(state, train)

            if self.flags.predict == 'mwis':
                gcn_wts = np.multiply(act_vals.flatten(), wts_nn.flatten())
            else:
                gcn_wts = act_vals.flatten()
            # 1-step CGS
            sol_part = np.argmax(gcn_wts)
            _, nb_v = np.nonzero(adj_nn[sol_part])
            nIS_vec[reverse_mapping[sol_part]] = 1
            nIS_vec[reverse_mapping[nb_v]] = 0
            best_IS_util = np.dot(nIS_vec, wts)
            reduced_nn = np.sum((nIS_vec == -1))
            # memorize process
            if train:
                buffers.append((state.copy(), act_vals.copy(), sol_part.copy()))

        solu = np.argwhere((nIS_vec == 1))
        mwis = set(solu.flatten())
        # backtrack memory
        if train:
            reward = best_IS_util / grd
            next_state = {}
            q_val = reward
            self.reward_mem.append(reward)
            for i in reversed(range(len(buffers))):
                if i == len(buffers) - 1:
                    reward = best_IS_util / grd
                else:
                    # reward = 0 # + self.gamma * q_val
                    reward = reward * self.gamma
                state, act_vals, action = buffers[i]
                self.memorize(state, act_vals, action, next_state, reward)
                next_state = state
                q_val = act_vals[action]

        return mwis, best_IS_util


N_bd = FLAGS.feature_size

# use gpu 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(0)

# Initialize session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

# Create model
# dqn_agent = DQNAgent(N_bd, 5000)
