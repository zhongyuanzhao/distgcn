# python3
# Make this standard template for testing and training
from __future__ import division
from __future__ import print_function

import sys
import os
import shutil
sys.path.append( '%s/gcn' % os.path.dirname(os.path.realpath(__file__)) )
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
from gcn.models import GCN_DQN
# import the libary for graph reduction and local search
# from reduce_lib import reducelib
import warnings
warnings.filterwarnings('ignore')

from gcn.utils import *
from runtime_config import flags
# Settings (FLAGS)
from test_utils import *
from heuristics import *

flags.DEFINE_string('test_datapath', './data/ER_Graph_Uniform_NP20_test', 'test dataset')
flags.DEFINE_float('epsilon', 1.0, 'test dataset')
flags.DEFINE_float('epsilon_min', 0.001, 'test dataset')
# test data path
FLAGS = flags.FLAGS
# Some preprocessing
num_supports = 1 + FLAGS.max_degree
model_func = GCN_DQN
nsr = np.power(10.0,-FLAGS.snr_db/20.0)


def add_rnd_q(cns, nIS_vec_local):
    global adj_0

    nIS_vec_local[cns] = 1
    tmp = sp.find(adj_0[cns, :] == 1)
    nIS_vec_local[tmp[1]] = 0
    remain_vec_tmp = (nIS_vec_local == -1)
    adj = adj_0
    adj = adj[remain_vec_tmp, :]
    adj = adj[:, remain_vec_tmp]
    if reduce_graph(adj, nIS_vec_local):
        return True
    return False

def reduce_graph(adj, wts, nIS_vec_local):
    global bsf_q

    # global wts

    remain_vec = (nIS_vec_local == -1)

    # reduce graph
    # reduced_node, reduced_adj, mapping, reverse_mapping, crt_is_size = api.reduce_graph(adj)
    reduced_node, reduced_adj, mapping, reverse_mapping, crt_is_size = fake_reduce_graph(adj)
    nIS_vec_sub = reduced_node.copy()
    nIS_vec_sub_tmp = reduced_node.copy()
    nIS_vec_sub[nIS_vec_sub_tmp == 0] = 1
    nIS_vec_sub[nIS_vec_sub_tmp == 1] = 0
    reduced_nn = reduced_adj.shape[0]

    # update MIS after reduction
    tmp = sp.find(adj[nIS_vec_sub == 1, :] == 1)
    nIS_vec_sub[tmp[1]] = 0
    nIS_vec_local[remain_vec] = nIS_vec_sub
    nIS_vec_local[nIS_vec_local == 2] = -1
    wts_nn = wts[remain_vec]

    bsf_q.append([adj, nIS_vec_local.copy(), remain_vec.copy(), reduced_adj, reverse_mapping.copy(), wts_nn.copy()])

    return False


def weighted_random_graph(N, p, dist, maxWts=1.0):
    graph = nx.generators.random_graphs.fast_gnp_random_graph(N,p)
    if dist.lower() == 'uniform':
        for u in graph:
            graph.nodes[u]['weight'] = np.random.uniform(0,maxWts)
    elif dist.lower() == 'normal_l1':
        for u in graph:
            graph.nodes[u]['weight'] = np.abs(np.random.randn())
    elif dist.lower() == 'normal_l2':
        for u in graph:
            graph.nodes[u]['weight'] = np.square(np.random.randn())

    return graph


class DQNAgent:
    def __init__(self, feature_size=32, memory_size=5000):
        self.feature_size = feature_size
        self.memory = deque(maxlen=memory_size)
        self.reward_mem = deque(maxlen=memory_size)
        self.smallconst = 0.000001 # prevent empty solution
        self.gamma = 0.95    # discount rate
        self.epsilon = FLAGS.epsilon  # exploration rate
        self.epsilon_min = FLAGS.epsilon_min
        self.epsilon_decay = 0.985
        self.learning_rate = FLAGS.learning_rate
        self.sess = tf.compat.v1.Session(config=config)
        self.model = self._build_model()
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.saver = tf.compat.v1.train.Saver(max_to_keep=1000)

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = model_func(placeholders, input_dim=self.feature_size, logging=True)
        return model

    def memorize(self, state, act_vals, solu, wts_nn, reward):
        self.memory.append((state.copy(), act_vals.copy(), solu.copy(), wts_nn.copy(), reward))
        self.reward_mem.append(reward)

    def makestate(self, adj, wts_nn):
        reduced_nn = wts_nn.shape[0]
        norm_wts = np.linalg.norm(wts_nn)
        features = np.multiply(np.ones([reduced_nn, self.feature_size]), wts_nn/norm_wts)
        # features = np.ones([reduced_nn, self.feature_size])
        features = sp.lil_matrix(features)
        features = preprocess_features(features)
        support = simple_polynomials(adj, FLAGS.max_degree)
        state = {"features": features, "support": support}
        return state

    def predict(self, state):
        feed_dict_val = construct_feed_dict4pred(state["features"], state["support"], placeholders)
        act_values, action = self.sess.run([self.model.outputs_softmax, self.model.pred], feed_dict=feed_dict_val)
        return act_values, action

    def act(self, state):
        act_values, action = self.predict(state)
        if np.random.rand() <= self.epsilon:
            return random.randrange(act_values.size)
        return action  # returns action

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return None
        minibatch = random.sample(self.memory, batch_size)
        states, targets_f = [], []
        losses = []
        for state, act_vals, solu, wts_nn, reward in minibatch:
            # target = np.mean(wts_nn.flatten()) * reward
            # target_f = np.ones((act_vals.size, 1)) * (0 - reward)
            target = reward
            # if not done:
            #     act_values, _ = self.predict(next_state)
            #     target = (reward + self.gamma * np.amax(act_values))
            # target_f, _ = self.predict(state)
            target_f = np.reshape(act_vals, (act_vals.size, 1))
            # target_f = np.zeros((act_vals.size, 1))
            # target_f = -np.abs(target_f)
            if FLAGS.predict == 'mwis':
                target_f[solu] = target # * wts_nn #+ wts_nn
            else:
                target_f[solu] = target * wts_nn #+ wts_nn
            # Filtering out states and targets for training
            states.append(state.copy())
            targets_f.append(target_f)

        for i in range(len(states)):
            state = states[i]
            target_f = targets_f[i]
            feed_dict = construct_feed_dict(state['features'], state['support'], target_f, placeholders)
            _, loss = self.sess.run([self.model.opt_op, self.model.loss], feed_dict=feed_dict)
            losses.append(loss)
            # history = self.model.fit(np.array(states), np.array(targets_f), epochs=1, verbose=0)
        # Keeping track of loss
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return np.nanmean(losses)

    def load(self, name):
        ckpt = tf.train.get_checkpoint_state(name)
        if ckpt:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print('loaded ' + ckpt.model_checkpoint_path)

    def save(self, name):
        self.saver.save(self.sess, os.path.join(name, "model.ckpt"))


    def solve_mwis(self, adj_0, wts_0, train=False):
        # buffer = deque(maxlen=20)
        q_ct = 0
        # best_IS_util = np.array([0.0])
        g_tmp = nx.from_scipy_sparse_matrix(adj_0)
        rm_nodes = np.where(wts_0 == 0)[0]
        kp_nodes = np.where(wts_0 > 0)[0]
        g_tmp.remove_nodes_from(rm_nodes)
        adj_0 = nx.adjacency_matrix(g_tmp)
        wts = wts_0[kp_nodes].reshape(len(wts_0[kp_nodes]), 1)
        reduced_nn = adj_0.shape[0]
        reduce_graph(adj_0, wts, -np.ones(reduced_nn))
        # reduce_graph(adj_0, wts/(np.max(wts)+1e-6), -np.ones(reduced_nn))

        q_item = bsf_q.pop(0)
        q_ct += 1

        adj = q_item[0]
        # remain_vec = deepcopy(q_item[2])
        reduced_adj = q_item[3]
        # reverse_mapping = deepcopy(q_item[4])
        # remain_nn = adj.shape[0]
        # reduced_nn = reduced_adj.shape[0]
        wts_nn = q_item[5]

        # GCN
        state = dqn_agent.makestate(reduced_adj, wts_nn)
        act_vals, act = dqn_agent.predict(state)
        if train:
            if np.random.rand() <= dqn_agent.epsilon:
                act_vals = np.random.uniform(size=act_vals.size)

        if FLAGS.predict == 'mwis':
            # gcn_wts = np.divide(wts_nn.flatten(), act_vals.flatten()+1e-8)
            gcn_wts = np.multiply(act_vals.flatten(), wts_nn.flatten())
            # gcn_wts = act_vals.flatten()+100
        else:
            gcn_wts = act_vals.flatten()
        # gcn_wts = np.multiply(act_vals.flatten(), wts_nn.flatten()) + wts_nn.flatten()

        mwis, _ = local_greedy_search(adj, gcn_wts)
        # mwis, _ = greedy_search(adj, gcn_wts)
        solu = list(mwis)
        mwis_rt = set(kp_nodes[solu])
        total_wt = np.sum(wts_nn[solu, 0])
        if train:
            sol_gd, greedy_util = local_greedy_search(adj, wts_nn)
            # wts_norm = wts_nn[list(sol_gd), :]/greedy_util.flatten()
            # dqn_agent.memorize(state.copy(), act_vals.copy(), list(sol_gd), wts_norm, 1.0)
            # reward = (total_wt + self.smallconst) / (greedy_util.flatten()[0] + self.smallconst)
            reward = (total_wt) / (greedy_util.flatten()[0] + 1e-6)
            # reward = reward if reward > 0 else 0
            if FLAGS.predict == 'mwis':
                wts_norm = wts_nn[solu, :]/greedy_util.flatten()
            else:
                wts_norm = wts_nn[solu, :]
            if not np.isnan(reward):
                dqn_agent.memorize(state.copy(), act_vals.copy(), list(mwis), wts_norm, reward)
                # if ((reward > np.mean(self.reward_mem)) or (len(self.reward_mem) < 200)):
                #     dqn_agent.memorize(state.copy(), act_vals.copy(), solu, wts_norm, reward)
                # elif np.random.uniform(0, 1) < 0.01:
                #     dqn_agent.memorize(state.copy(), act_vals.copy(), solu, wts_norm, reward)
            return mwis_rt, total_wt, reward
        return mwis_rt, total_wt, 1.0

    # def solve_mwis_iterative(self, adj_0, wts_0, train=False):
    #     # buffer = deque(maxlen=20)
    #     q_ct = 0
    #     best_IS_util = np.array([0.0])
    #     reduced_nn = adj_0.shape[0]
    #     reduce_graph(adj_0, -np.ones(nn))
    #     while reduced_nn > 0:
    #         if len(bsf_q) == 0:
    #             break
    #         q_item = bsf_q.pop(0)
    #         q_ct += 1
    #
    #         # adj = q_item[0]
    #         remain_vec = deepcopy(q_item[2])
    #         reduced_adj = q_item[3]
    #         # reverse_mapping = deepcopy(q_item[4])
    #         # remain_nn = adj.shape[0]
    #         reduced_nn = reduced_adj.shape[0]
    #         wts_nn = q_item[5]
    #
    #         # GCN
    #         state = dqn_agent.makestate(reduced_adj, wts_nn)
    #         act_vals, act = dqn_agent.predict(state)
    #         # if not test:
    #         #     if np.random.rand() <= dqn_agent.epsilon:
    #         #         act_vals = np.random.uniform(size=act_vals.size)
    #
    #         if FLAGS.predict == 'mwis':
    #             gcn_wts = np.multiply(act_vals.flatten(), wts_nn.flatten())
    #         else:
    #             gcn_wts = act_vals.flatten()
    #
    #         act = np.argmax(gcn_wts)
    #         remain_vtx, = np.where(remain_vec)
    #         cns = remain_vtx[act]
    #
    #         nIS_vec = deepcopy(q_item[1])
    #         nIS_vec[cns] = 1
    #         tmp = sp.find(adj_0[cns, :] == 1)
    #         nIS_vec[tmp[1]] = 0
    #         remain_vec_tmp = (nIS_vec == -1)
    #
    #         adj = adj_0
    #         adj = adj[remain_vec_tmp, :]
    #         adj = adj[:, remain_vec_tmp]
    #         # next_state = dqn_agent.makestate(adj, wts[remain_vec_tmp])
    #         # reward = wts_nn[act, 0]/(greedy_util.flatten()[0])
    #
    #         if np.sum(remain_vec_tmp) == 0:
    #             # get a solution
    #             # nIS_vec = api.local_search(adj_0, nIS_vec)
    #             # nIS_vec = fake_local_search(adj_0, nIS_vec)
    #             best_IS_util = np.dot(nIS_vec, wts)
    #             break
    #
    #         if reduce_graph(adj, nIS_vec):
    #             continue
    #     return best_IS_util, q_ct


N_bd = FLAGS.feature_size

# Define placeholders
placeholders = {
    'support': [tf.compat.v1.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.compat.v1.sparse_placeholder(tf.float32, shape=(None, N_bd)), # featureless: #points
    'labels': tf.compat.v1.placeholder(tf.float32, shape=(None, 1)), # 0: not linked, 1:linked
    'labels_mask': tf.compat.v1.placeholder(tf.int32),
    'dropout': tf.compat.v1.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.compat.v1.placeholder(tf.int32)  # helper variable for sparse dropout
}

# use gpu 0
os.environ['CUDA_VISIBLE_DEVICES']=str(0)

# Initialize session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

bsf_q = []
# Create model
dqn_agent = DQNAgent(N_bd, 5000)
