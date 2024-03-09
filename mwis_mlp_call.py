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
from gcn.models import MLP, MLP2
# import the libary for graph reduction and local search
# from reduce_lib import reducelib
import warnings
warnings.filterwarnings('ignore')

from gcn.utils import *
from runtime_config import flags, FLAGS
# Settings (FLAGS)
from test_utils import *
from heuristics import *
from mwis_base_call import MWISSolver
# flags.DEFINE_float('epsilon', 1.0, 'test dataset')
# flags.DEFINE_float('epsilon_min', 0.001, 'test dataset')
# test data path

# Some preprocessing
model_func = MLP2
nsr = np.power(10.0,-FLAGS.snr_db/20.0)


class DQNAgent(MWISSolver):
    def __init__(self, input_flags, memory_size=5000):
        super(DQNAgent, self).__init__(input_flags, memory_size)
        self.flags = input_flags
        self.num_supports = 1 + self.flags.max_degree
        # self.placeholders = {
        #     'support': [tf.compat.v1.sparse_placeholder(tf.float32) for _ in range(self.num_supports)],
        #     'features': tf.compat.v1.sparse_placeholder(tf.float32, shape=(None, self.feature_size)), # featureless: #points
        #     'adj': tf.compat.v1.sparse_placeholder(tf.float32),
        #     'labels': tf.compat.v1.placeholder(tf.float32, shape=(None, self.flags.diver_num)), # 0: not linked, 1:linked
        #     'labels_mask': tf.compat.v1.placeholder(tf.int32),
        #     'actions': tf.compat.v1.placeholder(tf.float32, shape=(None, 1)),  # real actions, including exploration
        #     'dropout': tf.compat.v1.placeholder_with_default(0., shape=()),
        #     'num_features_nonzero': tf.compat.v1.placeholder(tf.int32)  # helper variable for sparse dropout
        # }# Define placeholders
        self.sess = tf.compat.v1.Session(config=config)
        self.model = self._build_model()
        with self.sess.as_default():
            self.sess.run(tf.compat.v1.global_variables_initializer())
        self.saver = tf.compat.v1.train.Saver(max_to_keep=1000)

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = model_func(self.placeholders, hidden_dim=self.flags.hidden1, bias=True, logging=True)
        return model

    def makestate(self, adj, wts_nn):
        reduced_nn = wts_nn.shape[0]
        norm_wts = np.amax(wts_nn)
        # features = np.multiply(np.ones([reduced_nn, self.feature_size]), wts_nn/norm_wts)
        degrees = np.asarray(adj.sum(axis=1).astype(float)).flatten()
        features = np.ones([reduced_nn, self.feature_size])
        features[:, 0] = degrees
        features = sp.lil_matrix(features)
        features = sparse_to_tuple(features)
        support = simple_polynomials(adj, self.flags.max_degree)
        state = {"features": features, "support": support}
        return state

    def predict(self, state):
        feed_dict_val = construct_feed_dict4pred(state["features"], state["support"], self.placeholders)
        with self.sess.as_default():
            act_values, action = self.sess.run([self.model.outputs, self.model.pred], feed_dict=feed_dict_val)
        return act_values, action

    def act(self, state, train):
        act_values, action = self.predict(state)
        if train:
            if np.random.rand() <= self.epsilon:
                act_values = np.random.uniform(0, 1, size=act_values.shape)
                # act_values[act_values<0] = 0
                action = np.argmax(act_values)
        return act_values, action  # returns action

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return None
        minibatch = random.sample(self.memory, batch_size)
        states, targets_f = [], []
        losses = []
        batch_avg = 0
        batch_len = 0
        batch_tgt = np.array([])
        for state, act_vals, solu, wts_nn, reward in minibatch:
            # target = np.mean(wts_nn.flatten()) * reward
            # target_f = np.ones((act_vals.size, 1)) * (0 - reward)
            target = reward
            # if not done:
            #     act_values, _ = self.predict(next_state)
            #     target = (reward + self.gamma * np.amax(act_values))
            # target_f, _ = self.predict(state)
            target_f = act_vals
            # target_f = np.zeros((act_vals.size, 1))
            # target_f = -np.abs(target_f)
            if FLAGS.predict == 'mwis':
                target_f[solu,:] = target # * wts_nn #+ wts_nn
            else:
                target_f[solu,:] = target * wts_nn[solu,0:1]  #+ wts_nn
            # m2 = np.mean(target_f)
            # target_f = target_f /np.mean(target_f)
            # Filtering out states and targets for training
            states.append(state.copy())
            targets_f.append(target_f)
            batch_tgt = np.append(batch_tgt, target_f)

        batch_tgt = batch_tgt.flatten()
        batch_avg = np.mean(batch_tgt)
        batch_std = np.std(batch_tgt)
        with self.sess.as_default():
            for i in range(len(states)):
                state = states[i]
                # target_f = (targets_f[i]-batch_avg)/batch_std + 1
                target_f = targets_f[i] #/ batch_avg
                feed_dict = construct_feed_dict(state['features'], state['support'], target_f, self.placeholders)
                _, loss = self.sess.run([self.model.opt_op, self.model.loss], feed_dict=feed_dict)
                losses.append(loss)
            # history = self.model.fit(np.array(states), np.array(targets_f), epochs=1, verbose=0)
        # Keeping track of loss
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return np.nanmean(losses)

    def solve_mwis_train(self, adj_0, wts_0, train=False, grd=1.0):
        """
        GCN followed by LGS
        """
        adj = adj_0.copy()
        wts_nn = np.reshape(wts_0, (wts_0.shape[0], FLAGS.feature_size))

        # GCN
        state = self.makestate(adj, wts_nn)
        act_vals, act = self.act(state, train)
        # np.savetxt("viz_act_vals.csv", act_vals, delimiter=",")

        if FLAGS.predict == 'mwis':
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
            wts_norm = wts_nn/np.amax(wts_nn)
            if not np.isnan(reward):
                self.memorize(state.copy(), act_vals.copy(), list(mwis).copy(), wts_norm.copy(), reward)
            return mwis_rt, total_wt
        return mwis_rt, total_wt

# use gpu 0
os.environ['CUDA_VISIBLE_DEVICES']=str(0)

# Initialize session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

bsf_q = []
# Create model
