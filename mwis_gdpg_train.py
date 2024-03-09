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

import tensorflow as tf
from collections import deque
# import the libary for graph reduction and local search
# from reduce_lib import reducelib
import warnings
warnings.filterwarnings('ignore')

from gcn.utils import *
# Settings (FLAGS)
from runtime_config import *
from test_utils import *
from heuristics import *

flags.DEFINE_string('test_datapath', './data/ER_Graph_Uniform_NP20_test', 'test dataset')
# from mwis_dqn_call import dqn_agent
# from mwis_dqn_tree_call import DQNAgent
from mwis_gdpg_call import DQNAgent
# from mwis_mlp_call import DQNAgent
dqn_agent = DQNAgent(FLAGS, 5000)

# test data path
data_path = FLAGS.datapath
test_datapath = FLAGS.test_datapath
val_mat_names = sorted(os.listdir(data_path))
test_mat_names = sorted(os.listdir(test_datapath))

# Some preprocessing
noout = min(FLAGS.diver_num, FLAGS.diver_out) # number of outputs
time_limit = FLAGS.timeout  # time limit for searching
backoff_thresh = 1 - FLAGS.backoff_prob

num_supports = 1 + FLAGS.max_degree
nsr = np.power(10.0, -FLAGS.snr_db/20.0)

from directory import create_result_folder, find_model_folder
# outputfolder = create_result_folder(FLAGS, 'dqn')
model_origin = find_model_folder(FLAGS, 'dqn')

# for graph reduction and local search
# api = reducelib()

# use gpu 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(0)

# Initialize session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

try:
    dqn_agent.load(model_origin)
except:
    print("Unable to load {}".format(model_origin))

best_IS_vec = []
loss_vec = []

epislon_reset = [5, 10, 15, 20]
epislon_val = 1.0

best_ratio = 0.55
for epoch in range(FLAGS.epochs):
    losses = []
    cnt = 0
    f_ct = 0
    q_totals = []
    p_ratios = []
    newtime = time.time()
    for id in np.random.permutation(len(val_mat_names)):
        best_IS_num = -1
        mat_contents = sio.loadmat(data_path + '/' + val_mat_names[id])
        adj_0 = mat_contents['adj']
        wts = mat_contents['weights'].transpose()
        nn = adj_0.shape[0]
        wts = np.random.uniform(0, 1, size=(nn, 1))
        _, greedy_util = greedy_search(adj_0, wts)
        bsf_q = []
        q_ct = 0
        res_ct = 0
        out_id = -1

        start_time = time.time()
        mwis, ss_util = dqn_agent.solve_mwis(adj_0, wts, train=True, grd=greedy_util)
        # mwis, ss_util = dqn_agent.solve_mwis_train(adj_0, wts, train=True, grd=greedy_util)
        # mwis, ss_util = dqn_agent.solve_mwis_cgs_train(adj_0, wts, train=True, grd=greedy_util)
        p_ratio = ss_util.flatten()/greedy_util.flatten()
        solu = list(mwis)
        f_ct += 1
        q_totals.append(len(solu))
        p_ratios.append(p_ratio[0])
        if cnt < 200 - 1:
            cnt += 1
            continue
        else:
            cnt = 0
            runtime = time.time() - newtime
            newtime = time.time()

        avg_is_size = np.mean(q_totals)

        # # q_totals = []
        # for i in range(1):
        #     loss = dqn_agent.replay(2000)
        # print("Epoch: {}".format(epoch), "ID: %05d" % f_ct, "Avg_IS_Size: {:.4f}".format(avg_is_size),
        #       "Epsilon: {:.6f}".format(dqn_agent.epsilon), "Ratio: {:.6f}".format(p_ratio[0]),
        #       "Loss: {:.6f}".format(loss), "Epoch_Loss: {:.6f}".format(np.mean(losses)), "Epoch_Ratio: {:.6f}".format(np.mean(p_ratios)), "runtime: {:.3f}".format(runtime))
        test_ratio=[]
        test_ratio2 = []
        test_len = len(test_mat_names)
        for j in range(test_len):
            mat_contents = sio.loadmat(test_datapath + '/' + test_mat_names[j % test_len])
            adj_0 = mat_contents['adj']
            wts = mat_contents['weights'].transpose()
            nn = adj_0.shape[0]
            if j >= test_len:
                wts = np.random.uniform(0, 1, size=(nn, 1))
            # yy_util = mat_contents['mwis_utility']
            # greedy_util = mat_contents['greedy_utility']
            _, greedy_util = greedy_search(adj_0, wts)
            # _, greedy2_util = local_greedy_search_nstep(adj_0, wts, nstep=2)
            bsf_q = []
            q_ct = 0
            res_ct = 0
            out_id = -1
            _, best_IS_util = dqn_agent.solve_mwis(adj_0, wts, train=False)
            # _, best_IS_util = dqn_agent.solve_mwis_cgs_train(adj_0, wts, train=False)
            # best_IS_util, q_ct = solve_mwis_iterative(test=True)
            # test_ratio.append(best_IS_util / yy_util.flatten()[0])
            test_ratio.append(best_IS_util / greedy_util)
            # test_ratio2.append(greedy2_util/greedy_util)

        if np.mean(test_ratio) > best_ratio:
            dqn_agent.save(model_origin)
            best_ratio = np.mean(test_ratio)

        # loss = dqn_agent.replay(2000)
        loss = dqn_agent.replay(200)
        if loss is None:
            loss = 1.0
        losses.append(loss)

        print("Epoch: {}".format(epoch),
              "ID: %03d" % f_ct,
              "Train_Ratio: {:.6f}".format(np.mean(p_ratios)),
              "Epsilon: {:.6f}".format(dqn_agent.epsilon),
              "Test_Ratio: {:.6f}".format(np.mean(test_ratio)),
              # "Test_Ratio2: {:.6f}".format(np.mean(test_ratio2)),
              "Loss: {:.6f}".format(loss),
              "Epoch_Loss: {:.6f}".format(np.mean(losses)),
              "runtime: {:.3f}".format(runtime),
              "mem_val: {:.3f}".format(np.nanmean(dqn_agent.reward_mem)))
        p_ratios = []

    # dqn_agent.epsilon_min = dqn_agent.epsilon_min * 0.1
    loss_vec.append(np.mean(losses))
    if epoch+1 in epislon_reset:
        epislon_val = epislon_val*0.2
        dqn_agent.epsilon = epislon_val
        # sio.savemat('./%s/%s' % (outputfolder, val_mat_names[id]), {'er_graph': adj_0, 'nIS_vec': best_IS_vec, 'weights': wts, 'best_util': best_IS_util, 'yy_util': yy_util})
print(loss_vec)

