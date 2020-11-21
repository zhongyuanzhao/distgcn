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

import argparse
import time
import random
import scipy.io as sio
import numpy as np
import scipy.sparse as sp
from multiprocessing import Queue
from copy import deepcopy

# import tensorflow as tf
from collections import deque
# from models import GCN_DQN
import pandas as pd
# import the libary for graph reduction and local search
# from reduce_lib import reducelib
import warnings
warnings.filterwarnings('ignore')

from utils import *
# Settings (FLAGS)
# from runtime_config import *
from test_utils import *
from heuristics import *

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--datapath", default="./data/BA_Graph_Uniform_GEN21_test2", type=str, help="directory of test dataset")
parser.add_argument("--solver", default="optimal", type=str, help="MWIS solver: optimal, mp_greedy.")
args = parser.parse_args()

# test data path
# data_path = FLAGS.datapath

solver = args.solver.lower()
data_path = args.datapath
# data_path = './data/BA_Graph_Uniform_GEN21_test2'
# data_path = './data/ER_Graph_Uniform_GEN21_test2'
# test_datapath = FLAGS.test_datapath
val_mat_names = sorted(os.listdir(data_path))
# test_mat_names = sorted(os.listdir(test_datapath))

if solver=='mp_greedy':
    model_origin = "mp_clique_greedy_"+data_path.split('/')[-1]
elif solver=='optimal':
    model_origin = "mlp_gurobi_"+data_path.split('/')[-1]
else:
    raise NameError('Unsupported MWIS solver')

# plp.pulpTestAll()

# use gpu 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(0)

# Initialize session
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True

best_IS_vec = []
loss_vec = []

losses = []
cnt = 0
f_ct = 0
q_totals = []
p_ratios = []
outputfile = "./output/{}.csv".format(model_origin)
if os.path.isfile(outputfile):
    results = pd.read_csv(outputfile, index_col=0)
    # results.set_index('data', inplace=True)
else:
    results = pd.DataFrame([], columns=["data","p","runtime","status"])
    results['data'] = val_mat_names
    results['p'] = 0.0
    results.set_index('data', inplace=True)
newtime = time.time()
cnt2solve = len(val_mat_names)
timeout=4800
while cnt2solve:
    unsolved = results[results['p']==0.0].index.values
    cnt2solve = len(unsolved)
    for mat_name in unsolved:
        best_IS_num = -1
        # print(val_mat_names[id])
        mat_contents = sio.loadmat(data_path + '/' + mat_name)
        adj_0 = mat_contents['adj']
        wts = mat_contents['weights'].transpose()
        yy_util = mat_contents['mwis_utility']
        _, greedy_util = greedy_search(adj_0, wts)
        nn = adj_0.shape[0]
        bsf_q = []
        q_ct = 0
        res_ct = 0
        out_id = -1

        start_time = time.time()
        if solver=='mp_greedy':
            solution, ss_util = mp_greedy(adj_0, wts)
            status = ''
        elif solver=='optimal':
            solution, ss_util, status = mlp_gurobi(adj_0, wts, timeout=timeout)
        else:
            raise NameError('Unsupported solver')

        p_ratio = ss_util.flatten()/greedy_util.flatten()
        if p_ratio[0]==0:
            continue
        f_ct += 1
        # q_totals.append(q_total)
        p_ratios.append(p_ratio[0])
        # avg_is_size = np.mean(q_totals)
        # print("Epoch: {}".format(epoch), "ID: %05d" % f_ct, "Avg_IS_Size: {:.4f}".format(avg_is_size),
        #       "Epsilon: {:.6f}".format(dqn_agent.epsilon), "Ratio: {:.6f}".format(p_ratio[0]),
        #       "Loss: {:.6f}".format(loss), "Epoch_Loss: {:.6f}".format(np.mean(losses)), "Epoch_Ratio: {:.6f}".format(np.mean(p_ratios)), "runtime: {:.3f}".format(runtime))
        test_ratio=[]

        # best_IS_util,_,_ = solve_mwis(test=False)
        # test_ratio.append(best_IS_util[0] / yy_util[0, 0])

        runtime = time.time() - start_time

        print("ID: %03d" % f_ct,
              "File: {}".format(mat_name),
              "Ratio: {:.6f}".format(p_ratio[0]),
              "Avg_Ratio: {:.6f}".format(np.mean(p_ratios)),
              # "Avg_IS_Size: {:.4f}".format(avg_is_size),
              "runtime: {:.3f}".format(runtime))
        # results = results.append(
        #     {"data": val_mat_names[id],
        #      "p": p_ratio[0] ,
        #      "runtime":runtime},
        #     ignore_index=True
        # )
        results.loc[mat_name, 'p'] = p_ratio[0]
        results.loc[mat_name, 'runtime'] = runtime
        results.loc[mat_name, 'status'] = status
        # dqn_agent.save(model_origin)
        results.to_csv(outputfile)
        # sio.savemat('./%s/%s' % (outputfolder, val_mat_names[id]), {'er_graph': adj_0, 'nIS_vec': best_IS_vec, 'weights': wts, 'best_util': best_IS_util, 'yy_util': yy_util})
        cnt2solve -= 1
    timeout = timeout*10
