#!/usr/bin/ python3
# -*- coding: utf-8 -*-
# python3
# Make this standard template for testing and training
import networkx as nx
# from networkx.algorithms.approximation import independent_set
import numpy as np
import pandas as pd
import scipy.io as sio
import time
from collections import deque
from copy import deepcopy
from scipy.io import savemat
from scipy.spatial import distance_matrix
import dwave_networkx as dnx
import os
from itertools import chain, combinations
from heuristics import greedy_search, dist_greedy_search, local_greedy_search, mlp_gurobi
# visualization
import matplotlib.pyplot as plt
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from graph_util import *

from runtime_config import flags
flags.DEFINE_string('output', 'wireless', 'output folder')
flags.DEFINE_string('test_datapath', './data/ER_Graph_Uniform_NP20_test', 'test dataset')
flags.DEFINE_string('wt_sel', 'qr', 'qr: queue length * rate, q/r: q/r, q: queue length only, otherwise: random')
flags.DEFINE_float('load_min', 0.1, 'traffic load min')
flags.DEFINE_float('load_max', 1.0, 'traffic load max')
flags.DEFINE_float('load_step', 0.1, 'traffic load step')
flags.DEFINE_integer('instances', 10, 'number of layers.')
flags.DEFINE_integer('num_channels', 1, 'number of channels')
flags.DEFINE_integer('opt', 0, 'number of channels')

from mwis_dqn_call import dqn_agent
# from mwis_dqn_tree_call import DQNAgent
# dqn_agent = DQNAgent(flags.FLAGS, 5000)

from directory import create_result_folder, find_model_folder
model_origin = find_model_folder(flags.FLAGS, 'dqn')
# model_origin = find_model_folder(flags.FLAGS, 'dqn_crs')

dqn_agent.load(model_origin)

n_instances = flags.FLAGS.instances

def emv(samples, pemv, n=3):
    assert samples.size == pemv.size
    k = float(2/(n+1))
    return samples * k + pemv * (1-k)




train = False
n_networks = 500
# n_instances = 10
timeslots = 200
if train:
    # algolist = ['GCNr-Dist']
    algolist = ['Greedy', 'GCNr-Dist']
else:
    algolist = ['Greedy', 'GCNr-Dist', 'Benchmark']
    # algolist = ['Greedy', 'GCNr-Dist']
    if flags.FLAGS.opt == 0:
        algoname = 'GCNr-Dist'
        algolist = ['Greedy', algoname]
    elif flags.FLAGS.opt == 1:
        # algolist = ['Greedy', 'DGCN-LGS-it', 'Benchmark']
        algoname = 'DGCN-LGS-it'
        algolist = ['Greedy', algoname]
    elif flags.FLAGS.opt == 2:
        algoname = 'DGCN-RS'
        algolist = ['Greedy', algoname]
    elif flags.FLAGS.opt == 3:
        algoname = 'CGCN-CGS'
        algolist = ['Greedy', algoname]
    # algolist = [algoname]
sim_area = 250
sim_node = 100
sim_rc = 1
sim_ri = 4
n_ch = 1
p_overlap = 0.8
# link rate high and low bound (number of packets per time slot)
sim_rate_hi = 100
sim_rate_lo = 0
# Testing load range (upper limit = 1/(average degree of conflict graphs))
# 10.78 for 10 graphs, 10.56 for 20 graphs
load_min = flags.FLAGS.load_min
load_max = flags.FLAGS.load_max
load_step = flags.FLAGS.load_step
wt_sel = flags.FLAGS.wt_sel


output_dir = flags.FLAGS.output
output_csv = os.path.join(output_dir,
                          'metric_vs_load_summary_{}-channel_utility-{}_opt-{}_load-{:.1f}-{:.1f}_flood.csv'
                          .format(n_ch, wt_sel, flags.FLAGS.opt, load_min, load_max)
                          )

res_list = []
res_df = pd.DataFrame(columns=['graph', 'seed', 'load', 'name', 'avg_queue_len', 'med_queue_len', 'avg_utility', 'avg_degree'])
if os.path.isfile(output_csv):
    res_df = pd.read_csv(output_csv, index_col=0)

d_array = np.zeros((n_networks,), dtype=np.float)

if train:
    datapath = flags.FLAGS.datapath
    epochs = flags.FLAGS.epochs
else:
    datapath = flags.FLAGS.test_datapath
    epochs = 1

val_mat_names = sorted(os.listdir(datapath))

cnt = 0

print("Average degree of all conflict graphs: {}".format(np.mean(d_array)))

np.random.seed()
if train:
    loss = 1.0
else:
    loss = np.nan

wts_sample_file = os.path.join(output_dir, 'samples.txt')

load_array = np.round(np.arange(load_min, load_max+load_step, load_step), 2)
# load = load_array[np.random.randint(0, len(load_array) - 1)]
load = 0.85

buffer = deque(maxlen=20)
for idx in range(0, len(val_mat_names)):
    mat_contents = sio.loadmat(os.path.join(datapath, val_mat_names[idx]))
    gdict = mat_contents['gdict'][0, 0]
    seed = mat_contents['random_seed'][0, 0]
    graph_c, graph_i = poisson_graphs_from_dict(gdict)
    adj_gK = nx.adjacency_matrix(graph_i)

    flows = [e for e in graph_c.edges]
    # flows_r = [(e[1], e[0]) for e in graph_c.edges]
    # flows = flows + flows_r
    nflows = len(flows)
    netcfg = "Config: s {}, n {}, f {}, t {}".format(seed, sim_node, nflows, timeslots)

    d_list = []
    for v in graph_i:
        d_list.append(graph_i.degree[v])
    avg_degree = np.nanmean(d_list)

    i = 0
    for i in range(1, n_instances+1):
    # for load in load_array:
        # treeseed = int(1000 * time.time()) % 10000000
        treeseed = i
        np.random.seed(treeseed)
        skip = 0
        for index, row in res_df.iterrows():
            if row["graph"] == seed and row["seed"] == treeseed:
                skip = 1
                break
        if skip:
            continue
        arrival_rate = 0.5 * (sim_rate_lo + sim_rate_hi) * load

        interarrivals = np.random.exponential(1.0/arrival_rate, (nflows, int(2*timeslots*arrival_rate)))
        arrival_time = np.cumsum(interarrivals, axis=1)
        acc_pkts = np.zeros(shape=(nflows, timeslots))
        for t in range(0, timeslots):
            acc_pkts[:, t] = np.count_nonzero(arrival_time < t, axis=1)
        # arrival_pkts = np.zeros(shape=(nflows, timeslots))
        arrival_pkts = np.diff(acc_pkts, prepend=0)
        arrival_pkts = arrival_pkts.transpose()
        # link_rates = np.random.randint(sim_rate_lo, sim_rate_hi, [timeslots, nflows, n_ch])
        link_rates = np.random.normal(0.5 * (sim_rate_lo + sim_rate_hi), 0.25 * (sim_rate_hi - sim_rate_lo),
                                      size=[timeslots, nflows, n_ch])
        link_rates = link_rates.astype(int)
        link_rates[link_rates < sim_rate_lo] = sim_rate_lo
        link_rates[link_rates > sim_rate_hi] = sim_rate_hi

        # adj = nx.adjacency_matrix(graph_i)
        # adjnp = (nx.to_numpy_matrix(graph_i)>0).tolist()
        # mis_all = get_all_mis(adjnp)

        to_print = []
        time_start = time.time()

        weight_samples = []
        queue_mtx_dict = {}
        dep_pkts_dict = {}
        util_mtx_dict = {}
        wts_dict = {}
        for algo in algolist:
            queue_mtx_dict[algo] = np.zeros(shape=(timeslots, nflows))
            dep_pkts_dict[algo] = np.zeros(shape=(timeslots, nflows))
            util_mtx_dict[algo] = np.zeros(timeslots)
            util_mtx_dict[algo][0] = 1
            wts_dict[algo] = np.zeros(shape=(nflows, n_ch))

        for t in range(1, timeslots):
            for algo in algolist:
                queue_mtx_dict[algo][t, :] = queue_mtx_dict[algo][t-1, :] + arrival_pkts[t, :]
                queue_mtx_algo = np.multiply(np.expand_dims(queue_mtx_dict[algo][t, :], axis=1), np.ones(shape=(nflows, n_ch)))
                if wt_sel == 'qr':
                    wts0 = queue_mtx_algo * link_rates[t, :, :]
                elif wt_sel == 'q':
                    wts0 = queue_mtx_algo
                elif wt_sel == 'qor':
                    wts0 = queue_mtx_algo / link_rates[t, :, :]
                elif wt_sel == 'qrm':
                    wts0 = np.minimum(queue_mtx_algo, link_rates[t, :, :])
                else:
                    np.random.seed(i*1000+t)
                    wts0 = np.random.uniform(0, 1, (nflows, n_ch))
                wts1 = np.reshape(wts0, nflows * n_ch, order='F')

                if algo == "Greedy":
                    wts_dict[algo] = wts1
                    mwis, total_wt = local_greedy_search(adj_gK, wts_dict[algo])
                    # mwis2, total_wt2, reward = dqn_agent.solve_mwis(adj, wts_dict[algo], train=False)
                    mwis0, total_wt0, _ = mlp_gurobi(adj_gK, wts_dict[algo])
                    # mwis0, total_wt0 = greedy_search(adj_gK, wts1)
                    util_mtx_dict[algo][t] = total_wt/total_wt0
                elif algo == "Greedy-Th":
                    # wts = emv(wts0, wts)
                    wts_dict[algo] = wts1
                    mwis, total_wt = dist_greedy_search(adj_gK, wts_dict[algo], 0.1)
                    # mwis0, total_wt0 = greedy_search(adj_gK, wts1)
                    mwis0, total_wt0, _ = mlp_gurobi(adj_gK, wts_dict[algo])
                    util_mtx_dict[algo][t] = total_wt/total_wt0
                elif algo == 'Benchmark':
                    wts_dict[algo] = wts1
                    mwis, total_wt, _ = mlp_gurobi(adj_gK, wts_dict[algo])
                    # mwis, total_wt = greedy_search(adj_gK, wts1)
                    util_mtx_dict[algo][t] = 1.0
                elif algo == 'DGCN-LGS-it':
                    wts_dict[algo] = wts1
                    mwis0, total_wt0, _ = mlp_gurobi(adj_gK, wts1)
                    mwis, total_wt = dqn_agent.solve_mwis_dit(adj_gK, wts_dict[algo], train=train, grd=total_wt0)
                    util_mtx_dict[algo][t] = total_wt / total_wt0
                elif algo == 'DGCN-RS':
                    wts_dict[algo] = wts1
                    mwis0, total_wt0, _ = mlp_gurobi(adj_gK, wts1)
                    mwis, total_wt = dqn_agent.solve_mwis_rollout_wrap(adj_gK, wts_dict[algo], train=train,
                                                                       grd=total_wt0)
                    util_mtx_dict[algo][t] = total_wt / total_wt0
                elif algo == 'CGCN-CGS':
                    wts_dict[algo] = wts1
                    mwis0, total_wt0, _ = mlp_gurobi(adj_gK, wts1)
                    mwis, total_wt = dqn_agent.solve_mwis_cgs_train(adj_gK, wts_dict[algo], train=train,
                                                                       grd=total_wt0)
                    util_mtx_dict[algo][t] = total_wt / total_wt0
                else:
                    # weight_samples += list(wts)
                    # wts0 = queue_mtx_dict[algo][:, t] + link_rates[:, t]
                    # wts0 = np.minimum(queue_mtx_dict[algo][:, t], link_rates[:, t])**1.5
                    # wts = wts0**1.7
                    wts_dict[algo] = wts1
                    # wts = emv(wts0, wts)
                    # wts_dict[algo] = emv(wts0, wts_dict[algo])
                    # mwis0, total_wt0 = local_greedy_search(adj, wts_dict[algo])
                    # mwis0, total_wt0 = greedy_search(adj_gK, wts1)
                    mwis0, total_wt0,_ = mlp_gurobi(adj_gK, wts1)
                    mwis, total_wt = dqn_agent.solve_mwis(adj_gK, wts_dict[algo], train=train, grd=total_wt0)
                    # mwis, total_wt, reward = dqn_agent.solve_mwis(adj, wts, train=train)
                    util_mtx_dict[algo][t] = total_wt/total_wt0

                schedule_mv = np.array(list(mwis))
                link_rates_ts = np.reshape(link_rates[t, :, :], nflows*n_ch, order='F')
                link_rates_sh = link_rates_ts[schedule_mv]
                schedule = schedule_mv % nflows

                capacity = np.zeros(shape=(nflows,))
                capacity[schedule] = link_rates_sh
                dep_pkts_dict[algo][t, :] = np.minimum(queue_mtx_algo[:, 0], capacity)
                queue_mtx_dict[algo][t, :] = queue_mtx_dict[algo][t, :] - dep_pkts_dict[algo][t, :]
                # queue_mtx[queue_mtx[:, t] < 0, t] = 0
                # wts = 1/(queue_mtx[:, t-1] + 1)

        avg_q_dict = {}
        med_q_dict = {}
        for algo in algolist:
            avg_queue_length_ts = np.mean(queue_mtx_dict[algo], axis=1)
            med_queue_length_ts = np.median(queue_mtx_dict[algo], axis=1)
            avg_queue_len_links = np.mean(queue_mtx_dict[algo], axis=0)
            # pct_queue_len = np.percentile(queue_mtx.flatten(), 90)
            # pct_tpt = np.percentile(dep_pkts.flatten(), 90)
            # avg_tpt = np.mean(dep_pkts.flatten())
            avg_q_dict[algo] = np.mean(avg_queue_length_ts)
            med_q_dict[algo] = np.mean(med_queue_length_ts)
            # avg_q_dict[algo] = np.mean(avg_queue_length_ts)
            std_flow_q = np.std(avg_queue_len_links)

            # res_list.append(   {"name": algo,
            #                     "queue": np.transpose(queue_mtx_dict[algo]),
            #                     "depart": np.transpose(dep_pkts_dict[algo]),
            #                     "seed": seed,
            #                     "load": load,
            #                     "avg_queue": avg_q_dict[algo],
            #                     "std_flow_q": std_flow_q,
            #                     "utility": np.transpose(util_mtx_dict[algo])
            #                     })
            res_df = res_df.append({'graph': seed,
                                    'seed': treeseed,
                                    'load': load,
                                    'name': algo,
                                    'avg_queue_len': avg_q_dict[algo],
                                    'med_queue_len': med_q_dict[algo],
                                    'avg_utility': np.nanmean(util_mtx_dict[algo]),
                                    'avg_degree': avg_degree
                                    }, ignore_index=True)
        res_df.to_csv(output_csv)
        # with open(wts_sample_file,'a') as f:
        #     f.write('{}'.format(weight_samples))
        if train:
            loss = dqn_agent.replay(199)
            if loss is None:
                loss = 1.0
            if not np.isnan(loss):
                dqn_agent.save(model_origin)
        # else:
        #     dqn_agent.load(model_origin)
        #

        runtime = time.time() - time_start
        if train:
            # buffer.append(avg_util/greedy_avg_u)
            if wt_sel=='random':
                buffer.append(np.mean(util_mtx_dict[algoname]))
            else:
                buffer.append(avg_q_dict[algoname]/avg_q_dict['Greedy'])
            print("{}-{}: {}, load: {}, ".format(idx, i, netcfg, load),
                "q_median: {:.3f}, ".format(med_q_dict[algoname]/med_q_dict['Greedy']),
                "q_mean: {:.3f}, ".format(avg_q_dict[algoname]/avg_q_dict['Greedy']),
                # "q_median: {:.3f}, ".format(med_q_dict[algoname]/med_q_dict['Benchmark']),
                # "q_mean: {:.3f}, ".format(avg_q_dict[algoname]/avg_q_dict['Benchmark']),
                # "q_pct: {:.3f}, ".format(pct_queue_len/greedy_pct_q),
                # "t_avg: {:.3f}, ".format(avg_tpt/greedy_avg_t),
                "u_gdy: {:.3f}, ".format(np.mean(util_mtx_dict['Greedy'])),
                "u_gcn: {:.3f}, ".format(np.mean(util_mtx_dict[algoname])),
                "run: {:.3f}s, loss: {:.5f}, ratio: {:.3f}, ".format(runtime, loss, np.mean(buffer)),
                "e: {:.4f}, m_val: {:.4f}, m_len: {}".format(dqn_agent.epsilon, np.mean(dqn_agent.reward_mem), len(dqn_agent.reward_mem))
                )
        else:
            print("{}: {}, load: {}, ".format(i, netcfg, load),
                "q_median: {:.3f}, ".format(med_q_dict[algoname]/med_q_dict['Greedy']),
                "q_mean: {:.3f}, ".format(avg_q_dict[algoname]/avg_q_dict['Greedy']),
                # "q_median: {:.3f}, ".format(med_q_dict[algoname]/med_q_dict['Benchmark']),
                # "q_mean: {:.3f}, ".format(avg_q_dict[algoname]/avg_q_dict['Benchmark']),
                # "q_pct: {:.3f}, ".format(pct_queue_len/greedy_pct_q),
                # "t_avg: {:.3f}, ".format(avg_tpt/greedy_avg_t),
                "u_gdy: {:.3f}, ".format(np.mean(util_mtx_dict['Greedy'])),
                "u_gcn: {:.3f}, ".format(np.mean(util_mtx_dict[algoname])),
                "run: {:.3f}".format(runtime)
                # "runtime: {:.3f}, loss: {:.5f}, ratio: {:.3f}, ".format(runtime, loss, np.mean(buffer)),
                # "epsilon: {:.4f}, mem_val: {:.4f}, mem_len: {}".format(dqn_agent.epsilon, np.mean(dqn_agent.reward_mem), len(dqn_agent.reward_mem))
                )
        # print("{}, load: {}, avg_queue: {:.3f}, greedy_q_avg: {:.3f}, avg_util: {:.3f}, runtime: {:.3f}"
        #     .format(netcfg, load, avg_q_dict[algoname], avg_q_dict['Greedy'], np.mean(util_mtx_dict[algoname]), runtime))
        # i += 1


    # np.savetxt('./instance_209_q_gdy.csv', res_list[explen * i]['queue'], delimiter=',')
    # np.savetxt('./instance_209_q_gcn.csv', res_list[explen * i + 1]['queue'], delimiter=',')
    # np.savetxt('./instance_209_d_gdy.csv', res_list[explen * i]['depart'], delimiter=',')
    # np.savetxt('./instance_209_d_gcn.csv', res_list[explen * i + 1]['depart'], delimiter=',')
    # np.savetxt('./instance_209_r_gdy.csv', res_list[explen * i]['utility'], delimiter=',')
    # np.savetxt('./instance_209_r_gcn.csv', res_list[explen * i + 1]['utility'], delimiter=',')
    # if not train:
    #     np.savetxt('./instance_209_d_bmk.csv', res_list[explen * i + 2]['depart'], delimiter=',')
    #     np.savetxt('./instance_209_r_bmk.csv', res_list[explen * i + 2]['utility'], delimiter=',')
    #     np.savetxt('./instance_209_q_bmk.csv', res_list[explen * i + 2]['queue'], delimiter=',')
    # savemat('./wireless/metric_vs_load_full.mat', {'data': res_list})
# with open('./wireless/metric_vs_load_full.json', 'w') as fout:
#     json.dump(res_list, fout)

print("Done!")

