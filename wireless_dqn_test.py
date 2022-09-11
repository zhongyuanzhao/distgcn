#!/usr/bin/ python3
# -*- coding: utf-8 -*-
# python3
# Make this standard template for testing and training
import networkx as nx
# from networkx.algorithms.approximation import independent_set
import numpy as np
import pandas as pd
import time
from collections import deque
from copy import deepcopy
from scipy.io import savemat
from scipy.spatial import distance_matrix
import dwave_networkx as dnx
import os
from itertools import chain, combinations
from heuristics import greedy_search, dist_greedy_search, local_greedy_search, get_all_mis, get_mwis

# visualization
import matplotlib.pyplot as plt
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

from runtime_config import flags

flags.DEFINE_string('output', 'wireless', 'output folder')
flags.DEFINE_integer('seed', -1, '-1 no seed, other: set numpy random seed')
flags.DEFINE_string('wt_sel', 'qr', 'qr: queue length * rate, q/r: q/r, q: queue length only, otherwise: random')

from mwis_dqn_call import dqn_agent
from directory import create_result_folder, find_model_folder
model_origin = find_model_folder(flags.FLAGS, 'dqn')

# modelorigin = './result_IS4SAT_deep_ld1_c1_l1_cheb1_diver1_mwis_dqn'
dqn_agent.load(model_origin)

def emv(samples, pemv, n=3):
    assert samples.size == pemv.size
    k = float(2/(n+1))
    return samples * k + pemv * (1-k)


# Create a Piosson Point process 2D graph
def poisson_graphs(area, N, rc=1.0, ri=2.0):
    # N = np.random.poisson(lam=area*density)
    lenth_a = np.sqrt(area)
    xys = np.random.uniform(0, lenth_a, (N, 2))
    d_mtx = distance_matrix(xys, xys)

    # generate connectivity graph
    adj_c = np.zeros([N,N], dtype=int)
    adj_c[d_mtx <= rc] = 1
    np.fill_diagonal(adj_c, 0)
    graph_c = nx.from_numpy_matrix(adj_c)
    for u in graph_c:
        graph_c.nodes[u]['xy'] = xys[u, :]

    # generate interference graph
    adj_i = np.zeros([N,N], dtype=int)
    adj_i[d_mtx <= ri] = 1
    np.fill_diagonal(adj_i, 0)
    # graph_i = nx.from_numpy_matrix(adj_i)

    # generate conflict graph
    graph_cf = nx.Graph()
    i = 0
    for e in graph_c.edges:
        graph_cf.add_node(i, weight=1.0, name=e)
        i += 1
    for u in graph_cf:
        nu = graph_cf.nodes[u]['name']
        for v in graph_cf:
            if u == v:
                continue
            nv = graph_cf.nodes[v]['name']
            if adj_i[nu[0], nv[0]] + adj_i[nu[0], nv[1]] + adj_i[nu[1], nv[0]] + adj_i[nu[1], nv[1]]:
                graph_cf.add_edge(u, v)

    return graph_c, graph_cf

def visual_single(results, timeslots, seed, explist):
    fig1 = plt.figure(figsize=(10, 3))
    avg_q_ts = np.zeros(shape=(timeslots, 2))
    avg_q_ts[:, 0] = results[0]["avg_q_ts"]
    avg_q_ts[:, 1] = results[1]["avg_q_ts"]
    plt.plot(avg_q_ts, alpha=0.7)
    plt.xlabel('time slot')
    plt.ylabel('Avg queue length across network')
    plt.grid(True)
    plt.legend(explist)
    # plt.show()
    plt.savefig(os.path.join(flags.FLAGS.output, "link_vs_timeslot_s{}_prod.png".format(seed)))
    fig2 = plt.figure()
    avg_q_lk = np.zeros(shape=(nflows, 2))
    avg_q_lk[:, 0] = results[0]["avg_q_lk"]
    avg_q_lk[:, 1] = results[1]["avg_q_lk"]
    plt.boxplot(avg_q_lk)
    plt.grid(True)
    plt.xticks([1, 2], explist)
    plt.savefig(os.path.join(flags.FLAGS.output, "std_link_avg_queue_length_s{}_prod.png".format(seed)))
    return True


train = False
n_networks = 10
n_instances = 10
timeslots = 200
if train:
    # explist = ['GCNr-Dist']
    explist = ['Greedy', 'GCNr-Dist']
else:
    # explist = ['Greedy', 'Greedy-Th', 'GCNr-Dist']
    # explist = ['Greedy', 'GCNr-Dist']
    explist = ['Greedy', 'GCNr-Dist', 'Benchmark']
sim_area = 250
sim_node = 100
sim_rc = 1
sim_ri = 4
# link rate high and low bound (number of packets per time slot)
sim_rate_hi = 100
sim_rate_lo = 0
# Testing load range (upper limit = 1/(average degree of conflict graphs))
# 10.78 for 10 graphs, 10.56 for 20 graphs
load_min = 0.05
load_max = 0.15

res_list = []
res_df = pd.DataFrame(columns=['seed','load','name','avg_queue_len','std_flow_q'])
d_array = np.zeros((2*n_networks,), dtype=np.float)

graphs_c = []
graphs_i = []
mis_alls = []
if train:
    net_instances = range(100, n_networks+100)
else:
    net_instances = range(1, 2*n_networks+1)
# net_instances = range(100, n_networks+100)

cnt = 0
for seed in net_instances:
    # seed = flags.FLAGS.seed
    if not seed == -1:
        np.random.seed(seed)
        # print("Random seed: {}".format(seed))
    wt_sel = flags.FLAGS.wt_sel

    graph_c, graph_i = poisson_graphs(sim_area, sim_node, rc=sim_rc, ri=sim_ri)
    d_list = []
    for v in graph_i:
        d_list.append(graph_i.degree[v])
    adj = nx.adjacency_matrix(graph_i)
    adjnp = (nx.to_numpy_matrix(graph_i) > 0).tolist()
    mis_all = get_all_mis(adjnp)
    if mis_all.shape[1] > 200000:
        continue
    d_array[cnt] = np.mean(d_list)
    graphs_c.append(graph_c)
    graphs_i.append(graph_i)
    mis_alls.append(mis_all)
    cnt += 1
    if cnt >= 100:
        break

print("Average degree of all {} conflict graphs: {}".format(cnt, np.mean(d_array)))

np.random.seed()
if train:
    loss = 1.0
else:
    loss = np.nan

wts_sample_file = os.path.join('./wireless/', 'samples.txt')

explen = len(explist)

buffer = deque(maxlen=20)
for idx in range(0, len(graphs_c)):
    for i in range(1, 11):
        np.random.seed(i)

        wt_sel = flags.FLAGS.wt_sel
        graph_c = graphs_c[idx]
        graph_i = graphs_i[idx]
        seed = net_instances[idx]

        flows = [e for e in graph_c.edges]
        # flows_r = [(e[1], e[0]) for e in graph_c.edges]
        # flows = flows + flows_r
        nflows = len(flows)
        netcfg = "Config: s {}, n {}, f {}, t {}".format(seed, sim_node, nflows, timeslots)

        load_array = np.round(np.arange(load_min, load_max, 0.01), 2)
        # load = load_array[np.random.randint(0, len(load_array)-1)]
        load = 0.85
        arrival_rate = 0.5 * (sim_rate_lo + sim_rate_hi) * load

        interarrivals = np.random.exponential(1.0/arrival_rate, (nflows, int(2*timeslots*arrival_rate)))
        arrival_time = np.cumsum(interarrivals, axis=1)
        acc_pkts = np.zeros(shape=(nflows, timeslots))
        for t in range(0, timeslots):
            acc_pkts[:, t] = np.count_nonzero(arrival_time < t, axis=1)
        arrival_pkts = np.diff(acc_pkts, prepend=0)
        link_rates = np.random.randint(sim_rate_lo, sim_rate_hi, [nflows, timeslots])

        adj = nx.adjacency_matrix(graph_i)
        mis_all = mis_alls[idx]

        to_print = []
        time_start = time.time()

        weight_samples = []
        queue_mtx_dict = {}
        dep_pkts_dict = {}
        util_mtx_dict = {}
        wts_dict = {}
        for expname in explist:
            queue_mtx_dict[expname] = np.zeros(shape=(nflows, timeslots))
            dep_pkts_dict[expname] = np.zeros(shape=(nflows, timeslots))
            util_mtx_dict[expname] = np.zeros(timeslots)
            util_mtx_dict[expname][0] = 1
            wts_dict[expname] = np.zeros(link_rates[:, 0].shape)

        for t in range(1, timeslots):
            for expname in explist:
                queue_mtx_dict[expname][:, t] = queue_mtx_dict[expname][:, t-1] + arrival_pkts[:, t]
                if wt_sel == 'qr':
                    wts0 = queue_mtx_dict[expname][:, t] * link_rates[:, t]
                elif wt_sel == 'q':
                    wts0 = queue_mtx_dict[expname][:, t]**1.7
                elif wt_sel == 'qor':
                    wts0 = queue_mtx_dict[expname][:, t] / link_rates[:, t]
                elif wt_sel == 'qrm':
                    wts0 = np.minimum(queue_mtx_dict[expname][:, t], link_rates[:, t])
                    # wts0 = np.minimum(queue_mtx_dict[expname][:, t], link_rates[:, t])**1.5
                    # wts = wts**1.5
                else:
                    np.random.seed(i*1000+t)
                    wts0 = np.random.uniform(0, 1, nflows)
                # for u in graph_i:
                #     graph_i.nodes[u]['weight'] = wts[u]
                # wts = (wts + 1e-8)/(2*np.mean(wts)+1e-6)
                # wts = wts + 1e-6
                if expname == "Greedy":
                    wts_dict[expname] = wts0
                    mwis, total_wt1 = local_greedy_search(adj, wts_dict[expname])
                    mwis0, total_wt0 = get_mwis(mis_all, wts_dict[expname])
                    util_mtx_dict[expname][t] = total_wt1/total_wt0
                elif expname == "Greedy-Th":
                    # wts = emv(wts0, wts)
                    wts_dict[expname] = wts0
                    mwis, total_wt = dist_greedy_search(adj, wts_dict[expname], 0.1)
                elif expname == 'Benchmark':
                    wts_dict[expname] = wts0
                    mwis, total_wt = get_mwis(mis_all, wts_dict[expname])
                else:
                    # wts_dict[expname] = emv(wts0, wts_dict[expname])
                    wts_dict[expname] = wts0
                    mwis, total_wt2, reward = dqn_agent.solve_mwis(adj, wts_dict[expname], train=False)
                    mwis0, total_wt0 = get_mwis(mis_all, wts_dict[expname])
                    # mwis1, total_wt1 = local_greedy_search(adj, wts_dict[expname])
                    util_mtx_dict[expname][t] = total_wt2/total_wt0


                schedule = list(mwis)
                capacity = np.zeros(shape=(nflows,))
                capacity[schedule] = link_rates[schedule, t]
                dep_pkts_dict[expname][:, t] = np.minimum(queue_mtx_dict[expname][:, t], capacity)
                queue_mtx_dict[expname][:, t] = queue_mtx_dict[expname][:, t] - dep_pkts_dict[expname][:, t]

        if train:
            loss = dqn_agent.replay(199)
            if loss is None:
                loss = 1.0
        avg_q_dict = {}
        med_q_dict = {}
        for expname in explist:
            avg_queue_length_ts = np.mean(queue_mtx_dict[expname], axis=0)
            med_queue_length_ts = np.median(queue_mtx_dict[expname], axis=0)
            avg_queue_len_links = np.mean(queue_mtx_dict[expname], axis=1)
            # pct_queue_len = np.percentile(queue_mtx.flatten(), 90)
            # pct_tpt = np.percentile(dep_pkts.flatten(), 90)
            # avg_tpt = np.mean(dep_pkts.flatten())
            avg_q_dict[expname] = np.mean(avg_queue_length_ts)
            med_q_dict[expname] = np.mean(med_queue_length_ts)
            # avg_q_dict[expname] = np.mean(avg_queue_length_ts)

            # pct_queue_len = np.percentile(queue_mtx.flatten(), 90)
            # pct_tpt = np.percentile(dep_pkts.flatten(), 90)
            # avg_tpt = np.mean(dep_pkts.flatten())
            avg_queue_len = np.mean(avg_queue_length_ts)
            std_flow_q = np.std(avg_queue_len_links)
            # avg_util = np.mean(util_mtx)
            # to_print.append(avg_util)
            if expname == "Greedy":
                greedy_avg_q = avg_queue_len
                # greedy_pct_q = pct_queue_len
                # greedy_pct_t = pct_tpt
                # greedy_avg_t = avg_tpt
                # greedy_avg_u = avg_util
            res_list.append(   {"name": expname,
                                "queue": np.transpose(queue_mtx_dict[expname]),
                                "depart": np.transpose(dep_pkts_dict[expname]),
                                "seed": seed,
                                "load": load,
                                "avg_queue": avg_queue_len,
                                "std_flow_q": std_flow_q,
                                "utility": np.transpose(util_mtx_dict[expname])
                                })
            # res_df = res_df.append({'seed': seed,
            #                         'load': load,
            #                         'name': expname,
            #                         'avg_queue_len': avg_queue_len,
            #                         'std_flow_q': std_flow_q,
            #                         'avg_utility': avg_util
            #                         }, ignore_index=True)
        # with open(wts_sample_file,'a') as f:
        #     f.write('{}'.format(weight_samples))
        if not np.isnan(loss):
            dqn_agent.save(model_origin)
        # else:
        #     dqn_agent.load(model_origin)
        #

        runtime = time.time() - time_start
        if train:
            # buffer.append(avg_util/greedy_avg_u)
            if wt_sel=='random':
                buffer.append(np.mean(util_mtx_dict['GCNr-Dist']))
            else:
                buffer.append(avg_q_dict['GCNr-Dist']/avg_q_dict['Greedy'])
            print("{}-{}: {}, load: {}, ".format(idx, i, netcfg, load),
                "q_median: {:.3f}, ".format(med_q_dict['GCNr-Dist']/med_q_dict['Greedy']),
                "q_mean: {:.3f}, ".format(avg_q_dict['GCNr-Dist']/avg_q_dict['Greedy']),
                # "q_median: {:.3f}, ".format(med_q_dict['GCNr-Dist']/med_q_dict['Benchmark']),
                # "q_mean: {:.3f}, ".format(avg_q_dict['GCNr-Dist']/avg_q_dict['Benchmark']),
                # "q_pct: {:.3f}, ".format(pct_queue_len/greedy_pct_q),
                # "t_avg: {:.3f}, ".format(avg_tpt/greedy_avg_t),
                "u_gdy: {:.3f}, ".format(np.mean(util_mtx_dict['Greedy'])),
                "u_gcn: {:.3f}, ".format(np.mean(util_mtx_dict['GCNr-Dist'])),
                "run: {:.3f}s, loss: {:.5f}, ratio: {:.3f}, ".format(runtime, loss, np.mean(buffer)),
                "e: {:.4f}, m_val: {:.4f}, m_len: {}".format(dqn_agent.epsilon, np.mean(dqn_agent.reward_mem), len(dqn_agent.reward_mem))
                )
        else:
            print("{}: {}, load: {}, ".format(i, netcfg, load),
                "q_median: {:.3f}, ".format(med_q_dict['GCNr-Dist']/med_q_dict['Greedy']),
                "q_mean: {:.3f}, ".format(avg_q_dict['GCNr-Dist']/avg_q_dict['Greedy']),
                "q_median: {:.3f}, ".format(med_q_dict['GCNr-Dist']/med_q_dict['Benchmark']),
                "q_mean: {:.3f}, ".format(avg_q_dict['GCNr-Dist']/avg_q_dict['Benchmark']),
                # "q_pct: {:.3f}, ".format(pct_queue_len/greedy_pct_q),
                # "t_avg: {:.3f}, ".format(avg_tpt/greedy_avg_t),
                "u_gdy: {:.3f}, ".format(np.mean(util_mtx_dict['Greedy'])),
                "u_gcn: {:.3f}, ".format(np.mean(util_mtx_dict['GCNr-Dist'])),
                "d: {:.2f}, ".format(d_array[idx]),
                "run: {:.3f}".format(runtime)
                # "runtime: {:.3f}, loss: {:.5f}, ratio: {:.3f}, ".format(runtime, loss, np.mean(buffer)),
                # "epsilon: {:.4f}, mem_val: {:.4f}, mem_len: {}".format(dqn_agent.epsilon, np.mean(dqn_agent.reward_mem), len(dqn_agent.reward_mem))
                )
        # print("{}, load: {}, avg_queue: {:.3f}, greedy_q_avg: {:.3f}, avg_util: {:.3f}, runtime: {:.3f}"
        #     .format(netcfg, load, avg_q_dict['GCNr-Dist'], avg_q_dict['Greedy'], np.mean(util_mtx_dict['GCNr-Dist']), runtime))

    # res_df.to_csv('./wireless/metric_vs_load_summary.csv')

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

# visual_single(res_list, timeslots, seed, explist)
print("Done!")

