import sys, getopt
import argparse
import networkx as nx
# from networkx.algorithms.approximation import independent_set
import numpy as np
from scipy.io import savemat
from scipy.spatial import distance_matrix
import dwave_networkx as dnx
import os
from itertools import chain, combinations
from heuristics import greedy_search


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--datapath", default="./data/Random_Graph_Nb", type=str, help="output directory.")
parser.add_argument("--dist", default="uniform", type=str, help="weight distribution: uniform, normal_l1, normal_l2.")
parser.add_argument("--nbs", default="10, 20, 40, 80, 100, 120, 150", type=str, help="list of average numbers of neighbors.")
parser.add_argument("--ps", default="", type=str, help="list of densities.")
parser.add_argument("--sizes", default="200, 400, 600, 800, 1000", type=str, help="list of numbers of vertices.")
parser.add_argument("--n", default=100, type=int, help="number of instances per configuration.")
parser.add_argument("--bf", default=False, type=bool, help="if use brute force search.")
parser.add_argument("--type", default='ER', type=str, help="ER graph: ER; Poisson: PPP.")
args = parser.parse_args()

dist = args.dist.lower()
dist_dict = {'uniform': 'uni', 'normal_l1': 'nl1', 'normal_l2': 'nl2'}
size_list = [int(item) for item in args.sizes.split(',')]
nb_list = [float(item) for item in args.nbs.split(',')]
try:
    p_list = [float(item) for item in args.ps.split(',')]
except:
    p_list = []
# datapath = './data/Random_Graph_Nb20'
# datapath = './data/Random_Graph_Nb100'
datapath = args.datapath
if not os.path.isdir(datapath):
    os.mkdir(datapath)

# Create a random graph
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

# Create a Piosson Point process 2D graph
def weighted_poisson_graph(area, density, radius=1.0, dist='uniform', maxWts=1.0):
    N = np.random.poisson(lam=area*density)
    lenth_a = np.sqrt(area)
    xys = np.random.uniform(0, lenth_a, (N, 2))
    d_mtx = distance_matrix(xys, xys)
    adj_mtx = np.zeros([N,N], dtype=int)
    adj_mtx[d_mtx <= radius] = 1
    np.fill_diagonal(adj_mtx, 0)
    graph = nx.from_numpy_matrix(adj_mtx)
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


def weighted_barabasi_albert_graph(N, p, dist, maxWts=1.0):
    graph = nx.generators.random_graphs.barabasi_albert_graph(N, int(np.round(N*p)))
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

# maximum weighted independent set
def mwis_heuristic_1(graph):
    adj_0 = nx.adj_matrix(graph).todense()
    a = -np.array([graph.nodes[u]['weight'] for u in graph.nodes])
    IS = -np.ones(adj_0.shape[0])
    while np.any(IS==-1):
        rem_vector = IS == -1
        adj = adj_0.copy()
        adj = adj[rem_vector, :]
        adj = adj[:, rem_vector]

        u = np.argmin(a[rem_vector].dot(adj!=0)/a[rem_vector])
        n_IS = -np.ones(adj.shape[0])
        n_IS[u] = 1
        neighbors = np.argwhere(adj[u,:]!=0)
        if neighbors.shape[0]:
            n_IS[neighbors] = 0
        IS[rem_vector] = n_IS
    #print(IS)
    mwis1 = []
    val = 0.0
    for u in graph:
        if IS[u] > 0:
            val = val + graph.nodes[u]['weight']
            mwis1.append(u)
    # print("Total Weight: {}".format(val))
    # print(mwis1)
    # print(dnx.is_independent_set(graph, mwis1))
    return mwis1, val


def mwis_heuristic_2(graph):
    mis_set = []
    mwis = []
    maxval = 0
    for u in graph:
        mis = nx.maximal_independent_set(graph, [u])
        # print(mis)
        mis_set.append(mis)
        val = 0
        for u in mis:
            val += graph.nodes[u]['weight']
        if val > maxval:
            maxval = val
            mwis = mis
    # mis_set
    # print(maxval)
    # print(mwis)
    # print(dnx.is_independent_set(graph, mwis))
    return mwis, maxval


def mwis_heuristic_greedy(graph):
    adj = nx.adjacency_matrix(graph)
    weights = np.array([graph.nodes[u]['weight'] for u in graph])
    mwis, total_wt = greedy_search(adj, weights)
    return mwis, total_wt


def mis_check(adj, mis):
    return True

def mwis_bruteforce(graph):
    adj = nx.adjacency_matrix(graph)
    weights = np.array([graph.nodes[u]['weight'] for u in graph])
    vertices = list(range(len(weights)))
    p_sets = powerset(vertices)
    mwis = []
    maxweights = 0.0
    cnt = 0
    for p_set in p_sets:
        cnt += 1
        if len(p_set) == 0:
            continue
        l_set = list(p_set)
        if not dnx.is_independent_set(graph, l_set):
            continue
        utility = np.sum(weights[l_set])
        if utility > maxweights:
            mwis = l_set
            maxweights = utility
    return mwis, maxweights

N_test = args.n #50 #10
# N_test = 1000
correctness = {}
maxweights = {}
Nb_Avgs = [100]


def generate_single_config(N, p, N_test):
    for i in range(N_test):
        filename = '{}_n{}_p{}_b{}_{}.mat'.format(args.type, N, p, i, dist_dict[dist])
        filepath = os.path.join(datapath, filename)
        print("Generating {}".format(filename))
        if args.type.lower() == 'er':
            graph = weighted_random_graph(N, p, dist)
        elif args.type.lower() == 'ppp':
            density = N * 0.01
            r = (10 * np.sqrt(p)) / (np.sqrt(3.1415926) - 2 * np.sqrt(p))
            graph = weighted_poisson_graph(100, density, radius=r, dist=dist)
        elif args.type.lower() == 'ba':
            graph = weighted_barabasi_albert_graph(N, p, dist)
        else:
            continue
        mwis2, val2 = mwis_heuristic_2(graph)
        mwis1, val1 = mwis_heuristic_1(graph)
        mwis0, val0 = mwis_heuristic_greedy(graph)
        # if args.bf:
        #     mwis, val = mwis_bruteforce(graph)
        # if not args.bf:
        if val1 > val2:
            mwis = mwis1
            val = val1
        else:
            mwis = mwis2
            val = val2
        adj_0 = nx.adj_matrix(graph)
        wts = np.array([graph.nodes[u]['weight'] for u in graph.nodes])
        mwis_label = np.zeros((len(graph),), dtype=np.float)
        mwis_label[mwis] = 1
        savemat(filepath, {'adj': adj_0.astype(np.float), 'weights': wts, 'N': N, 'p': p, 'mwis_label': mwis_label,
                           'mwis_utility': val, 'greedy_utility': val0})


# for N in [403, 1209]:
#     for p in [0.95, 0.9, 0.75, 0.5, 0.25, 0.1, 0.05]:
for N in size_list:
    if len(p_list) == 0:
        for Nb in nb_list:
            p = round(Nb/N, 3)
            generate_single_config(N, p, N_test)
    else:
        for p in p_list:
            generate_single_config(N, p, N_test)
