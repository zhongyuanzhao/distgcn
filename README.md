# Distributed Scheduling using Graph Neural Networks

Zhongyuan Zhao, Gunjan Verma, Chirag Rao, Ananthram Swami, and Santiago Segarra, &quot; Distributed Scheduling using Graph Neural Networks,&quot;  IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2021, pp. 4720-4724, doi: [10.1109/ICASSP39728.2021.9414098](https://doi.org/10.1109/ICASSP39728.2021.9414098). 


[![Watch ICASSP 2021 presentation on YouTube](https://img.youtube.com/vi/0ZzkDT5Q3Cs/0.jpg)](https://www.youtube.com/watch?v=0ZzkDT5Q3Cs)


- Full paper on [IEEEXplore](https://doi.org/10.1109/ICASSP39728.2021.9414098), _arXiv preprint_ <https://arxiv.org/abs/2011.09430>
- [Slides](https://sigport.org/sites/default/files/docs/Zhao_ICASSP2021_0.pdf), [SigPort](https://sigport.org/documents/distributed-scheduling-using-graph-neural-networks)
- [Source Code](https://github.com/zhongyuanzhao/distgcn/)
- Journal version manuscript (under review): <https://arxiv.org/abs/2109.05536>


## Abstract
A fundamental problem in the design of wireless networks is to efficiently schedule transmission in a distributed manner. The main challenge stems from the fact that optimal link scheduling involves solving a maximum weighted independent set (MWIS) problem, which is NP-hard. For practical link scheduling schemes, distributed greedy approaches are commonly used to approximate the solution of the MWIS problem. However, these greedy schemes mostly ignore important topological information of the wireless networks. To overcome this limitation, we propose a distributed MWIS solver based on graph convolutional networks (GCNs). In a nutshell, a trainable GCN module learns topology-aware node embeddings that are combined with the network weights before calling a greedy solver. In small- to middle-sized wireless networks with tens of links, even a shallow GCN-based MWIS scheduler can leverage the topological information of the graph to reduce in half the suboptimality gap of the distributed greedy solver with good generalizability across graphs and minimal increase in complexity.

_Key words_: Maximum weighted independent set; graph convolutional network; wireless network; scheduling

**BibTex**

```txt
@INPROCEEDINGS{9414098,
  author={Zhao, Zhongyuan and Verma, Gunjan and Rao, Chirag and Swami, Ananthram and Segarra, Santiago},
  booktitle={ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Distributed Scheduling Using Graph Neural Networks}, 
  year={2021},
  volume={},
  number={},
  pages={4720-4724},
  doi={10.1109/ICASSP39728.2021.9414098}}
```


## Instructions for replicate numerical experiments
The recommended system setup is Ubuntu linux 16.04LTS or 18.04LTS.

### 1. MWIS solvers on random graphs.
1.1 (Optinal) If you want replicate everything from scratch, run the following commands to generate training and testing datasets, and train the models. Otherwise, skip the following commands and go to step 1.2.
```bash
cd icassp; # to the root of this repository
bash ./bash/run_data_generation.sh; # generate training and testing datasets
bash ./bash/generalization_dqn.sh
```


1.2 Run following command to get the results in Figs. 3(a) and 3(b) based on trained models.
```bash
cd icassp; # to the root of this repository
bash ./bash/generalization_dqn_test.sh
```

1.3 For the testing datasets already exist in the repository, the total utilities of optimal solutions and approximate solutions of message passing [Paschalidis'15] are in following csv files:
```bash
./output/mlp_gurobi_BA_Graph_Uniform_GEN21_test2.csv
./output/mlp_gurobi_ER_Graph_Uniform_GEN21_test2.csv
./output/mp_clique_greedy_BA_Graph_Uniform_GEN21_test2.csv
./output/mp_clique_greedy_ER_Graph_Uniform_GEN21_test2.csv
```
The column **p** in these csvfiles represent the ratio of total utility of a solution from the tested algorithm to the total utility of greedy solution.

However, if you generated new test datasets as in step 1.1, you need also re-run the benchmark algorithms (optimal and message passing) on the new test datasets. 
Before running the python scripts, backup the existing optimal and benchmark solutions, otherwise the code will skip over the test graphs.
```bash
mv ./output/mlp_gurobi_BA_Graph_Uniform_GEN21_test2.csv ./output/mlp_gurobi_BA_Graph_Uniform_GEN21_test2_old.csv
mv ./output/mlp_gurobi_ER_Graph_Uniform_GEN21_test2.csv ./output/mlp_gurobi_ER_Graph_Uniform_GEN21_test2_old.csv
mv ./output/mp_clique_greedy_BA_Graph_Uniform_GEN21_test2.csv ./output/mp_clique_greedy_BA_Graph_Uniform_GEN21_test2_old.csv
mv ./output/mp_clique_greedy_ER_Graph_Uniform_GEN21_test2.csv ./output/mp_clique_greedy_ER_Graph_Uniform_GEN21_test2_old.csv
```
To get the correpsonding optimal solutions from Integer Programming with Gurobi solver (follow the [official doc](https://www.gurobi.com/documentation/) to install Gurobi), 
```bash
python3 mwis_mlp_test.py --datapath "./data/ER_Graph_Uniform_GEN21_test2" --solver "mlp_gurobi"
python3 mwis_mlp_test.py --datapath "./data/BA_Graph_Uniform_GEN21_test2" --solver "mlp_gurobi"
```
likewise, the approximate solutions from distributed message passing algorithm  in [Paschalidis'15], (based on clique constraints and distributed greedy estimation), can be obtained by 
```bash
python3 mwis_mlp_test.py --datapath "./data/ER_Graph_Uniform_GEN21_test2" --solver "mp_greedy"
python3 mwis_mlp_test.py --datapath "./data/BA_Graph_Uniform_GEN21_test2" --solver "mp_greedy"
```


### 2. GCN-based distributed scheduling.

The distributed scheduling experiment can be ran by following commands
```bash
bash ./bash/test_wireless.sh
```
you may change the `setval` and/or other hyperparameters in `./bash/test_wireless.sh` to the model you trained.


