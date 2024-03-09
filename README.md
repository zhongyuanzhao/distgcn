# Link Scheduling using Graph Neural Networks

Z. Zhao, G. Verma, C. Rao, A. Swami and S. Segarra, "Link Scheduling Using Graph Neural Networks," in <i>IEEE Transactions on Wireless Communications</i>, vol. 22, no. 6, pp. 3997-4012, June 2023, doi: [10.1109/TWC.2022.3222781](https://doi.org/10.1109/TWC.2022.3222781).


- Journal paper in [IEEE TWC](https://ieeexplore.ieee.org/document/9962800), [arXiv preprint](https://arxiv.org/abs/2109.05536), Source code: [main](https://github.com/zhongyuanzhao/distgcn/tree/main/)
- Conference paper on [IEEEXplore](https://doi.org/10.1109/ICASSP39728.2021.9414098), [arXiv preprint](https://arxiv.org/abs/2011.09430), Source code: [icassp2021 branch](https://github.com/zhongyuanzhao/distgcn/tree/icassp2021/)
- [Slides](https://sigport.org/sites/default/files/docs/Zhao_ICASSP2021_0.pdf), [SigPort](https://sigport.org/documents/distributed-scheduling-using-graph-neural-networks)

[![Watch ICASSP 2021 presentation on YouTube](https://img.youtube.com/vi/0ZzkDT5Q3Cs/0.jpg)](https://www.youtube.com/watch?v=0ZzkDT5Q3Cs)


## Abstract
Efficient scheduling of transmissions is a key problem in wireless networks. The main challenge stems from the fact that optimal link scheduling involves solving a maximum weighted independent set (MWIS) problem, which is known to be NP-hard. In practical schedulers, centralized and distributed greedy heuristics are commonly used to approximately solve the MWIS problem. However, these greedy heuristics mostly ignore important topological information of the wireless network. To overcome this limitation, we propose fast heuristics based on graph convolutional networks (GCNs) that can be implemented in centralized and distributed manners. Our centralized heuristic is based on tree search guided by a GCN and 1-step rollout. In our distributed MWIS solver, a GCN generates topology-aware node embeddings that are combined with per-link utilities before invoking a distributed greedy solver. Moreover, a novel reinforcement learning scheme is developed to train the GCN in a non-differentiable pipeline. Test results on medium-sized wireless networks show that our centralized heuristic can reach a near-optimal solution quickly, and our distributed heuristic based on a shallow GCN can reduce by nearly half the suboptimality gap of the distributed greedy solver with minimal increase in complexity. The proposed schedulers also exhibit good generalizability across graph and weight distributions.



_Key words_: Maximum weighted independent set; graph convolutional network; wireless networks; scheduling; reinforcement learning

**BibTex**

```txt
@ARTICLE{9962800,
  author={Zhao, Zhongyuan and Verma, Gunjan and Rao, Chirag and Swami, Ananthram and Segarra, Santiago},
  journal={IEEE Transactions on Wireless Communications}, 
  title={Link Scheduling Using Graph Neural Networks}, 
  year={2023},
  volume={22},
  number={6},
  pages={3997-4012},
  keywords={Wireless networks;Processor scheduling;Complexity theory;Scheduling;Optimal scheduling;Ad hoc networks;Routing;MWIS;graph convolutional networks;wireless networks;scheduling;reinforcement learning},
  doi={10.1109/TWC.2022.3222781}}
```

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


