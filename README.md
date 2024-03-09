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
The recommended system setup is Ubuntu linux 20.04LTS.

The [old instructions](https://github.com/zhongyuanzhao/distgcn/blob/icassp2021/README.md) are also helpful, but not repeated here.

Training script `./bash/twc_train_gcn_gdpg.sh`

Test on graphs `./bash/twc_test_gdpg_500graphs.sh `

Test for wireless scheduling in single channel: `./bash/test_wireless_gcn_dqn.sh`, `./bash/test_wireless_gcn_rollout.sh` 

Test for wireless scheduling in multiple subchannels: `./bash/twc_major_wireless_mc_test.sh`

