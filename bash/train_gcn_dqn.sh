#!/bin/bash
# python3 mwis_dqn.py --training_set='ERGUNI' --learning_rate=0.00001 --num_layer=30 --hidden1=32 --diver_num=1 --diver_out=1 --backoff_prob=0.0 --datapath='./data/Random_Graph_N20' --test_datapath=./data/ER_Graph_Uniform_NP20_test  --skip=False > dqn_train_test.txt

# python3 mwis_dqn.py --training_set='ERGUNI' --learning_rate=0.00001 --num_layer=40 --hidden1=32 --diver_num=1 --diver_out=1 --backoff_prob=0.0 --datapath='./data/Random_Graph_N20' --test_datapath=./data/ER_Graph_Uniform_N20_test --skip=False


# python3 mwis_dqn.py --training_set='ERGUNI' --learning_rate=0.00001 --num_layer=20 --hidden1=32 --diver_num=1 --diver_out=1 --backoff_prob=0.0 --datapath='./data/Random_Graph_N20' --predict=mis


# python3 demo_mwis.py --training_set='ERGNTX' --learning_rate=0.00001 --num_layer=20 --hidden1=32 --diver_num=32 --diver_out=1 --backoff_prob=0.0 --greedy=1 --datapath='./data/ER_Graph_Uniform_N20_test'

# python3 mwis_parallel.py --training_set='ERGNTX' --learning_rate=0.00001 --num_layer=20 --hidden1=32 --diver_num=32 --diver_out=32 --backoff_prob=0.0 --greedy=2 --datapath='./data/ER_Graph_Uniform_N20_test'



# python3 mwis_dqn_test.py --training_set='ERGUNI' --learning_rate=0.00001 --num_layer=22 --hidden1=32 --diver_num=1 --diver_out=1 --backoff_prob=0.0 --datapath='./data/ER_Graph_Uniform_NP20_test' --predict=mis > dqn_mis_rollout_test.txt;
# python3 mwis_dqn_test.py --training_set='ERGUNI' --learning_rate=0.00001 --num_layer=22 --hidden1=32 --diver_num=1 --diver_out=1 --backoff_prob=0.0 --datapath='./data/ER_Graph_Uniform_NP20_test' --predict=mwis > dqn_mwis_rollout_test.txt;


# python3 mwis_dqn.py --num_layer=22 --diver_num=1 --datapath=./data/Random_Graph_N20 --learning_rate=0.00001 --training_set=ERGUNI --test_datapath=./data/ER_Graph_Uniform_N20_test --predict=mwis

# python3 mwis_dqn_origin.py --epsilon=1 --epsilon_min=0.002 --feature_size=1 --diver_num=1 --datapath=./data/ER_Graph_Uniform_mixN_mixp_train1 --test_datapath=./data/ER_Graph_Uniform_GEN21_test1 --max_degree=1 --predict=mwis --learning_rate=0.00001 --hidden1=32 --num_layer=20 --epochs=5
python3 mwis_dqn_origin.py --epsilon=0.2 --epsilon_min=0.002 --feature_size=1 --diver_num=1 --datapath=./data/ER_Graph_Uniform_mixN_mixp_train1 --test_datapath=./data/ER_Graph_Uniform_GEN21_test1 --max_degree=1 --predict=mwis --learning_rate=0.00001 --hidden1=32 --num_layer=20 --epochs=5
python3 mwis_dqn_origin.py --epsilon=0.1 --epsilon_min=0.002 --feature_size=1 --diver_num=1 --datapath=./data/ER_Graph_Uniform_mixN_mixp_train1 --test_datapath=./data/ER_Graph_Uniform_GEN21_test1 --max_degree=1 --predict=mwis --learning_rate=0.000001 --hidden1=32 --num_layer=20 --epochs=5
python3 mwis_dqn_origin.py --epsilon=0.05 --epsilon_min=0.002 --feature_size=1 --diver_num=1 --datapath=./data/ER_Graph_Uniform_mixN_mixp_train1 --test_datapath=./data/ER_Graph_Uniform_GEN21_test1 --max_degree=1 --predict=mwis --learning_rate=0.0000001 --hidden1=32 --num_layer=20 --epochs=10

# python3 wireless_dqn_train.py --wt_sel=random --epsilon=1.0 --epsilon_min=0.0002 --feature_size=1 --diver_num=1 --datapath=./data/ER_Graph_Uniform_mixN_mixp_train0 --test_datapath=./data/ER_Graph_Uniform_GEN21_test1 --max_degree=1 --predict=mwis --learning_rate=0.00001 --hidden1=32 --num_layer=20 --epochs=2
# python3 wireless_dqn_train.py --wt_sel=random --epsilon=0.5 --epsilon_min=0.0002 --feature_size=1 --diver_num=1 --datapath=./data/ER_Graph_Uniform_mixN_mixp_train0 --test_datapath=./data/ER_Graph_Uniform_GEN21_test1 --max_degree=1 --predict=mwis --learning_rate=0.00001 --hidden1=32 --num_layer=20 --epochs=2
# python3 wireless_dqn_train.py --wt_sel=random --epsilon=0.1 --epsilon_min=0.0002 --feature_size=1 --diver_num=1 --datapath=./data/ER_Graph_Uniform_mixN_mixp_train0 --test_datapath=./data/ER_Graph_Uniform_GEN21_test1 --max_degree=1 --predict=mwis --learning_rate=0.00001 --hidden1=32 --num_layer=20 --epochs=2
# python3 wireless_dqn_train.py --wt_sel=random --epsilon=0.1 --epsilon_min=0.0002 --feature_size=1 --diver_num=1 --datapath=./data/ER_Graph_Uniform_mixN_mixp_train0 --test_datapath=./data/ER_Graph_Uniform_GEN21_test1 --max_degree=1 --predict=mwis --learning_rate=0.00001 --hidden1=32 --num_layer=20 --epochs=2
python3 wireless_dqn_train.py --wt_sel=qrm --epsilon=0.0001 --epsilon_min=0.0001 --feature_size=1 --diver_num=1 --datapath=./data/ER_Graph_Uniform_mixN_mixp_train0 --test_datapath=./data/ER_Graph_Uniform_GEN21_test1 --max_degree=1 --predict=mwis --learning_rate=0.00001 --hidden1=32 --num_layer=20 --epochs=1 > wireless_dqn_test.out
