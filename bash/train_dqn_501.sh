#!/bin/bash
graph='ER';
dist='Uniform';
train_data="${graph}_Graph_${dist}_mixN_mixp_train0";
test_data="${graph}_Graph_${dist}_GEN21_test1";
setval='IS4SAT';
python3 mwis_dqn_origin.py --training_set=${setval} --epsilon=1 --epsilon_min=0.002 --feature_size=1 --diver_num=1 --datapath=./data/${train_data} --test_datapath=./data/${test_data} --max_degree=1 --predict=mwis --learning_rate=0.00001 --hidden1=32 --num_layer=20 --epochs=25
# python3 mwis_dqn_origin.py --training_set=${setval} --epsilon=1 --epsilon_min=0.002 --feature_size=1 --diver_num=1 --datapath=./data/${train_data} --test_datapath=./data/${test_data} --max_degree=1 --predict=mwis --learning_rate=0.00001 --hidden1=32 --num_layer=2 --epochs=25
python3 mwis_dqn_origin.py --training_set=${setval} --epsilon=1 --epsilon_min=0.002 --feature_size=1 --diver_num=1 --datapath=./data/${train_data} --test_datapath=./data/${test_data} --max_degree=1 --predict=mwis --learning_rate=0.00001 --hidden1=32 --num_layer=3 --epochs=25
# python3 mwis_dqn_origin.py --training_set=${setval} --epsilon=1 --epsilon_min=0.002 --feature_size=1 --diver_num=1 --datapath=./data/${train_data} --test_datapath=./data/${test_data} --max_degree=1 --predict=mwis --learning_rate=0.00001 --hidden1=32 --num_layer=4 --epochs=25
python3 mwis_dqn_origin.py --training_set=${setval} --epsilon=1 --epsilon_min=0.002 --feature_size=1 --diver_num=1 --datapath=./data/${train_data} --test_datapath=./data/${test_data} --max_degree=1 --predict=mwis --learning_rate=0.00001 --hidden1=32 --num_layer=1 --epochs=25
