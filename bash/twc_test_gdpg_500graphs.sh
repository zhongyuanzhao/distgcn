#!/bin/bash
# testfolder="${graph}_Graph_${dist}_GEN21_test2";

setval="ERGDPG"

dist='Uniform';

for graph in 'ER' 'BA' ; do
	test_data="${graph}_Graph_${dist}_GEN21_test1";

	python3 mwis_dqn_test.py --training_set=${setval} --epsilon=.0002 --feature_size=1 --diver_num=1 --datapath=./data/${test_data} --max_degree=1 --predict=mwis --learning_rate=0.00001 --hidden1=32 --num_layer=20 --epochs=10
	python3 mwis_dqn_test.py --training_set=${setval} --epsilon=.0002 --feature_size=1 --diver_num=1 --datapath=./data/${test_data} --max_degree=1 --predict=mwis --learning_rate=0.00001 --hidden1=32 --num_layer=3 --epochs=10
	python3 mwis_dqn_test.py --training_set=${setval} --epsilon=.0002 --feature_size=1 --diver_num=1 --datapath=./data/${test_data} --max_degree=1 --predict=mwis --learning_rate=0.00001 --hidden1=32 --num_layer=2 --epochs=10
	python3 mwis_dqn_test.py --training_set=${setval} --epsilon=.0002 --feature_size=1 --diver_num=1 --datapath=./data/${test_data} --max_degree=1 --predict=mwis --learning_rate=0.00001 --hidden1=32 --num_layer=1 --epochs=10

done

