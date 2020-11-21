#!/bin/bash

folder='dqngen';
mkdir -p ${folder};

# dist='Normal_l2';
dist='Uniform';

for graph in 'ER' 'BA'; do	
	setval="DGCN${graph}";
	train_data="${graph}_Graph_${dist}_mixN_mixp_train0";

	for layers in 1 3 20; do
		python3 mwis_dqn_origin.py --training_set=${setval} --epsilon=1 --epsilon_min=0.002 --feature_size=1 --diver_num=1 --datapath=./data/${train_data} --test_datapath=./data/ER_Graph_Uniform_GEN21_test1 --max_degree=1 --predict=mwis --learning_rate=0.00001 --hidden1=32 --num_layer=${layers} --epochs=5
		python3 mwis_dqn_origin.py --training_set=${setval} --epsilon=0.2 --epsilon_min=0.002 --feature_size=1 --diver_num=1 --datapath=./data/${train_data} --test_datapath=./data/ER_Graph_Uniform_GEN21_test1 --max_degree=1 --predict=mwis --learning_rate=0.00001 --hidden1=32 --num_layer=${layers} --epochs=5
		python3 mwis_dqn_origin.py --training_set=${setval} --epsilon=0.1 --epsilon_min=0.002 --feature_size=1 --diver_num=1 --datapath=./data/${train_data} --test_datapath=./data/ER_Graph_Uniform_GEN21_test1 --max_degree=1 --predict=mwis --learning_rate=0.000001 --hidden1=32 --num_layer=${layers} --epochs=5
		python3 mwis_dqn_origin.py --training_set=${setval} --epsilon=0.05 --epsilon_min=0.002 --feature_size=1 --diver_num=1 --datapath=./data/${train_data} --test_datapath=./data/ER_Graph_Uniform_GEN21_test1 --max_degree=1 --predict=mwis --learning_rate=0.0000001 --hidden1=32 --num_layer=${layers} --epochs=10
		mv result_${setval}_deep_ld1_c32_l${layers}_cheb1_diver1_mwis_dqn ./${folder}/result_${setval}_deep_ld1_c32_l${layers}_cheb1_diver1_mwis_dqn_${graph}_${dist};
	done
done

testfolder20="ER_Graph_Uniform_GEN21_test2";
testfolder21="ER_Graph_Normal_l2_GEN21_test2";
testfolder22="BA_Graph_Uniform_GEN21_test2";
testfolder23="BA_Graph_Normal_l2_GEN21_test2";


for graph in 'ER' 'BA'; do

	setval="DGCN${graph}";

	for layers in 1 3 20; do

		rm -rf ./result_${setval}_deep_ld1_c32_l${layers}_cheb1_diver1_mwis_dqn ;
		cp -r ./${folder}/result_${setval}_deep_ld1_c32_l${layers}_cheb1_diver1_mwis_dqn_${graph}_${dist} ./result_${setval}_deep_ld1_c32_l${layers}_cheb1_diver1_mwis_dqn ;

		python3 mwis_dqn_test.py --training_set=${setval} --epsilon=.0002 --feature_size=1 --diver_num=1 --datapath=./data/${testfolder20} --max_degree=1 --predict=mwis --learning_rate=0.00001 --hidden1=32 --num_layer=${layers} --epochs=10
		mv ./output/result_${setval}_deep_ld1_c32_l20_cheb1_diver1_mwis_dqn.csv ./output/result_${setval}_deep_ld1_c32_l20_cheb1_diver1_mwis_dqn_${graph}_${dist}_${testfolder20}.csv

		# python3 mwis_dqn_test.py --training_set=${setval} --epsilon=.0002 --feature_size=1 --diver_num=1 --datapath=./data/${testfolder21} --max_degree=1 --predict=mwis --learning_rate=0.00001 --hidden1=32 --num_layer=${layers} --epochs=10
		# mv ./output/result_${setval}_deep_ld1_c32_l20_cheb1_diver1_mwis_dqn.csv ./output/result_${setval}_deep_ld1_c32_l20_cheb1_diver1_mwis_dqn_${graph}_${dist}_${testfolder21}.csv

		python3 mwis_dqn_test.py --training_set=${setval} --epsilon=.0002 --feature_size=1 --diver_num=1 --datapath=./data/${testfolder22} --max_degree=1 --predict=mwis --learning_rate=0.00001 --hidden1=32 --num_layer=${layers} --epochs=10
		mv ./output/result_${setval}_deep_ld1_c32_l20_cheb1_diver1_mwis_dqn.csv ./output/result_${setval}_deep_ld1_c32_l20_cheb1_diver1_mwis_dqn_${graph}_${dist}_${testfolder22}.csv

		# python3 mwis_dqn_test.py --training_set=${setval} --epsilon=.0002 --feature_size=1 --diver_num=1 --datapath=./data/${testfolder23} --max_degree=1 --predict=mwis --learning_rate=0.00001 --hidden1=32 --num_layer=${layers} --epochs=10
		# mv ./output/result_${setval}_deep_ld1_c32_l20_cheb1_diver1_mwis_dqn.csv ./output/result_${setval}_deep_ld1_c32_l20_cheb1_diver1_mwis_dqn_${graph}_${dist}_${testfolder23}.csv

	done
done

