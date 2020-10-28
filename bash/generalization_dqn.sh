#!/bin/bash
folder='dqngen';
mkdir ${folder};
graph='ER';
# graph='PPP';

setval='DQNGEN';
# dist='Normal_l2';
# dist='Uniform';
# rm -r ./result_${setval}_deep_ld32_c32_l20_cheb1_diver32_res32;
size=150;

for graph in 'ER' 'PPP'; do
	for dist in 'Uniform' 'Normal_l2'; do
		testfolder="${graph}_Graph_${dist}_GEN21_test2";

		train_data="${graph}_Graph_${dist}_N${size}_mixp_train2";
		if [ -d "./data/${train_data}" ]; then
			echo "${train_data} already exists, skip generation.";
		else
			python3 Data_Generation.py --datapath ./data/${train_data} --n 1000 --sizes "${size}" --ps "0.02,0.05,0.075,0.10,0.15" --type "${graph}" >> log_data_generation_${dist}.txt;
			python3 Data_Generation.py --datapath ./data/${train_data} --n 100 --sizes "${size}" --ps "0.90,0.80,0.70,0.60,0.50,0.40,0.30" --type "${graph}" --dist "${dist}" >> log_data_generation_${dist}.txt;
			python3 Data_Generation.py --datapath ./data/${train_data} --n 30 --sizes "20" --nbs "18,16,14,12,10,8,6,4,2" --type "${graph}" --dist "${dist}" >> log_data_generation_${dist}.txt;
			# python3 Data_Generation.py --datapath ./data/${graph}_Graph_${dist}_N${size}_train0 --n 100 --sizes "${size}" --nbs "90,80,70,60,50,40,30" --type "${graph}" >> log_data_generation_${dist}.txt;
			# python3 Data_Generation.py --datapath ./data/${graph}_Graph_${dist}_N${size}_train0 --n 100 --sizes "20" --nbs "18,16,14,12,10,8,6,4,2" --type "${graph}" >> log_data_generation_${dist}.txt;
		fi

		python3 mwis_dqn_origin.py --training_set=${setval} --epsilon=1 --epsilon_min=0.002 --feature_size=1 --diver_num=1 --datapath=./data/${train_data} --test_datapath=./data/ER_Graph_Uniform_GEN21_test1 --max_degree=1 --predict=mwis --learning_rate=0.00001 --hidden1=32 --num_layer=20 --epochs=5
		python3 mwis_dqn_origin.py --training_set=${setval} --epsilon=0.2 --epsilon_min=0.002 --feature_size=1 --diver_num=1 --datapath=./data/${train_data} --test_datapath=./data/ER_Graph_Uniform_GEN21_test1 --max_degree=1 --predict=mwis --learning_rate=0.00001 --hidden1=32 --num_layer=20 --epochs=5
		python3 mwis_dqn_origin.py --training_set=${setval} --epsilon=0.1 --epsilon_min=0.002 --feature_size=1 --diver_num=1 --datapath=./data/${train_data} --test_datapath=./data/ER_Graph_Uniform_GEN21_test1 --max_degree=1 --predict=mwis --learning_rate=0.000001 --hidden1=32 --num_layer=20 --epochs=5
		python3 mwis_dqn_origin.py --training_set=${setval} --epsilon=0.05 --epsilon_min=0.002 --feature_size=1 --diver_num=1 --datapath=./data/${train_data} --test_datapath=./data/ER_Graph_Uniform_GEN21_test1 --max_degree=1 --predict=mwis --learning_rate=0.0000001 --hidden1=32 --num_layer=20 --epochs=10

		mv result_${setval}_deep_ld1_c32_l20_cheb1_diver1_mwis_dqn ./${folder}/result_${setval}_deep_ld1_c32_l20_cheb1_diver1_mwis_dqn_${graph}_${dist};

	done
done

testfolder20="ER_Graph_Uniform_GEN21_test2";
testfolder21="ER_Graph_Normal_l2_GEN21_test2";
testfolder22="PPP_Graph_Uniform_GEN21_test2";
testfolder23="PPP_Graph_Normal_l2_GEN21_test2";


for graph in 'ER' 'PPP'; do
	for dist in 'Uniform' 'Normal_l2'; do
		testfolder="${graph}_Graph_${dist}_GEN21_test2";

		rm -rf ./result_${setval}_deep_ld1_c32_l20_cheb1_diver1_mwis_dqn ;
		cp -r ./${folder}/result_${setval}_deep_ld1_c32_l20_cheb1_diver1_mwis_dqn_${graph}_${dist} ./result_${setval}_deep_ld1_c32_l20_cheb1_diver1_mwis_dqn ;

		python3 mwis_dqn_test.py --training_set=${setval} --epsilon=.0002 --feature_size=1 --diver_num=1 --datapath=./data/${testfolder20} --max_degree=1 --predict=mwis --learning_rate=0.00001 --hidden1=32 --num_layer=20 --epochs=10
		mv ./output/result_${setval}_deep_ld1_c32_l20_cheb1_diver1_mwis_dqn.csv ./output/result_${setval}_deep_ld1_c32_l20_cheb1_diver1_mwis_dqn_${graph}_${dist}_${testfolder20}.csv

		python3 mwis_dqn_test.py --training_set=${setval} --epsilon=.0002 --feature_size=1 --diver_num=1 --datapath=./data/${testfolder21} --max_degree=1 --predict=mwis --learning_rate=0.00001 --hidden1=32 --num_layer=20 --epochs=10
		mv ./output/result_${setval}_deep_ld1_c32_l20_cheb1_diver1_mwis_dqn.csv ./output/result_${setval}_deep_ld1_c32_l20_cheb1_diver1_mwis_dqn_${graph}_${dist}_${testfolder21}.csv

		python3 mwis_dqn_test.py --training_set=${setval} --epsilon=.0002 --feature_size=1 --diver_num=1 --datapath=./data/${testfolder22} --max_degree=1 --predict=mwis --learning_rate=0.00001 --hidden1=32 --num_layer=20 --epochs=10
		mv ./output/result_${setval}_deep_ld1_c32_l20_cheb1_diver1_mwis_dqn.csv ./output/result_${setval}_deep_ld1_c32_l20_cheb1_diver1_mwis_dqn_${graph}_${dist}_${testfolder22}.csv

		python3 mwis_dqn_test.py --training_set=${setval} --epsilon=.0002 --feature_size=1 --diver_num=1 --datapath=./data/${testfolder23} --max_degree=1 --predict=mwis --learning_rate=0.00001 --hidden1=32 --num_layer=20 --epochs=10
		mv ./output/result_${setval}_deep_ld1_c32_l20_cheb1_diver1_mwis_dqn.csv ./output/result_${setval}_deep_ld1_c32_l20_cheb1_diver1_mwis_dqn_${graph}_${dist}_${testfolder23}.csv

	done
done

# testfolder10="ER_Graph_Uniform_GEN21_test1";
# testfolder11="ER_Graph_Normal_l2_GEN21_test1";
# testfolder12="PPP_Graph_Uniform_GEN21_test1";
# testfolder13="PPP_Graph_Normal_l2_GEN21_test1";

# trainfolder20="ER_Graph_Uniform_N${size}_mixp_train2";
# trainfolder21="ER_Graph_Normal_l2_N${size}_mixp_train2";
# trainfolder22="PPP_Graph_Uniform_N${size}_mixp_train2";
# trainfolder23="PPP_Graph_Normal_l2_N${size}_mixp_train2";

