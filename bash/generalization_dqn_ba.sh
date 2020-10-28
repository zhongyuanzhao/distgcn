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


testfolder20="ER_Graph_Uniform_GEN21_test2";
testfolder21="ER_Graph_Normal_l2_GEN21_test2";
testfolder22="BA_Graph_Uniform_GEN21_test2";
testfolder23="BA_Graph_Normal_l2_GEN21_test2";

dist='Uniform'

for layers in 1 2 3 4 20; do
# for layers in 20; do
	for setval in 'IS4SAT' 'DQNBA'; do

		# rm -rf ./result_${setval}_deep_ld1_c32_l20_cheb1_diver1_mwis_dqn ;
		# cp -r ./${folder}/result_${setval}_deep_ld1_c32_l20_cheb1_diver1_mwis_dqn_${graph}_${dist} ./result_${setval}_deep_ld1_c32_l20_cheb1_diver1_mwis_dqn ;

		python3 mwis_dqn_test.py --training_set=${setval} --epsilon=.0002 --feature_size=1 --diver_num=1 --datapath=./data/${testfolder20} --max_degree=1 --predict=mwis --learning_rate=0.00001 --hidden1=32 --num_layer=${layers} --epochs=10
		mv ./output/result_${setval}_deep_ld1_c32_l${layers}_cheb1_diver1_mwis_dqn.csv ./output/result_${setval}_deep_ld1_c32_l${layers}_cheb1_diver1_mwis_dqn_${testfolder20}.csv

		# python3 mwis_dqn_test.py --training_set=${setval} --epsilon=.0002 --feature_size=1 --diver_num=1 --datapath=./data/${testfolder21} --max_degree=1 --predict=mwis --learning_rate=0.00001 --hidden1=32 --num_layer=20 --epochs=10
		# mv ./output/result_${setval}_deep_ld1_c32_l20_cheb1_diver1_mwis_dqn.csv ./output/result_${setval}_deep_ld1_c32_l20_cheb1_diver1_mwis_dqn_${graph}_${dist}_${testfolder21}.csv

		python3 mwis_dqn_test.py --training_set=${setval} --epsilon=.0002 --feature_size=1 --diver_num=1 --datapath=./data/${testfolder22} --max_degree=1 --predict=mwis --learning_rate=0.00001 --hidden1=32 --num_layer=${layers} --epochs=10
		mv ./output/result_${setval}_deep_ld1_c32_l${layers}_cheb1_diver1_mwis_dqn.csv ./output/result_${setval}_deep_ld1_c32_l${layers}_cheb1_diver1_mwis_dqn_${testfolder22}.csv

		# python3 mwis_dqn_test.py --training_set=${setval} --epsilon=.0002 --feature_size=1 --diver_num=1 --datapath=./data/${testfolder23} --max_degree=1 --predict=mwis --learning_rate=0.00001 --hidden1=32 --num_layer=20 --epochs=10
		# mv ./output/result_${setval}_deep_ld1_c32_l20_cheb1_diver1_mwis_dqn.csv ./output/result_${setval}_deep_ld1_c32_l20_cheb1_diver1_mwis_dqn_${graph}_${dist}_${testfolder23}.csv

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

