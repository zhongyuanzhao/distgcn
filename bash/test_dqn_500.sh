#!/bin/bash
# testfolder="${graph}_Graph_${dist}_GEN21_test2";


# python3 mwis_dqn_test.py --epsilon=.0002 --feature_size=1 --diver_num=1 --datapath=./data/ER_Graph_Uniform_GEN21_test2 --max_degree=1 --predict=mwis --learning_rate=0.00001 --hidden1=32 --num_layer=20 --epochs=10
# python3 mwis_dqn_test.py --epsilon=.0002 --feature_size=1 --diver_num=1 --datapath=./data/ER_Graph_Uniform_GEN21_test2 --max_degree=1 --predict=mwis --learning_rate=0.00001 --hidden1=32 --num_layer=2 --epochs=10
# python3 mwis_dqn_test.py --epsilon=.0002 --feature_size=1 --diver_num=1 --datapath=./data/ER_Graph_Uniform_GEN21_test2 --max_degree=1 --predict=mwis --learning_rate=0.00001 --hidden1=32 --num_layer=3 --epochs=10
# python3 mwis_dqn_test.py --epsilon=.0002 --feature_size=1 --diver_num=1 --datapath=./data/ER_Graph_Uniform_GEN21_test2 --max_degree=1 --predict=mwis --learning_rate=0.00001 --hidden1=32 --num_layer=4 --epochs=10

# python3 wireless_dqn_test.py --wt_sel=qrm --epsilon=1 --epsilon_min=0.0002 --feature_size=1 --diver_num=1 --datapath=./data/ER_Graph_Uniform_mixN_mixp_train0 --test_datapath=./data/ER_Graph_Uniform_GEN21_test1 --max_degree=1 --predict=mwis --learning_rate=0.00001 --hidden1=32 --num_layer=1 --epochs=10 > bash/wireless_dqn_test_flood.out ;
# python3 wireless_dqn_test.py --wt_sel=qrm --epsilon=1 --epsilon_min=0.0002 --feature_size=1 --diver_num=1 --datapath=./data/ER_Graph_Uniform_mixN_mixp_train0 --test_datapath=./data/ER_Graph_Uniform_GEN21_test1 --max_degree=1 --predict=mwis --learning_rate=0.00001 --hidden1=32 --num_layer=20 --epochs=10 > bash/wireless_dqn_test_flood_deep.out 


python3 wireless_dqn_test.py --wt_sel=qrm --epsilon=1 --epsilon_min=0.0002 --feature_size=1 --diver_num=1 --datapath=./data/ER_Graph_Uniform_mixN_mixp_train0 --test_datapath=./data/ER_Graph_Uniform_GEN21_test1 --max_degree=1 --predict=mwis --learning_rate=0.00001 --hidden1=32 --num_layer=1 --epochs=10 > bash/wireless_dqn_test_flood.out ;
