#!/bin/bash

python3 wireless_dqn_test_mc.py --wt_sel=qr --num_channels=3 --load_min=0.1 --load_max=1.2 --load_step=0.1 --feature_size=1 --epsilon_min=0.005 --diver_num=1 --datapath=./data/wireless_train --test_datapath=./data/wireless_test --max_degree=1 --predict=mwis --hidden1=32 --num_layer=1 --instances=2 --training_set=IS4SAT --opt=5 --output=./wireless/ > wireless_LGS-v_test_mg3_qr_fullrange_opt5.out &


python3 wireless_dqn_test_mc.py --wt_sel=qr --num_channels=3 --load_min=0.1 --load_max=1.2 --load_step=0.1 --feature_size=1 --epsilon_min=0.005 --diver_num=1 --datapath=./data/wireless_train --test_datapath=./data/wireless_test --max_degree=1 --predict=mwis --hidden1=32 --num_layer=1 --instances=2 --training_set=IS4SAT --opt=6 --output=./wireless/ > wireless_CRS-v_test_mg3_qr_fullrange_opt6.out &


python3 wireless_dqn_test_mc.py --wt_sel=qr --num_channels=3 --load_min=0.1 --load_max=1.2 --load_step=0.1 --feature_size=1 --epsilon_min=0.005 --diver_num=1 --datapath=./data/wireless_train --test_datapath=./data/wireless_test --max_degree=1 --predict=mwis --hidden1=32 --num_layer=1 --instances=2 --training_set=IS4SAT --opt=7 --output=./wireless/ > wireless_LGS-v_test_mg3_qr_fullrange_opt7.out 


echo "submitted wireless"