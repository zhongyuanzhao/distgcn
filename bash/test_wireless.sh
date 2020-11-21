#!/bin/bash
outfolder="wireless";
mkdir -p ${outfolder};

setval='IS4SAT'
# setval='DGCNER'
# setval='DGCNBA'

python3 wireless_dqn_test.py --training_set=${setval} --wt_sel=qrm --epsilon=1 --epsilon_min=0.0002 --feature_size=1 --diver_num=1 --max_degree=1 --predict=mwis --learning_rate=0.00001 --hidden1=32 --num_layer=1 --epochs=10 > output/wireless_dqn_test_flood.out ;
python3 wireless_dqn_test.py --training_set=${setval} --wt_sel=qrm --epsilon=1 --epsilon_min=0.0002 --feature_size=1 --diver_num=1 --max_degree=1 --predict=mwis --learning_rate=0.00001 --hidden1=32 --num_layer=20 --epochs=10 > output/wireless_dqn_test_flood_deep.out 
