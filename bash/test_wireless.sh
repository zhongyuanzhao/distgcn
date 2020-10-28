#!/bin/bash
outfolder="wireless";
mkdir -p ${outfolder};

outputfile="${outfolder}/test_seed_$1_wts_$2.out"
python3 wireless.py --seed=$1 --wt_sel=$2 --output=${outfolder} --epsilon=1 --epsilon_min=0.001 --feature_size=1 --diver_num=1 --max_degree=1 --learning_rate=0.0001 --hidden1=1 --num_layer=1 > ${outputfile}