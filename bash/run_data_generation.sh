#!/bin/bash

dist='Uniform';

for graph in 'ER' 'BA'; do

	train_data="${graph}_Graph_${dist}_mixN_mixp_train0";
	if [ -d "./data/${train_data}" ]; then
		echo "./data/${train_data} already exists, skip generation.";
	else
		echo "$Start generating training dataset: ./data/{train_data} .";
		python3 Data_Generation.py --datapath ./data/${train_data} --n 200 --sizes "50,100,150,200,250" --ps "0.02,0.05,0.075,0.10,0.15" --type "${graph}"  --dist "${dist}" > log_data_generation_${train_data}.txt;
		python3 Data_Generation.py --datapath ./data/${train_data} --n 20 --sizes "50,100,150,200,250" --ps "0.90,0.80,0.70,0.60,0.50,0.40,0.30" --type "${graph}" --dist "normal_l2" >> log_data_generation_${train_data}.txt;
		python3 Data_Generation.py --datapath ./data/${train_data} --n 30 --sizes "20" --nbs "18,16,14,12,10,8,6,4,2" --type "${graph}" --dist "normal_l2" >> log_data_generation_${train_data}.txt;
	fi

	testfolder="${graph}_Graph_${dist}_GEN21_test2";
	if [ -d "./data/${testfolder}" ]; then
		echo "./data/${testfolder} already exist, skip generation";
	else
		echo "$Start generating testing dataset: ./data/{testfolder} .";
		python3 Data_Generation.py --datapath ./data/${testfolder} --n 20 --sizes "100" --nbs "2,5,10,15,20" --type "${graph}" --dist "${dist}" >> log_data_generation_${testfolder}.txt;
		python3 Data_Generation.py --datapath ./data/${testfolder} --n 20 --sizes "150" --nbs "2,5,10,15,20" --type "${graph}" --dist "${dist}" >> log_data_generation_${testfolder}.txt;
		python3 Data_Generation.py --datapath ./data/${testfolder} --n 20 --sizes "200" --nbs "2,5,10,15,20" --type "${graph}" --dist "${dist}" >> log_data_generation_${testfolder}.txt;
		python3 Data_Generation.py --datapath ./data/${testfolder} --n 20 --sizes "250" --nbs "2,5,10,15,20" --type "${graph}" --dist "${dist}" >> log_data_generation_${testfolder}.txt;
		python3 Data_Generation.py --datapath ./data/${testfolder} --n 20 --sizes "300" --nbs "2,5,10,15,20" --type "${graph}" --dist "${dist}" >> log_data_generation_${testfolder}.txt;
	fi

done
