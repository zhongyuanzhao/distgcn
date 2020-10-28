#!/bin/bash
graph='BA';
# graph='PPP';

folder='regnet';
setval='ERGREG';
# dist='Normal_l2';
dist='Uniform';
feature_size=1;
num_layer=20;
hidden1=32;
diver_num=1;
modelfolder="result_${setval}_deep_ld${feature_size}_c${hidden1}_l${num_layer}_cheb1_diver${diver_num}_reg2";
rm -r ./${modelfolder};
mkdir ${folder};


testfolder="${graph}_Graph_${dist}_GEN21_test2";
if [ -d "./data/${testfolder}" ]; then
	echo "./data/${testfolder} already exist, skip Data_Generation";
else
	python3 Data_Generation.py --datapath ./data/${testfolder} --n 20 --sizes "100" --nbs "2,5,10,15,20" --type "${graph}" --dist "${dist}" >> log_data_generation_${dist}.txt;
	python3 Data_Generation.py --datapath ./data/${testfolder} --n 20 --sizes "150" --nbs "2,5,10,15,20" --type "${graph}" --dist "${dist}" >> log_data_generation_${dist}.txt;
	python3 Data_Generation.py --datapath ./data/${testfolder} --n 20 --sizes "200" --nbs "2,5,10,15,20" --type "${graph}" --dist "${dist}" >> log_data_generation_${dist}.txt;
	python3 Data_Generation.py --datapath ./data/${testfolder} --n 20 --sizes "250" --nbs "2,5,10,15,20" --type "${graph}" --dist "${dist}" >> log_data_generation_${dist}.txt;
	python3 Data_Generation.py --datapath ./data/${testfolder} --n 20 --sizes "300" --nbs "2,5,10,15,20" --type "${graph}" --dist "${dist}" >> log_data_generation_${dist}.txt;
	# testfolder='${graph}_Graph_${dist}_N200_test1';
fi

testfolder3="${graph}_Graph_${dist}_GEN21_test3";
if [ -d "./data/${testfolder3}" ]; then
	echo "./data/${testfolder3} already exist, skip Data_Generation";
else
	python3 Data_Generation.py --datapath ./data/${testfolder3} --n 20 --sizes "100" --nbs "2,5,10,15,20" --type "${graph}" --dist "${dist}" >> log_data_generation_${dist}.txt;
	python3 Data_Generation.py --datapath ./data/${testfolder3} --n 20 --sizes "150" --nbs "3,7.5,15,22.5,30" --type "${graph}" --dist "${dist}" >> log_data_generation_${dist}.txt;
	python3 Data_Generation.py --datapath ./data/${testfolder3} --n 20 --sizes "200" --nbs "4,10,20,30,40" --type "${graph}" --dist "${dist}" >> log_data_generation_${dist}.txt;
	python3 Data_Generation.py --datapath ./data/${testfolder3} --n 20 --sizes "250" --nbs "5,12.5,25,37.5,50" --type "${graph}" --dist "${dist}" >> log_data_generation_${dist}.txt;
	python3 Data_Generation.py --datapath ./data/${testfolder3} --n 20 --sizes "300" --nbs "6,15,30,45,60" --type "${graph}" --dist "${dist}" >> log_data_generation_${dist}.txt;
# testfolder='${graph}_Graph_${dist}_N200_test1';
fi

testfolder1="${graph}_Graph_${dist}_GEN21_test1";
if [ -d "./data/${testfolder1}" ]; then
	echo "./data/${testfolder1} already exist, skip Data_Generation";
else
	python3 Data_Generation.py --datapath ./data/${testfolder1} --n 2 --sizes "100" --nbs "2,5,10,15,20" --type "${graph}" --dist "${dist}" >> log_data_generation_${dist}.txt;
	python3 Data_Generation.py --datapath ./data/${testfolder1} --n 2 --sizes "150" --nbs "2,5,10,15,20" --type "${graph}" --dist "${dist}" >> log_data_generation_${dist}.txt;
	python3 Data_Generation.py --datapath ./data/${testfolder1} --n 2 --sizes "200" --nbs "2,5,10,15,20" --type "${graph}" --dist "${dist}" >> log_data_generation_${dist}.txt;
	python3 Data_Generation.py --datapath ./data/${testfolder1} --n 2 --sizes "250" --nbs "2,5,10,15,20" --type "${graph}" --dist "${dist}" >> log_data_generation_${dist}.txt;
	python3 Data_Generation.py --datapath ./data/${testfolder1} --n 2 --sizes "300" --nbs "2,5,10,15,20" --type "${graph}" --dist "${dist}" >> log_data_generation_${dist}.txt;
fi


