#!/bin/bash
python3 Data_Generation.py --datapath ./data/ER_Graph_Uniform > log_data_generation_uniform.txt;
echo "Uniform Weighted Graph Generation Completed";
python3 Data_Generation.py --datapath ./data/ER_Graph_Normal_l1 --dist normal_l1 > log_data_generation_normal_l1.txt;
echo "Normal L1 Weighted Graph Generation Completed";
python3 Data_Generation.py --datapath ./data/ER_Graph_Normal_l2 --dist normal_l2 > log_data_generation_normal_l2.txt;
echo "Normal L2 Weighted Graph Generation Completed";

python3 Data_Generation.py --datapath ./data/Ramdom_Graph_NB100_test --n 50 --nbs "100" > log_data_generation_uniform.txt;

python3 Data_Generation.py --datapath ./data/ER_Graph_Uniform_N200_h0 --n 1000 --sizes "200" --nbs "20,40,60,80,100,120,140,160,180" > log_data_generation_uniform.txt;

python3 Data_Generation.py --datapath ./data/ER_Graph_Uniform_N200 --n 2 --sizes "20" --nbs "2,4,6,8,10,12,14,16,18" > log_data_generation_uniform.txt;