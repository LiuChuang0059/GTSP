#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10

# config=$1

# echo $(scontrol show hostnames $SLURM_JOB_NODELIST)
# source ~/.bashrc
# conda activate graph-aug

# echo CUDA_VISIBLE_DEVICES $CUDA_VISIBLE_DEVICES

# echo "python main.py --configs $config --num_workers 0 --devices $CUDA_VISIBLE_DEVICES"
# CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python main.py --configs $config --num_workers 8 --devices $CUDA_VISIBLE_DEVICES


for data in MUTAG #IMDB-BINARY #NCI109
do
	for ratio in 0.25 #0.5 0.75
	do
		for run_id in 1 #2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
		do
			python main.py --dataset $data\
					   --epochs 10\
				       --batch_size 128\
				       --eval_batch_size 128\
				       --num_layers $layer\
				       --run $run_id\
				       --sparse\
				   	   --weight_sparse\
				   	   --init-density 1.0 \
	                   --final-density $ratio\
	                   --update-frequency 200 \
	                   --final-prune-epoch 80 \
	                   --init_prune_epoch 10
		done
	done
done








