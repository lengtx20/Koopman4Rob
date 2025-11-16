#!/bin/bash
#PBS -N koopman_plane
#PBS -q v1_gpu72
#PBS -l select=1:ncpus=4:mem=24gb:ngpus=1
#PBS -l walltime=12:00:00
#PBS -o plane_train.log
#PBS -e plane_train.er


source miniconda3/bin/activate
conda activate koopman4rob

cd RSR/Koopman4rob
python3 main_demo.py


