#!/bin/bash
#PBS -N koopman_plane_cpu
#PBS -q cpu                  
#PBS -l select=1:ncpus=8:mem=8gb
#PBS -l walltime=15:00:00     
#PBS -o plane_train_cpu.log
#PBS -e plane_train_cpu.err

cd $PBS_O_WORKDIR

source /rds/general/user/jw3425/home/miniconda3/bin/activate
conda activate koopman4rob


python main_demo.py stage=train datasets=null 
