#!/bin/bash

#SBATCH --mem=532480
#SBATCH --time=0
#SBATCH --cpus-per-task=6

~/miniconda3/envs/strain-design/bin/python TrainGeneralistModel.py $replicate $data_dir
