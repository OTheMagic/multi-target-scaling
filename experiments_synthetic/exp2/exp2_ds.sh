#!/bin/bash
#SBATCH --job-name=exp2_ds
#SBATCH --output=exp2/exp2_ds_output.txt
#SBATCH --error=exp2/exp2_ds_error.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=12:00:00 
#SBATCH --partition=main

# Make sure the output folder exists
mkdir -p exp2

# Run your Python experiment
/home1/brianfan/.conda/envs/multi_target_scaling/bin/python exp2_ds.py