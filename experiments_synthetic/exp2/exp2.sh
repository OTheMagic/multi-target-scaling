#!/bin/bash
#SBATCH --job-name=exp2
#SBATCH --output=exp2_output.txt
#SBATCH --error=exp2_error.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=4G
#SBATCH --time=12:00:00 
#SBATCH --partition=main

# Make sure the output folder exists
mkdir -p exp2

# Run your Python experiment
/home1/brianfan/.conda/envs/multi_target_scaling/bin/python exp2.py