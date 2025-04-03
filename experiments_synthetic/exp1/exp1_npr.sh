#!/bin/bash
#SBATCH --job-name=exp1_npr
#SBATCH --output=exp1/exp1_npr_output.txt
#SBATCH --error=exp1/exp1_npr_error.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=12:00:00 
#SBATCH --partition=main

# Make sure the output folder exists
mkdir -p exp1

# Run your Python experiment
/home1/brianfan/.conda/envs/multi_target_scaling/bin/python exp1_npr.py