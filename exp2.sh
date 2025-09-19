#!/bin/bash
#SBATCH --job-name=exp1_lpr1
#SBATCH --output=exp1/exp1_lpr_output.txt
#SBATCH --error=exp1/exp1_lpr_error.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=23:00:00 
#SBATCH --partition=main

# Make sure the output folder exists
mkdir -p exp2

# Run your Python experiment
/home1/brianfan/.conda/envs/multi_target_scaling/bin/python exp2.py