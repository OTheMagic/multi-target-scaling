#!/bin/bash
#SBATCH --job-name=vcsn
#SBATCH --output=interesting_small_exps/vcsn_output.txt
#SBATCH --error=interesting_small_exps/vcsn_error.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=12:00:00 
#SBATCH --partition=main

# Make sure the output folder exists
mkdir -p interesting_small_exps

# Run your Python experiment
/home1/brianfan/.conda/envs/multi_target_scaling/bin/python vol_comp_small_n.py