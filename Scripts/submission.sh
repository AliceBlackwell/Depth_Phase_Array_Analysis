#!/bin/bash

#SBATCH --job-name=depth_phase_array_analysis
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=25G
#SBATCH --array=1-121

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=ee18ab@leeds.ac.uk
#SBATCH --mail-type=BEGIN,END

# python environment
module load miniforge
conda activate dpa-env

# run job
python main.py $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_MAX
