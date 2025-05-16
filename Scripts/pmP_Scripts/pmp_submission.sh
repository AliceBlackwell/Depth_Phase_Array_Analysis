#!/bin/sh

# Script to write new submission scripts for Arc4 (December 2022)

# Print counter which runs through EQ catalogue
printf "Initial value of SGE=%d\n" $(($SGE_TASK_ID))

# Run with current modules and location
#$-V -cwd

# Specify to run on Voltec
# -P voltec

# Set running time limits
#$ -l h_rt=3:00:00

# Ask for some mpi cores
#$ -pe ib 1

# request memory
#$ -l h_vmem=8G

# Send start and end emails
#$ -m be

# Set number of submissions (1 per EQ)
#$ -t 1-3572:1
#763
# Restrict number of concurrent processes
#$ -tc 100

# Script to run
python Crustal_thickness_code_constantvel.py $(($SGE_TASK_ID))
#python Finding_failure_rate.py $(($SGE_TASK_ID))
echo "Crustal_thickness_code finished!"
