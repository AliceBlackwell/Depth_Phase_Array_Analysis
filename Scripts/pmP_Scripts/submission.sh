#!/bin/bash

#SBATCH --job-name=obspyDMT_catalogue
#SBATCH --time=08:00:00
#SBATCH --mem=25G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=ee18ab@leeds.ac.uk
#SBATCH --mail-type=BEGIN,END

# python environment
#conda activate environment

# run job
#python 0_ObspyDMT.py 1
#python 1_Select_version_ObspyDMT_Process_Data.py 1
#python 1S_rewrite_1,2_ObspyDMT_Process_Data.py 1
#python 2_Relocate.py 1
#python /users/ee18ab/Relocation_Scripts/ISClocRelease2.2.6/etc/ISCloc_submission_script_Oct_2024.py
python Crustal_thickness_code_constantvel.py 1
