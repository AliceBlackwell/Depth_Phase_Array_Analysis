#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wrapper to create catalogue, download data, process data, array process, make array figures, relocate earthquakes in 3D with ISCloc, find crustal thickness with pmP.
[Comment in/out steps needed]

Created on 15/05/2025
@author: ee18ab
"""

# Import ------------------------------------------------------ 
import pickle
import sys
import os
from pathlib import Path

from obspydmt import run_obspyDMT
from Z_processing import process_Z_components
from NE_processing import process_NE_components
from array_processing import run_array_processing
from figures import make_figures
from iscloc_wrapper import run_iscloc
from iscloc_results import strip_iscloc_results

sys.path.append(os.path.abspath('pmP_Scripts'))
from pmP_crustal_thickness import determine_crustal_thickness

# Choose steps to run ------------------------------------------
#[Only skip steps if you already have the necessary outputs -- e.g. ObspyDMT catalogue, processed data]
#[Recommend doing everything in one go if you are looking at a single event]
#[If looking at multiple events, download intial obspyDMT catalogue separately then run rest of steps per event (can use a task array)]

make_obspydmt_catalogue = False
download_data = False
process_data = True
array_process_data = True
make_array_figures = True
relocate_with_iscloc = True
find_crustal_thickness = True


# Set up -------------------------------------------------------
cat_name = 'ObspyDMT_Events'

# Make project file structure
'''Parent_dir -- Scripts -- pmP_scripts
              -- Results -- ISCloc (optional)
              -- Processed_DATA
              -- ObspyDMT_dir'''

current_path = Path(__file__).resolve()
scripts_dir = current_path.parent
project_root = scripts_dir.parent

results_dir = project_root / 'Results'
obspydmt_dir = project_root / cat_name
data_dir = project_root / 'Processed_DATA'

for folder in [results_dir, obspydmt_dir, data_dir]:
    folder.mkdir(parents=True, exist_ok=True)

# Assign file pathways
pmP_dir = str(scripts_dir) +'/pmP_Scripts'

inputs_dir = str(results_dir) + '/ISCloc/inputs/'
station_list_dir =  str(results_dir) +'/ISCloc/stations/station_list.'
outputs_dir = str(results_dir) + '/ISCloc/outputs/'


# Earthquake to analyse ------------------------------------------
event = 1 # row number in ObspyDMT catalogue or txt file catalogue in individual_catalogues


# Download initial ObpsyDMT event catalogue ----------------------
if make_obspydmt_catalogue:
    # change search parameters in obspydmt.py
    run_obspyDMT(str(obspydmt_dir), make_catalogue=True, split_catalogue=False, download_data_Z=False, download_data_NEZ=False, single_event_download=False)

# Load in event catalogue
cat_file = str(obspydmt_dir) + '/EVENTS-INFO/catalog.ml.pkl'

with open(cat_file, 'rb') as f:
    catalogue = pickle.load(f)
print('No. of Events in Catalogue:', len(catalogue))


# Download seismic data ------------------------------------------
if download_data:
    # Download event data for whole ObspyDMT catalogue in sequence
    #run_obspyDMT(str(obspydmt_dir), make_catalogue=False, split_catalogue=False, download_data_Z=False, download_data_NEZ=True, single_event_download=False)

    # Download data for specific event in catalogue
    run_obspyDMT(str(obspydmt_dir), make_catalogue=False, split_catalogue=True, download_data_Z=False, download_data_NEZ=True, single_event_download=True, event=event) #events count from 1


if process_data:
    # Process Z component data ---------------------------------------
    re_processing = True
    process_Z_components(catalogue, event, re_processing, str(obspydmt_dir), str(project_root))

    # Process N/E/1/2 component data ---------------------------------
    re_processing = True
    process_NE_components(catalogue, event, re_processing, str(obspydmt_dir), str(project_root))


# Run array processing -------------------------------------------
if array_process_data:
    component = 'ZNE'   # string: 'Z' or 'ZNE'

    # ISCloc preparation
    run_array_processing(catalogue, event, str(results_dir), str(data_dir), component, do_array_processing=True, depth_conversion=False, iscloc=True)

    # Run again for cleaned diffential time outputs (for pmP detection etc)
    run_array_processing(catalogue, event, str(results_dir), str(data_dir), component, do_array_processing=False, depth_conversion=True, iscloc=False)


# Make array figures ---------------------------------------------
if make_array_figures:
    component = 'Z' # 'Z' or 'T'
    make_figures(catalogue, event, component, str(data_dir), str(results_dir))


# Run ISCloc -----------------------------------------------------
if relocate_with_iscloc:
    # Compile ISCloc with 'make' in src directory, move iscloc_nodb to the same file level as this script, put 'export QETC=file_path_to/ISClocRelease2.2.6/etc' in .bashrc
    run_iscloc(inputs_dir, station_list_dir, outputs_dir)

    # Make 3D earthquake relocation catalogue ------------------------
    strip_iscloc_results(str(results_dir)+'/Final_3D_Catalogue', analysis_only=False, iscloc_inputs=inputs_dir, iscloc_outputs=outputs_dir, include_original_phase_results=False)


# Detect pmP -----------------------------------------------------
if find_crustal_thickness:
    determine_crustal_thickness(catalogue, event, pmP_dir, str(results_dir), final_EQ_cat_txt=str(results_dir)+'/Final_3D_Catalogue.txt', reprocess=True, make_figures=True, plot_velocity_models=False, include_sea=False)

print()
print('Scripts complete.')



