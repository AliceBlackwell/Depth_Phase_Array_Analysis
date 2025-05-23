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
import shutil
from pathlib import Path

from obspydmt import run_obspyDMT
from Z_processing import process_Z_components
from NE_processing import process_NE_components
from array_processing import run_array_processing
from figures import make_figures
from iscloc_wrapper import run_iscloc
from iscloc_results import strip_iscloc_results, extract_iscloc_relocation_depth

sys.path.append(os.path.abspath('pmP_Scripts'))
from pmP_crustal_thickness import determine_crustal_thickness
from pmP_catalogue import assemble_clean_pmP_cat

# Choose steps to run ------------------------------------------
#[Only skip steps if you already have the necessary outputs -- e.g. ObspyDMT catalogue, processed data]
#[Recommend doing everything in one go if you are looking at a single event]
#[If looking at multiple events, download intial obspyDMT catalogue separately then run rest of steps per event (can use a task array)]

# Run once 
make_obspydmt_catalogue = False

# Can be run as part of a task array, 1 process per event
download_data = False
process_data = True
array_process_data = True
make_array_figures = True
relocate_with_iscloc = True  # can be run once for all files in ISCloc/inputs --> all_events=True
find_crustal_thickness = True

# Run once if necessary
make_final_catalogues = False


# Earthquake to analyse ------------------------------------------
inputs = sys.argv[1:]
if not inputs:
    event = 1 # defaults to 1st event (potentially only event) in catalogue
    total_events = 1 # defaults to 1 event (potentially only event) in catalogue
    
else:
    try:
        event = int(sys.argv[1:][0]) # row number in ObspyDMT catalogue or txt file catalogue in individual_catalogues
        total_events = int(sys.argv[1:][1]) #in task array/loop etc.
    except:
        print()
        print("ERROR: Input command should be formatted 'python main.py n m', where n is the event number in obspyDMT catalogue to process and m is the total events being processed.")
        print("Leave m and n blank if running for a single event.")
        print()
        sys.exit()

# Set up -------------------------------------------------------
cat_name = 'ObspyDMT_Events_test'

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

final_EQ_cat_name='Final_3D_Catalogue'
final_EQ_cat_txt=str(results_dir)+'/' + final_EQ_cat_name + '.txt'


# Download initial ObpsyDMT event catalogue ----------------------
if make_obspydmt_catalogue:
    # Delete pre-existing ObspyDMT directory if exists
    if os.path.exists(str(obspydmt_dir)) and os.path.isdir(str(obspydmt_dir)):
        shutil.rmtree(str(obspydmt_dir))
        print('Deleted %s (Pre-existing ObspyDMT Catalogue)' %str(obspydmt_dir))

    # change search parameters in obspydmt.py
    run_obspyDMT(str(obspydmt_dir), make_catalogue=True, split_catalogue=True, download_data_Z=False, download_data_NEZ=False, single_event_download=False)

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
    run_obspyDMT(str(obspydmt_dir), make_catalogue=False, split_catalogue=False, download_data_Z=False, download_data_NEZ=True, single_event_download=True, event=event) #events count from 1


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
    run_array_processing(catalogue, event, str(results_dir), str(data_dir), component, do_array_processing=False, depth_conversion=False, iscloc=True)
    

    # Run again for cleaned diffential time outputs (for pmP detection etc)
    run_array_processing(catalogue, event, str(results_dir), str(data_dir), component, do_array_processing=False, depth_conversion=True, iscloc=False)


# Make array figures ---------------------------------------------
if make_array_figures:
    component = 'Z' # 'Z' or 'T'
    make_figures(catalogue, event, component, str(data_dir), str(results_dir))


# Run ISCloc -----------------------------------------------------
if relocate_with_iscloc:
    # Must have run compile_iscloc.sh in ISClocRelease2.2.6/src2.2.7
    run_iscloc(inputs_dir, station_list_dir, outputs_dir, all_events=False, catalogue=catalogue, event=event)
    
    # Find new relocated event depth from ISCloc (for pmP detection)
    depth = extract_iscloc_relocation_depth(outputs_dir, catalogue, event)
    
    # Make 3D earthquake relocation catalogue ------------------------
    if int(event) == int(total_events): # only run once, on final event
        strip_iscloc_results(final_EQ_cat_txt, analysis_only=False, iscloc_inputs=inputs_dir, iscloc_outputs=outputs_dir, include_original_phase_results=False)


# Detect pmP -----------------------------------------------------
if find_crustal_thickness:
    
    if os.path.exists(final_EQ_cat_txt):
        # Can only use the final catalogue for inital depths in pmP scripts if strip_iscloc_results has been run and generated Final_3D_Catalogue.txt
        determine_crustal_thickness(catalogue, event, pmP_dir, str(results_dir), reprocess=True, make_figures=True, plot_velocity_models=False, include_sea=False, final_EQ_cat_txt=final_EQ_cat_txt, depth=False)
    
    else:
        determine_crustal_thickness(catalogue, event, pmP_dir, str(results_dir), reprocess=True, make_figures=True, plot_velocity_models=False, include_sea=False, final_EQ_cat_txt=False, depth=depth) 

    # Make 3D earthquake relocation catalogue ------------------------
    if int(event) == int(total_events): # only run once, on final event
        assemble_clean_pmP_cat(uncleaned_pmP_cat=str(results_dir) + '/Final_pmP_catalogue_5.9.txt', final_3D_EQ_cat=final_EQ_cat_txt, obspydmt_cat_name=cat_file, cleaned_pmP_cat=str(results_dir) + '/Final_cleaned_pmP_catalogue.txt')

if make_final_catalogues == True:
    strip_iscloc_results(final_EQ_cat_txt, analysis_only=False, iscloc_inputs=inputs_dir, iscloc_outputs=outputs_dir, include_original_phase_results=False)	
    assemble_clean_pmP_cat(uncleaned_pmP_cat=str(results_dir) + '/Final_pmP_catalogue_5.9.txt', final_3D_EQ_cat=final_EQ_cat_txt, obspydmt_cat_name=cat_file, cleaned_pmP_cat=str(results_dir) + '/Final_cleaned_pmP_catalogue.txt')

print()
print('Scripts complete.')



