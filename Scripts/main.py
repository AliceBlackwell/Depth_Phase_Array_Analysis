#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wrapper to create catalogue, download data, process data, array process, make array figures, relocate earthquakes in 3D with ISCloc, find crustal thickness with pmP.
[Comment in/out steps needed]

Created on 15/05/2025
@author: Alice Blackwell (and Hanna-Riia Allas for pmP Scripts)
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

from DOIs import make_doi_list

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
make_final_catalogues = True
make_data_doi_list = True


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

    # change search parameters directly in obspydmt.py
    '''  run_obspyDMT Flags: 
    make_catalogue: True/False, choose to create ObspyDMT initial catalogue of events
    split_catalogue: True/False, choose to split ObspyDMT catalogue text file into one text file per event (placed into individual_catalogues directory) --> useful for data download using task arrays/parallelisation
    download_data_Z: True/False, choose to download Z component data for ObspyDMT event catalogue
    download_data_NEZ: True/False, choose to download N,E,Z component data for ObspyDMT event catalogue
    single_event_download: True/False, choose whether you are running script for a single event or a whole catalogue downoad, i.e. True if using a task array/parallelisation which works with 1 event per process, False if running one job which works sequentially through the ObspyDMT catalogue.'''
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
    ''' process_Z_components Flags: 
    re_processing: True/False, choose to start data processing from 'saved' point --> .npy arrays saved in Processed_DATA/evname/Arrays directory which store names of stations which pass the processing steps per earthquake, these are then used to generate final MSEED and xml outputs again in Processed_DATA/evname/Stations and Processed_Data/evname/Data.'''
    re_processing = True
    process_Z_components(catalogue, event, re_processing, str(obspydmt_dir), str(project_root))

    # Process N/E/1/2 component data ---------------------------------
    ''' process_NE_components Flags: 
    re_processing: True/False, choose to start data processing from 'saved' point --> .npy arrays saved in Processed_DATA/evname/Arrays directory which store names of stations which pass the processing steps per earthquake, these are then used to generate final MSEED and xml outputs again in Processed_DATA/evname/Stations and Processed_Data/evname/Data.'''
    re_processing = True
    process_NE_components(catalogue, event, re_processing, str(obspydmt_dir), str(project_root))


# Run array processing -------------------------------------------
if array_process_data:
    ''' run_array_processing Flags: 
    component: 'Z' or 'ZNE', choose components to use in array processing
    do_array_processing: True/False, choose to redo array processing, or start from 'saved' point --> array_Z.npy and array_T.npy arrays saved in Results/evname/ directory which store array processing results (vespagrams, picks etc.) per earthquake, these are then used to generate final pick text files and/or ISCloc input files. If set to False and array_Z.npy/array_T.npy are missing, script will default to running array processing.
    depth_conversion: True/False, choose to generate final pick text files used for pmP analysis (based upon 1D event relocation in depth, not to be used as a new earthquake relocation, only for ad-hoc array pick summary files!)
    iscloc: True/False, choose to generate ISCloc input files (found in Results/ISCloc/inputs), ready for 3D earthquake relocation. '''
    component = 'ZNE'   # string: 'Z' or 'ZNE'

    # ISCloc preparation
    run_array_processing(catalogue, event, str(results_dir), str(data_dir), component, do_array_processing=False, depth_conversion=False, iscloc=True)
    

    # Run again for cleaned diffential time outputs (for pmP detection etc)
    run_array_processing(catalogue, event, str(results_dir), str(data_dir), component, do_array_processing=False, depth_conversion=True, iscloc=False)


# Make array figures ---------------------------------------------
if make_array_figures:
    # Turn figures on and off in figures.py script
    ''' make_figures Flags:
    component: 'Z' or 'T', choose component for which to make ad-hoc array figures. 
    
    Figures available: 
    	beampacking_figure -- polar plot showing beampacking search grid for P or S, and resultant peak normalised amplitude. 
        timeshifted_traces -- slowness corrected ad-hoc array traces used for beamforming
        vespagram_fig      -- simple vespagram for ad-hoc arrays
        threshold          -- plot showing the dynamic threshold selected for an ad-hoc array
        picking_figure     -- vespagram with picks
        comparison_grid    -- slowness and backazimuth with ranges set between 'calculated' values and 'beampack-determined' values per ad-hoc array, with resultant P wave beams 
        corr_fig           -- cross-correlation plots per trace in an ad-hoc array, comparing trace to beam
        QC_fig             -- vespagram plots used for quality controlling the ad-hoc arrays, checking for coherent peaks along the expected slowness found for the ad-hoc array
        final_vespa        -- vespagram with finalised picks only
        beampack_beams     -- combined figure with polar plot from beampacking_figure, and a comparison between the 'calculated' and 'beampacking-determined' beams per ad-hoc array
        vespas_combined    -- vespagram, optimum beam and all picks (including highlighted final picks)
        QC_vespas_combined -- vespgram alongside quality control vespagram tests from QC_fig. 
        
        Recommend as default: picking_figure, beampack_beams and vespas_combined. '''
        
    component = 'Z'
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
        ''' determine_crustal_thickness Flags:
        reprocess: True/False, choose to start data processing from 'saved' point --> .npy arrays saved
        make_figures: True/False, choose to make ad-hoc array pmP figures
        plot_velocity_models: True/False, choose to plot velocity model used during pmP detection
        include_sea: True/False, choose to include sea or not in velocity model used for pmP detection (include_sea = True is largely untested!).'''
        determine_crustal_thickness(catalogue, event, pmP_dir, str(results_dir), reprocess=True, make_figures=True, plot_velocity_models=False, include_sea=False, final_EQ_cat_txt=final_EQ_cat_txt, depth=False)
    
    else:
        determine_crustal_thickness(catalogue, event, pmP_dir, str(results_dir), reprocess=True, make_figures=True, plot_velocity_models=False, include_sea=False, final_EQ_cat_txt=False, depth=depth) 

    # Make final cleaned pmP catalogue ------------------------
    if int(event) == int(total_events): # only run once, on final event
        assemble_clean_pmP_cat(uncleaned_pmP_cat=str(results_dir) + '/Final_pmP_catalogue_5.9.txt', final_3D_EQ_cat=final_EQ_cat_txt, obspydmt_cat_name=cat_file, cleaned_pmP_cat=str(results_dir) + '/Final_cleaned_pmP_catalogue.txt')

# Force scripts to make final catalogues --------------------------
if make_final_catalogues == True:
    strip_iscloc_results(final_EQ_cat_txt, analysis_only=False, iscloc_inputs=inputs_dir, iscloc_outputs=outputs_dir, include_original_phase_results=False)	
    assemble_clean_pmP_cat(uncleaned_pmP_cat=str(results_dir) + '/Final_pmP_catalogue_5.9.txt', final_3D_EQ_cat=final_EQ_cat_txt, obspydmt_cat_name=cat_file, cleaned_pmP_cat=str(results_dir) + '/Final_cleaned_pmP_catalogue.txt')

# Make list of station/seismic data DOIs
if make_data_doi_list == True:
    make_doi_list()

print()
print('Scripts complete.')



