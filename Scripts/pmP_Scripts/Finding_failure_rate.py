#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Tue Jul 11 15:38:12 2023

Author: Hanna-Riia Allas

Code to identify depth phase precursors in teleseismic earthquake data (pmP phase) and
use them to calculate crustal thickness from pP-pmP differential arrival times.

This code uses the outputs of the Relocation code and should be run after 2_Relocate.py.

"""

# Importing modules

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import os
import sys
import shutil
import re
from obspy.taup import TauPyModel

import obspy
from obspy import read_events
import matplotlib.pyplot as plt

# Importing classes

from Classes_for_crthk_code import Cr_subarray
from crust1 import crustModel

# Importing functions

from Functions_for_crthk_code import Get_crustal_thickness
from Functions_for_crthk_code import Build_velocity_model
from Functions_for_crthk_code import Crustal_thickness_forward_modelling
from Functions_for_crthk_code import Plot_bounce_points
from Functions_for_crthk_code import Plot_delay_time_histogram
from Functions_for_crthk_code import Plot_crthk_histogram
from Functions_for_crthk_code import Plot_all_event_data
from Functions_for_crthk_code import write_out_results


# ---------- PRELIMINARIES ----------

# Reprocess_flag
reprocess = True

#------------------------------------------------------------------------------
make_figures = False
plot_velocity_models = False
include_sea = False
#------------------------------------------------------------------------------

# Load in events from txt file (un-relocated catalogue)
#catalogue_name = '/localhome/not-backed-up/ee18ab/00_ANDES_DATA/ANDES_ObspyDMT'
catalogue_name = '/nobackup/ee18ab/ObspyDMT_ANDES/ANDES_ObspyDMT'

# Load in one event from txt file
catalogue_file = catalogue_name + '/EVENTS-INFO/catalog.ml.pkl'

import pickle
with open(catalogue_file, 'rb') as f:
    catalogue = pickle.load(f)

# Load in event data
input_no = sys.argv[1:]

#input_no = [2086] # TEMPORARY

event = catalogue[int(input_no[0])-1]
event_id = re.sub("[^0-9]", "", str(event.resource_id))
evla = event.origins[0].latitude
evlo = event.origins[0].longitude    
evmag = event.magnitudes[0].mag
evdp = event.origins[0].depth/1000
yyyy = int(event.origins[0].time.year)
mn = int(event.origins[0].time.month)
dd = int(event.origins[0].time.day)
hh = int(event.origins[0].time.hour)
mm = int(event.origins[0].time.minute)
ss = int(event.origins[0].time.second)

# Convert to correct date formats for labelling
origin_time = obspy.UTCDateTime(yyyy,mn,dd,hh,mm,ss)

if evdp < 40 or evdp > 350 or evlo > -64:
    print(evdp, evla, 'ending')
    sys.exit()

def add_zeroes(mn):
    if len(str(mn)) == 2:
        pass
    else:
        mn = '0'+ str(mn)
    return mn

mn_0 = add_zeroes(mn)
dd_0 = add_zeroes(dd)
hh_0 = add_zeroes(hh)
mm_0 = add_zeroes(mm)
ss_0 = add_zeroes(ss)

#evname_obspyDMT = str(yyyy)+str(mn)+str(dd)+'_'+str(hh)+str(mm)+str(ss)
event_name = str(yyyy)+str(mn_0)+str(dd_0)+str(hh_0)+str(mm_0)+str(ss_0)
print("Event names is:",event_name)

#------------------------------------------------------------------------------

# Define paths to directories and relocation code outputs file
gen_dir = '/nobackup/ee18ab/Rewritten_pmP_Code_Jan_2025'
res_dir = '/nobackup/ee18ab/uol_scripts/Results'

#constant_vels = [5.9,6.0,6.1,6.2]
constant_vels=[5.9]

f = open('total_pP_arrays_eventID.txt', 'a+')
for a in range (len(constant_vels)):
    try:
        ev_dir = os.path.join(res_dir, '%s' %event_name)                                       
        outputs = np.load(os.path.join(ev_dir, 'array_Z.npy'), allow_pickle=True)
        output_dir = os.path.join(ev_dir, 'pmP_constant_vel_%s' %constant_vels[a])

        if reprocess == True:
            # move the velocity models folder into each event directory to prevent errors when building velocity models (needed for running multiple events at once on arc)
            vel_model_folder = os.path.join(gen_dir, 'Velocity_model_files')

            source_folder = vel_model_folder
            destination_folder = os.path.join(output_dir, 'Velocity_model_files')

            if os.path.exists(destination_folder):
                shutil.rmtree(destination_folder)
                
            shutil.copytree(source_folder, destination_folder)
            vel_models = destination_folder

            # create folder for figures
            fig_dir = os.path.join(output_dir, 'Crust_code_figures')   
            try: 
                os.mkdir(fig_dir)
            except FileExistsError:
                pass

            # Load in event depth from relocation results file

            all_EQ_outputs = '/nobackup/ee18ab/uol_scripts/Results/Final_Andes_Catalogue_with_S.csv'  # UPDATE TO ISCloc FINAL CATALOGUE??                 

            with open(all_EQ_outputs, 'r') as file:
                for line in file:
                    if event_id in line:
                        ev_row = line.strip().split('\t')
                        evdepth = float(ev_row[10])
                        evdepth_exists = True
                        print("Event depth:", evdepth)
                        break
                    else:
                        evdepth_exists = False
                        
            if evdepth_exists == False:
                print("No relocated event depth available.")
                evdepth = evdp # original catalogue depth used
                #sys.exit()
                     

   

        # ---------- CREATE LIST OF Cr_subarray CLASS OBJECTS ----------

        test_subarrays_list = []
        for i in range(len(outputs)):
            test_subarray = Cr_subarray(outputs[i], evdepth)
            #print(type(outputs[i].binned_stream), outputs[i].binned_stream)
            test_subarrays_list.append(test_subarray)
        
        # ---------- CHECK THAT THE SUBARRAYS TO BE TESTED HAVE PASSED ALL CLEANING STEPS -----------

        # The following code checks that the subarrays on the test_subarrays_list are also present in the final clean outputs file by searching for the subarray-event distance value in the text file (works as a subarray signature of sorts).
        # Since it is (currently) searching only the pP picks file, the code also ensures that a pP signal has definitely been picked for these subarrays. 

        cleaned_pP = os.path.join(ev_dir, 'outputs_cleaned_pP.txt')                      

        subarray_clean = [False] * len(test_subarrays_list)

        with open(cleaned_pP, 'r') as file:
            for line in file:
                for i in range(len(test_subarrays_list)):
                    if str(test_subarrays_list[i].outputs.ev_array_gcarc) in line:
                         subarray_clean[i] = True
                                   
        test_subarrays_cleaned = []
        for i in range(len(subarray_clean)):
            if subarray_clean[i] == True:
                test_subarrays_cleaned.append(test_subarrays_list[i])
            else:
                pass
            
        test_subarrays_list = test_subarrays_cleaned

        assert len(test_subarrays_list) > 0

        #f.write(str(event_name) + '\t' + str(len(test_subarrays_list)) + '\n') # FINDING TOTAL NO. TESTED SUBARRAYS
        f.write(str(event_id) + '\t' + str(len(test_subarrays_list)) + '\n')
    except Exception as e:
        print(e)
f.close()
