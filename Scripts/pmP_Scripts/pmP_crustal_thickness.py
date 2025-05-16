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
import pickle
from obspy.taup import TauPyModel

import obspy
from obspy import read_events
import matplotlib.pyplot as plt

# Importing classes
from crust1 import crustModel # NEED TO CHANGE FILE PATHS IN crust1.py
from Classes_for_crthk_code import Cr_subarray

# Importing functions

from Functions_for_crthk_code import Get_crustal_thickness
from Functions_for_crthk_code import Build_velocity_model
from Functions_for_crthk_code import Crustal_thickness_forward_modelling
from Functions_for_crthk_code import Plot_bounce_points
from Functions_for_crthk_code import Plot_delay_time_histogram
from Functions_for_crthk_code import Plot_crthk_histogram
from Functions_for_crthk_code import Plot_all_event_data
from Functions_for_crthk_code import write_out_results

cur_dir = os.path.dirname(os.path.abspath(__file__))
par_dir = os.path.abspath(os.path.join(cur_dir, '..'))
sys.path.append(par_dir)
import classes

#------------------------------------------------------------------------------

def determine_crustal_thickness(catalogue, event, gen_dir, res_dir, reprocess, make_figures, plot_velocity_models, include_sea, final_EQ_cat_txt=False, depth=False):

    # Load in event data
    input_no = event

    event = catalogue[int(input_no)-1]
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
    #gen_dir = '/users/ee18ab/Relocation_Scripts/Rewritten_pmP_Code_Jan_2025'
    #res_dir = '/users/ee18ab/Relocation_Scripts/Results'

    # Constant crustal velocity
    constant_vels=[5.9]

    for a in range (len(constant_vels)):
        try:
            ev_dir = os.path.join(res_dir, '%s' %event_name)                                       
            outputs = np.load(os.path.join(ev_dir, 'array_Z.npy'), allow_pickle=True)
            output_dir = os.path.join(ev_dir, 'pmP_outputs_%s' %constant_vels[a])

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
                evdepth_exists = False
                if  final_EQ_cat_txt != False:
                    all_EQ_outputs = final_EQ_cat_txt  # ISCloc FINAL CATALOGUE               

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
                
                elif depth != False:
                    evdepth = depth
                    evdepth_exist = True
                            
                if evdepth_exists == False:
                    print("No relocated event depth available.")
                    evdepth = evdp # original catalogue depth used
                    #sys.exit()
                         

                # Set up summary text file for this script

                txt_file_name = 'Crthk_code_Summary.txt'
                outputs_txt = os.path.join(output_dir, txt_file_name)
                f = open(outputs_txt, 'w')
                f.write('Event name: ' + str(event_name) + '\n')
                f.write('Event depth (relocated): ' + str(evdepth) + ' km' + '\n')
                f.write('Event magnitude: ' + str(evmag) + '\n')
                f.write("Event latitude: " + str(evla) + "\n" + "Event longitude: " + str(evlo))
                f.write('\n')
                f.close()

                print("Event latitude:", evla, "\nEvent longitude:", evlo)

                # ---------- CREATE LIST OF Cr_subarray CLASS OBJECTS ----------

                test_subarrays_list = []
                for i in range(len(outputs)):
                    test_subarray = Cr_subarray(outputs[i], evdepth)
                    #print(type(outputs[i].binned_stream), outputs[i].binned_stream)
                    test_subarrays_list.append(test_subarray)
                
                # ---------- CHECK THAT THE SUBARRAYS TO BE TESTED HAVE PASSED ALL CLEANING STEPS -----------

                # The following code checks that the subarrays on the test_subarrays_list are also present in the final clean outputs file by searching for the subarray-event distance value in the text file (works as a subarray signature of sorts).
                # Since it is (currently) searching only the pP picks file, the code also ensures that a pP signal has definitely been picked for these subarrays. 

                cleaned_pP = os.path.join(ev_dir, 'outputs_cleaned_pP.txt')   # UPDATE TO ISCloc SURVIVORS??  No, will lose many data points...use cleaned outputs from 1D processing to remove outliers                 

                subarray_clean = [False] * len(test_subarrays_list)

                try:
                    with open(cleaned_pP, 'r') as file:
                        for line in file:
                            for i in range(len(test_subarrays_list)):
                                if str(test_subarrays_list[i].outputs.ev_array_gcarc) in line:
                                     subarray_clean[i] = True
                except:
                    print('No pP-P pairs from 1D relocation process')
                    sys.exit()
                                           
                test_subarrays_cleaned = []
                for i in range(len(subarray_clean)):
                    if subarray_clean[i] == True:
                        test_subarrays_cleaned.append(test_subarrays_list[i])
                    else:
                        pass
                    
                test_subarrays_list = test_subarrays_cleaned

                f = open(outputs_txt, 'a+')
                f.write("No of subarrays to enter pmP search: " + str(len(test_subarrays_list)))
                f.write('\n')
                f.close()

                assert len(test_subarrays_list) > 0, "No ad-hoc array candidates to check for pmP."
                       
                # ---------- EXTRACT FIRST-PASS CRUSTAL THICKNESS AT EVENT COORDINATES ----------

                crthk = Get_crustal_thickness(evla, evlo, event_name, vel_models)
                print("First-pass crustal thickness at event coordinates is ", crthk, " km")

                if crthk > evdepth:
                    print("Error: Crust1 thickness is greater than event depth.")
                    f = open(outputs_txt, 'a+')
                    f.write("First-pass crustal thickness is greater than event depth.")
                    f.write('\n')
                    f.close()

                # -----    
                f = open(outputs_txt, 'a+')
                f.write("First-pass crustal thickness at event coordinates extracted from Crust1: " + str(crthk) + " km")
                f.write('\n')
                f.close()

                # ----- Extract Crust 1.0 model (velocity) values for epicentre location, use crustal averages and merge onto ak135f ------
                
                crust1 = crustModel()
                model_at_epicentre = crust1.get_point(evla, evlo) # vp, vs, density, layer thickness, and the top of the layer with respect to sea level.
                #print(model_at_epicentre)   

                # Is there a sea layer?
                layers = list(model_at_epicentre.keys())  
                #print(layers)  
                if 'water' in layers:
                    sea = True
                else:
                    sea = False
                
                if sea == True and include_sea == True:
                    
                    # Make Crust 1.0 velocity model with sea layer
                    model_layers = [val for val in model_at_epicentre.values()]
                    #print(model_layers)
                    
                    p_vel = []
                    s_vel = []
                    density = []
                    thickness = []
                        
                    for i in range (len(model_at_epicentre)):
                        p_vel.append(model_layers[i][0])  
                        s_vel.append(model_layers[i][1]) 
                        density.append(model_layers[i][2]) 
                        thickness.append(model_layers[i][3])
                        
                    mean_p_vel = np.average(p_vel)
                    mean_s_vel = np.average(s_vel)
                    mean_density = np.average(density)           
                    
                    f = open(destination_folder + '/crust1_average_withsea.txt', 'w')
                    f.write('0.00,' + str(mean_p_vel) + ',' + str(mean_s_vel) + ',' + str(mean_density) + '\n')
                    #f.write('1.00,' + str(mean_p_vel) + ',' + str(mean_s_vel) + ',' + str(mean_density) + '\n')
                    f.write('0.00,' + str(mean_p_vel) + ',' + str(mean_s_vel) + ',' + str(mean_density) + '\n')
                    f.write('0.00,0.00,0.00,0.00' + '\n')  #ak135f values
                    f.close()
                    
                    crthk = float(model_layers[-1][4])
                    crustal_velocity_model = destination_folder + '/crust1_average_withsea.txt'
                    
                
                if sea == True and include_sea == False:
                
                    model_layers = [val for val in model_at_epicentre.values()]
                    
                    p_vel = []
                    s_vel = []
                    density = []
                    thickness = []
                        
                    for i in range (1,len(model_at_epicentre)):
                        p_vel.append(model_layers[i][0])  
                        s_vel.append(model_layers[i][1])
                        density.append(model_layers[i][2]) 
                        thickness.append(model_layers[i][3])
                        
                    mean_p_vel = np.average(p_vel)
                    mean_s_vel = np.average(s_vel)
                    mean_density = np.average(density)           
                    
                    f = open(destination_folder + '/crust1_average_nosea.txt', 'w')
                    f.write('0.00,' + str(mean_p_vel) + ',' + str(mean_s_vel) + ',' + str(mean_density) + '\n')
                    #f.write('1.00,' + str(mean_p_vel) + ',' + str(mean_s_vel) + ',' + str(mean_density) + '\n')
                    f.write('0.00,' + str(mean_p_vel) + ',' + str(mean_s_vel) + ',' + str(mean_density) + '\n')
                    f.write('0.00,0.00,0.00,0.00' + '\n') #ak135f values
                    f.close()
                    
                    crthk = float(abs(model_layers[-1][4])-abs(model_layers[1][4]))
                    crustal_velocity_model = destination_folder + '/crust1_average_nosea.txt'
                    
                if sea == False:
                    
                    model_layers = [val for val in model_at_epicentre.values()]
                    
                    # make Crust 1.0 velocity model witout a sea layer
                    
                    # Check for above sea-level sediments
                    if model_layers[0][4] > 0:
                        layer_depth_correction = -1*model_layers[0][4]
                        print('layer depth correction:', layer_depth_correction)
                    
                    p_vel = []
                    s_vel = []
                    density = []
                    thickness = []
                        
                    for i in range (len(model_at_epicentre)):
                        p_vel.append(model_layers[i][0])  
                        s_vel.append(model_layers[i][1])  
                        density.append(model_layers[i][2])  
                        thickness.append(model_layers[i][3])
                        
                    mean_p_vel = np.mean(p_vel)
                    mean_s_vel = np.mean(s_vel)
                    mean_density = np.mean(density)           
                    
                    f = open(destination_folder + '/crust1_average_nosea.txt', 'w')
                    f.write('0.00,' + str(constant_vels[a]) + ',' + str(mean_s_vel) + ',' + str(mean_density) + '\n')
                    #f.write('1.00,' + str(mean_p_vel) + ',' + str(mean_s_vel) + ',' + str(mean_density) + '\n')
                    f.write('0.00,' + str(constant_vels[a]) + ',' + str(mean_s_vel) + ',' + str(mean_density) + '\n')
                    f.write('0.00,0.00,0.00,0.00' + '\n')  #ak135f values
                    f.close()
                    
                    crthk = float(abs(model_layers[-1][4]+layer_depth_correction))
                    crustal_velocity_model = destination_folder + '/crust1_average_nosea.txt'
                    

                # ---------- CALCULATE PREDICTED ARRIVALS FOR PHASES OF INTEREST ----------
                 
                custom_model = Build_velocity_model(crthk, event_name, crustal_velocity_model, vel_models)
                
                for subarray in test_subarrays_list:
                    subarray.predict_arrivals(custom_model, crthk)

                # ---------- CALCULATE pmP ARRIVAL RELATIVE TO ALREADY IDENTIFIED DEPTH PHASES ----------

                for subarray in test_subarrays_list:
                    subarray.predict_pmP_arrival_relative_to_DP() 
                    subarray.predict_smP_arrival_relative_to_DP()
                print('pP-pmP diff', subarray.pmP_P_diff_m)
                
                        
                # ---------- FIND pmP AND pP BOUNCE POINTS FOR EVENT-SUBARRAY PAIRS USING FIRST-PASS CRUSTAL THICKNESS VALUES ----------

                for subarray in test_subarrays_list:
                    subarray.get_model_pierce_points(crthk, custom_model)
                    
                # ---------- SEARCH FOR PEAKS IN A WINDOW AROUND THE PREDICTED pmP ARRIVAL ----------

                for subarray in test_subarrays_list:
                    subarray.find_pmP_picks_window()
                    
                # ---------- CREATE WINDOW SEARCH PLOTS, SAVE FIGURES ----------

                if make_figures == True:
                    for subarray in test_subarrays_list:
                        #print('passing HR figure')
                        subarray.plot_pmP_window_picking_figure(fig_dir)
                else:
                    pass

                # ---------- DISCARD SUBARRAYS WITH NO WINDOW PICKS ----------

                test_subarrays_with_window_picks = []
                test_subarrays_without_window_picks = []

                for subarray in test_subarrays_list:
                    if subarray.pmP_window_picks[0] != [0]:
                        test_subarrays_with_window_picks.append(subarray)
                        print("Signal(s) detected within the search window for subarray ", subarray.outputs.array_no, ". Proceeding to signal-to-noise check.")
                    else:
                        test_subarrays_without_window_picks.append(subarray)
                        print("No signals detected within the search window for subarray ", subarray.outputs.array_no, ". Remove from further processing.")

                # -----
                f = open(outputs_txt, 'a+')
                f.write("\n" + "No of subarrays without any peaks within the search window: " + str(len(test_subarrays_without_window_picks)) + '\n')
                f.write("No of subarrays with peaks within the search window: " + str(len(test_subarrays_with_window_picks)))
                f.write('\n')
                f.close()
                # -----
                                 
                # ---------- CHECK SIGNAL-TO-NOISE RATIO ----------

                for subarray in test_subarrays_with_window_picks:
                    subarray.signal_to_noise_check()

                # ---------- SORT SUBARRAYS BASED ON THE NUMBER OF PICKS THAT PASSED SIGNAL-TO-NOISE FILTER ----------

                test_subarrays_with_multiple_snr_filtered_picks = []
                test_subarrays_with_one_snr_filtered_pick = []
                test_subarrays_without_snr_filtered_picks = []

                for subarray in test_subarrays_with_window_picks:
                    if len(subarray.snr_filtered_picks) > 1:
                        test_subarrays_with_multiple_snr_filtered_picks.append(subarray)
                        #print("Multiple picks passed the signal-to-noise filter for subarray ", subarray.outputs.array_no)
                    elif len(subarray.snr_filtered_picks) == 1:
                        test_subarrays_with_one_snr_filtered_pick.append(subarray)
                        #print("Only one pick passed the signal-to-noise filter, ending pmP search for subarray ", subarray.outputs.array_no)
                    else:
                        test_subarrays_without_snr_filtered_picks.append(subarray)
                        #print("No suitable pmP picks were identified for subarray ", subarray.outputs.array_no)

                # -----        
                f = open(outputs_txt, 'a+')
                f.write("\n" + "No of subarrays with no picks above the snr threshold: " + str(len(test_subarrays_without_snr_filtered_picks)))
                f.write("\n" + "No of subarrays with one pick above the snr threshold: " + str(len(test_subarrays_with_one_snr_filtered_pick)))
                f.write("\n" + "No of subarrays with multiple picks above the snr threshold: " + str(len(test_subarrays_with_multiple_snr_filtered_picks)))
                f.write('\n')
                f.close()
                # -----

                # -----------------------------------------------------------------------------        
                        
                print(len(test_subarrays_list), " subarrays entered pmP search.")
                print(len(test_subarrays_without_window_picks), " subarray(s) had no picks within the search window.")
                print(len(test_subarrays_without_snr_filtered_picks), " subarray(s) had no window picks above the signal-to-noise filtering threshold.")
                print(len(test_subarrays_with_one_snr_filtered_pick), " subarray(s) had one window pick above the signal-to-noise filtering threshold.")
                print(len(test_subarrays_with_multiple_snr_filtered_picks), " subarray(s) had multiple window picks above the signal-to-noise filtering threshold.")

                print("Subarray(s) with one snr-filtered pick:")
                for subarray in test_subarrays_with_one_snr_filtered_pick:
                    print(subarray.outputs.ev_array_gcarc)

                print("Subarray(s) with multiple picks:")
                for subarray in test_subarrays_with_multiple_snr_filtered_picks:
                    print(subarray.outputs.ev_array_gcarc)   

                # -----------------------------------------------------------------------------
                    
                # ---------- STOP CODE HERE IF NO pmP PICKS WERE DETECTED ----------

                subarrays_with_pmP_picks = []
                for subarray in test_subarrays_list:
                    if subarray.pmP_pick_exists == True:
                        subarray.extract_amplitudes()
                        subarray.extract_amplitudes_PW()
                        subarrays_with_pmP_picks.append(subarray)
                    
                if len(subarrays_with_pmP_picks) == 0:
                    print("No pmP picks returned for event ", event_name)
                    f = open(outputs_txt, 'a+')
                    f.write("\n" + "No subarrays with pmP picks returned.")
                    f.write('\n')
                    f.close()
                    
                assert len(subarrays_with_pmP_picks) > 0, "No candidate ad-hoc arrays with pmP picks."
                      
                # ---------- CALCULATE CRUSTAL THICKNESSES FROM pmP-pP DELAY TIME USING FORWARD MODELLING APPROACH ---------
                    
                Crustal_thickness_forward_modelling(subarrays_with_pmP_picks, crthk, evdepth, event_name, crustal_velocity_model, vel_models)

                print("Crustal thickness values from forward modelling:")
                for subarray in subarrays_with_pmP_picks:
                    print(subarray.outputs.ev_array_gcarc,": crustal thickness", subarray.new_crthk_FM, "km")

                # -----            
                f = open(outputs_txt, 'a+')
                f.write("\n" + "Crustal thickness values from FM for subarrays with one snr filtered pick: " + "\n")
                f.write("\n" + "Epicentral Distance (degrees), New crustal Thickness (km) " + "\n")
                for subarray in subarrays_with_pmP_picks:
                    f.write(str(subarray.outputs.ev_array_gcarc) + " " + str(subarray.new_crthk_FM) + " km" + "\n")
                f.write('\n')
                f.close()    
                # -----
                    
                # ---------- CALCULATE CRUSTAL THICKNESSES FROM pmP-DP DELAY TIME USING RAY PARAMETER ----------
                    
                #for subarray in test_subarrays_with_one_snr_filtered_pick:
                #    subarray.calculate_crustal_thickness_ray_param()
                    
                # ---------- CALCULATE NEW pmP and pP BOUNCE POINTS FOR EVENT-SUBARRAY PAIRS WITH A pmP PICK ----------

                for subarray in subarrays_with_pmP_picks:    
                    subarray.get_improved_pierce_points(crustal_velocity_model, vel_models)

                # ---------- PLOT BOUNCE POINTS FOR SUBARRAYS WITH pmP PICKS ONLY ----------

                '''if make_figures == True:    
                    Plot_bounce_points(subarray_list = subarrays_with_pmP_picks, event_name = event_name, evla = evla, evlo = evlo, first_pass_crthk = crthk, fig_dir=fig_dir, gen_dir=gen_dir)'''

                # ---------- PLOT HISTOGRAM OF pmP-pP DELAY TIMES PICKED FROM THE DATA -----------

                if make_figures == True:
                    Plot_delay_time_histogram(subarray_list = subarrays_with_pmP_picks, event_name = event_name, fig_dir=fig_dir)

                # ---------- PLOT HISTOGRAM OF CRUSTAL THICKNESS VALUES CALCULATED FROM THE DATA ----------

                if make_figures == True:
                    Plot_crthk_histogram(subarray_list = subarrays_with_pmP_picks, event_name = event_name, first_pass_crthk = crthk, fig_dir=fig_dir)

                # ---------- CREATE A MAP WITH ALL BOUNCE POINTS, CALCULATED CRUSTAL THICKNESS VALUES, AND RECEIVER FUNCTION DATA -----------

                '''if make_figures == True:
                    Plot_all_event_data(subarray_list = test_subarrays_list, event_name = event_name, evla = evla, evlo = evlo, first_pass_crthk = crthk, fig_dir=fig_dir, gen_dir=gen_dir)'''

                # ---------- SAVE ALL SUBARRAYS WITH A pmP PICK INTO FILE ----------
                # subarrays with pmP picks
                successful_subarrays_path = os.path.join(output_dir, 'successful_Cr_subarrays.npy')

                if os.path.exists(successful_subarrays_path):
                    os.remove(successful_subarrays_path)
                    
                np.save(successful_subarrays_path, subarrays_with_pmP_picks, allow_pickle=True)
                
                # all tested subarrays
                tested_subarrays_path = os.path.join(output_dir, 'tested_Cr_subarrays.npy')

                if os.path.exists(tested_subarrays_path):
                    os.remove(tested_subarrays_path)
                    
                np.save(tested_subarrays_path, test_subarrays_list, allow_pickle=True)

                # ----------- WRITE OUT RESULTS WITH pmP PICKS -----------

                write_out_results(test_subarrays_list, event_name, event_id, res_dir, '/Final_pmP_catalogue_%s.txt' %constant_vels[a])

                          #==================================================================================================================================

            # Plotting extra bits and pieces for paper

            # Load in .npy if only plotting results
            if reprocess == False:
                successful_subarrays_path = os.path.join(output_dir, 'successful_Cr_subarrays.npy')
                subarrays_with_pmP_picks = np.load(successful_subarrays_path, allow_pickle=True)
                

                for subarray in subarrays_with_pmP_picks:        
                        outputs = subarray.outputs
                        envelope = subarray.outputs.PW_optimum_beam_envelope
                        
                        DP_picks = subarray.outputs.phase_id_picks

                        P_index = DP_picks[0]
                        pmP_index = subarray.pmP_pick
                        pP_index = DP_picks[1]
                        
                        P_amplitude = envelope[int(P_index)]
                        pmP_amplitude = envelope[int(pmP_index)]
                        pP_amplitude = envelope[int(pP_index)]
                        
                        subarray.P_pw_amplitude = P_amplitude
                        subarray.pmP_pw_amplitude = pmP_amplitude
                        subarray.pP_pw_amplitude = pP_amplitude

                # re-write out summary text files (optional)
                write_out_results(subarrays_with_pmP_picks, event_name, event_id, res_dir, '/Final_pmP_catalogue_%s.txt' %constant_vels[a])

            if make_figures == True:
                for subarray in subarrays_with_pmP_picks:
                    subarray.plot_pmP_window_picking_figure_paper(fig_dir)
            else:
                pass

            # testing average width of the P arrival to determine a window size   
            f = open(output_dir + '/P_lamda.txt', 'w')
            for subarray in test_subarrays_list:
                f.write(str(subarray.P_lamda) + '\n')
            f.close()

            if plot_velocity_models == True:
                depth, pvel, svel, density = np.loadtxt(output_dir+'/Velocity_model_files/ak135.tvel', unpack=True, usecols=(0,1,2,3), skiprows=2)
                mdepth, mpvel, msvel, mdensity = np.loadtxt(output_dir+'/Velocity_model_files/ak135_modified.tvel', unpack=True, usecols=(0,1,2,3), skiprows=2)

                crthk = test_subarrays_list[-1].new_crthk_FM

                #print(depth)
                fig,ax = plt.subplots(1,2, figsize=(10,7))
                ax[0].plot(mpvel,-mdepth, label='Modified - P', color='cyan')
                ax[0].plot(msvel,-mdepth, label='Modified - S', color='orange')
                ax[0].plot(pvel,-depth, label='ak135 - P', linestyle='-.', color='r')
                ax[0].plot(svel,-depth, label='ak135 - S', linestyle='-.', color='b')
                ax[0].axhline(-crthk, color='k', linestyle=':', label='Crust 1.0 Moho')
                ax[0].set_ylim(-(np.max(depth)),0)
                leg = ax[0].legend(loc='lower left')
                leg.get_frame().set_facecolor('white')
                ax[0].set_xlabel('Velocity (km/s)')
                ax[0].set_ylabel('Depth (km)')
                
                ax[1].plot(mpvel,-mdepth, label='Modified - P', color='cyan')
                ax[1].plot(msvel,-mdepth, label='Modified - S', color='orange')
                ax[1].plot(pvel,-depth, label='ak135 - P', linestyle='-.', color='r')
                ax[1].plot(svel,-depth, label='ak135 - S', color='b', linestyle='-.')
                ax[1].axhline(-crthk, color='k', linestyle=':', label='Crust 1.0 Moho')
                ax[1].set_ylim(-100, 0)
                ax[1].set_xlim(3,9)
                leg = ax[1].legend(loc='lower left')
                leg.get_frame().set_facecolor('white')
                ax[1].set_xlabel('Velocity (km/s)')
                #ax[1].set_ylabel('Depth (km)')
                plt.savefig(output_dir+'/Velocity_model_files/Velocity_models.png')
                plt.close()
            
            
        except Exception as e:
            print('Event %s failed' %event_name)
            print(e, ' Error on line {}'.format(sys.exc_info()[-1].tb_lineno))
            txt_file_name = 'Failed_Events_pmP.txt'
            failed_txt = os.path.join(res_dir, txt_file_name)
            print(failed_txt)
            f = open(failed_txt, 'a+')
            f.write(str(event_name) + '\t' + str(e) + ' Error on line {}'.format(sys.exc_info()[-1].tb_lineno) + '\n')
            f.close()
            pass

