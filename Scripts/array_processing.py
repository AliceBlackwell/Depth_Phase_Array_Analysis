#!/usr/bin/env python3

'''
Script to relocate intermediate-depth eathquakes automatically in depth.
Written by Alice Blackwell.
Date: 24th September 2024

Loads in pre-processed teleseismic ZNE components for an event, creates ad-hoc arrays, applies to array processing and automatically picks depth phases. 

Depth conversion can be forward modelled using depth phase to primary phase differential times, or using an external phase based approach (e.g. ISCloc).

pre-requisites:
+ ObspyDMT catalogue of events to relocate [script 0]
+ Processed data associated with target events (Z,N,E) [script 1 & 1S]
'''

# Import modules
import math
import obspy
import os
import glob
import time
import sys 
import numpy as np
import pandas as pd
import pickle
import re

from obspy import taup
from obspy.taup.taup_create import build_taup_model
from obspy.taup import TauPyModel
from obspy.core.stream import Stream

from sklearn.neighbors import BallTree
from sklearn.cluster import DBSCAN
from obspy.core import UTCDateTime
from pathlib import Path

from copy import deepcopy

import matplotlib.pyplot as plt

from string import ascii_uppercase
import itertools

# Alice's Scripts
from classes import Earthquake_event, Stations, Array, Global, ISCloc

#======================================================================

def run_array_processing(catalogue, event, results_dir, data_dir, component, do_array_processing, depth_conversion, iscloc):

    #--------- Set Event ---------------

    # Define event catalogue (ObspyDMT file structure), and processed seismic data locations
    #catalogue_name = '/users/ee18ab/Relocation_Scripts/20100523_Example'

    # Load event from ObspyDMT catalogue
    #cat_file = catalogue_name + '/EVENTS-INFO/catalog.ml.pkl'

    #with open(cat_file, 'rb') as f:
    #    catalogue = pickle.load(f)
    
    inputs = event
        
    # Select event from catalogue and define attributes
    event = Earthquake_event(catalogue[inputs-1])
    event.define_event()

    print("Event names is:",event.evname)


    #--------- Set Variables ------------

    #Set folder pathways
    results_parent_dir = results_dir
    data_dir = data_dir + '/' + event.evname 

    # ***Choose seismic component***
    #component = 'ZNE'   # string: 'Z' or 'ZNE'

    # Choose whether to make and pick array vespagrams
    #do_array_processing = False
    #depth_conversion = False
    #iscloc = True

    # Define some processing variables       
    rsample = 10
    tpratio = 0.05
    mu = 2.0 # phase weighting, not currently used        
    frqmin = 1/10 #P
    frqmax = 1/1 #P, 0.03-0.2Hz for S
    datatype = 'Vel'

    # Define velocity model:
    vel_model = taup.TauPyModel(model='ak135')

    # Beampacking/Vespagram Analysis (not directly used - just a reminder)
    slow = np.arange(0.03, 0.111, 0.001)
    back = np.arange(0, 361, 1)

    # Define station nomenclature
    def iter_all_strings():
        for size in itertools.count(1):
            for s in itertools.product(ascii_uppercase, repeat=size):
                yield "".join(s)

    sta_no = []
    for s in itertools.islice(iter_all_strings(), 10000):
        sta_no.append(s)

    sta_name = sta_no[inputs-1]
            
    # --------- Set up Directories -----------

    # Create results directory (if it doesn't exist)
    try:
        directory = '%s' %results_parent_dir
        os.mkdir(directory)
        print('Directory %s created' %directory )

    except FileExistsError:
        pass

    # Create results directory for event, within parent results directory
    try:
        # Create folder
        directory = '%s' %event.evname
        ev_dir = os.path.join(results_parent_dir, directory)
        os.mkdir(ev_dir)
        print('Directory %s created' %directory )
            
    except FileExistsError:
        directory = '%s' %event.evname
        ev_dir = os.path.join(results_parent_dir, directory)
        pass

    # ---------- Set up Summary txt File -------------
    txt_file_name = 'Script_2_Summary.txt'
    outputs_txt = os.path.join(ev_dir, txt_file_name)
    f = open(outputs_txt, 'a+')
    f.write('event name: ' + str(event.evname) + '\n')
    f.write('event depth: ' + str(event.evdp) + '\n')
    f.write('event magnitude: ' + str(event.evm) + '\n')
    f.write('\n')

    #=============== PROCESSING =====================
    if do_array_processing == True or (len(glob.glob(ev_dir + '/array_Z.npy'))==0 and component=='Z') or (len(glob.glob(ev_dir + '/array_T.npy'))==0 and component=='ZNE'):
        try:
            # Set depth conversion flag to True
            depth_conversion = True
            
            # ---------- Remove ad-hoc array files if they already exist ---------
            if os.path.isfile(ev_dir + '/array_stations_original.txt') == True:
                os.remove(ev_dir + '/array_stations_original.txt')
            if os.path.isfile(ev_dir + '/array_stations_post_x_corr.txt') == True:
                os.remove(ev_dir + '/array_stations_post_x_corr.txt')
            if os.path.isfile(ev_dir + '/array_stations_successful.txt') == True:
                os.remove(ev_dir + '/array_stations_successful.txt')
            if os.path.isfile(ev_dir + '/array_stations_successful_S.txt') == True:
                os.remove(ev_dir + '/array_stations_successful_S.txt')

            if component == 'Z':
                
                # Load in processed Z data
                stream = obspy.read(data_dir + "/Data/" + "*.MSEED")
                stream_Z = stream.select(component='Z')

            if component == 'ZNE':

                # Load in processed Z,N,E data
                stream = obspy.read(data_dir + "/Data/" + "*.MSEED")
                stream_N = stream.select(component='N')
                stream_E = stream.select(component='E')
                stream_Z = stream.select(component='Z')
                
                streamZNE = stream_N + stream_E + stream_Z
                
                f.write('Made ZNE stream' + '\n')
                
                # Create stream with only stations which have Z,N,E components
                ststring = [0] * len(streamZNE)
                for i in range(0, len(streamZNE)):
                    ststring[i] = streamZNE[i].stats.network + '.' + streamZNE[i].stats.station
                    #print('ID', ststring[i])
                
                stations = np.unique(ststring)
                #print(stations) 
                
                final_stream = Stream()
                for j in range (len(stations)):
                    st = streamZNE.select(network = stations[j][:2], station = stations[j][3:])
                    #print(st)
                    if len(st) == 3:
                        final_stream.extend(st)
                
                streamZNE = final_stream
                
                # Select one component to prevent multiple inputs per station
                #stream_Z = streamZNE.select(component='Z')


            # --------- Populate Station Attributes (using Z stream) ----------

            # Find attributes
            stations = Stations(stream_Z, event, data_dir)
            stations.get_station_attributes()

            # Create table of station network, name, longitude and latitude
            data= {
                'Station_Network': stations.stnet,
                'Station_Name': stations.stname,
                'Station_Longitude': stations.stlo,
                'Station_Latitude': stations.stla}

            df=pd.DataFrame(data)
            print(df)
            f.write('Stream Z dataframe' + '\n')
            f.write(str(df)) 
            f.write('\n')
            
        except Exception as e:
            print(e)
            txt_file_name = 'Failed_Events_from_Array_Processing.txt'
            failed_txt = os.path.join(results_parent_dir, txt_file_name)
            f = open(failed_txt, 'a+')
            f.write(str(event.evname) + '\t' + str(event.event_id) + '\t' + str('No Pre-Loaded and Processed Data available') + '\t' + 'Error on line {}'.format(sys.exc_info()[-1].tb_lineno) + '\n')
            f.close()
            raise Exception('No Pre-Loaded and Processed Data available')


        # --------- Record Time ----------
        TIME_S = time.time()
        TIME_Sp = time.ctime(TIME_S)
        print('Now processing event ', event.evname)
        print ('Start time is ', TIME_Sp)
        f.write('Start time is ' + str(TIME_Sp) + '\n')
        f.write('Beginning array processing' + '\n')
        f.write('\n')
        #f.close()

        # --------- Create Arrays (using Z stream) ----------
        min_array_diameter = 2.5
        min_stations = 10
        print_sub_arrays = 0

        try:
            f.write('Making arrays for Z data' + '\n')
            array_stream_Z, centroids, array_stlats, array_stlons = stations.make_arrays(min_array_diameter, min_stations, print_sub_arrays)
            np.save(data_dir + '/Arrays/array_stream_Z.npy', np.asarray(array_stream_Z, dtype=object), allow_pickle=True)
            np.save(data_dir + '/Arrays/array_centroids.npy', np.asarray(centroids, dtype=object), allow_pickle=True)
            np.save(data_dir + '/Arrays/array_stlats.npy', np.asarray(array_stlats, dtype=object), allow_pickle=True)
            np.save(data_dir + '/Arrays/array_stlons.npy', np.asarray(array_stlons, dtype=object), allow_pickle=True)
     
        except Exception as e:
            print('Event %s can not be processed' %event.evname)
            print(e)
            print('Arrays could not be created')
            txt_file_name = 'Failed_Events_from_Array_Processing.txt'
            failed_txt = os.path.join(results_parent_dir, txt_file_name)
            f = open(failed_txt, 'a+')
            f.write(str(event.evname) + '\t' + 'Arrays could not be created' + '\t' + str(e) + '\n')
            f.close()
            exit()

        if array_stream_Z == 0:
            print('Event %s can not be processed' %event.evname)
            print('Arrays could not be created, station distribution inappropiate')
            txt_file_name = 'Failed_Events_from_Array_Processing.txt'
            failed_txt = os.path.join(results_parent_dir, txt_file_name)
            f = open(failed_txt, 'a+')
            f.write(str(event.evname) + '\t' + 'Arrays could not be created, station distribution inappropiate' + '\n')
            f.close()
            exit()
        
        try:
            if component == 'ZNE': 
                # -------- Re-create arrays for N and E components, and for all components --------

                f.write('Making arrays for N & E data' + '\n')

                # Select one component per stream
                stream_N = streamZNE.select(component='N')
                stream_E = streamZNE.select(component='E')
                array_stream_N = stations.recreate_arrays_for_other_components(array_stream_Z, stream_N) # N component in array list
                array_stream_E = stations.recreate_arrays_for_other_components(array_stream_Z, stream_E) # E component in array list
                array_stream_ZNE = stations.recreate_arrays_for_other_components(array_stream_Z, streamZNE) # ZNE components in array list
                
                #streamT = obspy.read(data_dir + "/Data/" + "*T.MSEED")
                #array_stream_T = stations.recreate_arrays_for_other_components(array_stream_Z, streamT)
                #np.save('array_stream_T.npy', np.asarray(array_stream_T, dtype=object), allow_pickle=True) # for array plotting
                #sys.exit()
            # -------- Define trim parameters --------- 

            # Calculate theoretical arrival times for 30 and 90 degrees epicentral distances
            # Use arrival times at 30 & 90 degrees to determine the trimming parameters for the traces
            arrivals = vel_model.get_travel_times(source_depth_in_km=event.evdp, distance_in_degree=30, phase_list=["P"])
            arr=arrivals[0]
            P_time_30 = arr.time

            trim_start = math.floor(P_time_30)-40

            if component == 'Z':
                arrivals = vel_model.get_travel_times(source_depth_in_km=event.evdp, distance_in_degree=90, phase_list=["sP"])
                arr=arrivals[0]
                sP_time_90 = arr.time

                trim_end = math.ceil(sP_time_90)+40

            if component == 'ZNE':
                arrivals = vel_model.get_travel_times(source_depth_in_km=event.evdp, distance_in_degree=90, phase_list=["sS"])
                arr=arrivals[0]
                sS_time_90 = arr.time
                
                trim_end = math.ceil(sS_time_90)+40

            print('Trim times:', trim_start, trim_end)
            f.write('Doing TauP to calculate trim times' + '\n')
            f.write('Trim times:' + str(trim_start) + ' ' + str(trim_end) + '\n')


            # ========= RELOCATE ==========

            # Loop over each array, individually find backazimuth, slowness, create vespagrams and pick P/pP/sP/S/sS arrivals.

            # Find parameters for Z (and apply to N and E if using).

            # ------- Record Time --------
            TIME_sub_array_processing = time.time()
            TIME_sub_array_processingp = time.ctime(TIME_sub_array_processing)
            print('Starting beam processing:', TIME_sub_array_processingp)

            no_arrays = len(array_stream_Z) # to loop through!

            # Empty list to store array classes
            array_Z_list = []

            if component == 'ZNE':
                array_N_list = []
                array_E_list = []
                array_T_list = []

            # Will be repeated if traces are removed by the x-correlation quality check (compares each trace to the beam)
            repeat_loop = False
            trace_to_keep = []
            failed_counter = 0
            QC_failed_counter = 0
            
        except Exception as e:
            print('Event %s can not be processed' %event.evname)
            print(e)
            print('Array processing set-up failed')
            txt_file_name = 'Failed_Events_from_Array_Processing.txt'
            failed_txt = os.path.join(results_parent_dir, txt_file_name)
            f = open(failed_txt, 'a+')
            f.write(str(event.evname) + '\t' + 'Array processing set-up failed' + '\t' + str(e) + '\n')
            f.close()
            exit()

        #for a in range (0,5): 
        for a in range (0, no_arrays): 
            print('Array ' + str(a) + ' out of ' + str(no_arrays)) 
            f = open(outputs_txt, 'a+')
            f.write('Array ' + str(a) + ' out of ' + str(no_arrays) + '\n')
            
            try:   
                # ------------- Z ---------------
                # ------- Use Z data to find array parameters ---------
                
                f.write('Z stream' + '\n')
                
                # Set up class for array
                array_Z = Array(array_stream_Z[a], event, rsample, str(sta_name)+str(a))
                array_Z.array_to_vespagram(data_dir, ev_dir, vel_model, trim_start, trim_end, beampack=True)
                array_Z.output_array_stations(ev_dir, str(sta_name)+str(a), 'post_x_corr')
                array_Z.QC_vespagram(trim_start, trim_end, phases=['P','pP','sP'])
                array_Z.get_picks(trim_start, trim_end, phases=['P','pP','sP'])   
                array_Z.id_phases(trim_start, trim_end, phases=['P','pP','sP'])
                #array_Z.relative_to_absolute_conversion(trim_start, phases=['P','pP','sP'])
                
                # === PhaseNet Beam Prep ===
                #array_Z.create_PhaseNet_beams() # NOT USED
                #fig = array_Z.prep_PhaseNet_beams(ev_dir)
                #fig.savefig(ev_dir +'/phasenet_pw_colormesh_%s.png' %a, dpi=300) 
                #sys.exit()
                
                array_Z.finalise_peaks()
                print('Final Picks:', array_Z.phase_id_picks)
                print('Z component has been processed for Array %s' %str(sta_name)+str(a))
                
                # Save out Z array class
                if np.sum(array_Z.phase_id_picks) != 0: # only save out array if it has picks
                    array_Z_list.append(array_Z)
                    array_Z.output_array_stations(ev_dir, str(sta_name)+str(a), 'successful')
                    failed_counter_Z = failed_counter
                    QC_failed_counter_Z = QC_failed_counter
                    array_Z.array_statistics = [no_arrays, failed_counter_Z, QC_failed_counter_Z]
                print('____________________________')
                print()

                if component == 'ZNE':
                    '''# ------------- N ---------------
                    # Copy stream_Z class, and replace stream data with a different component to make stream_N and stream_E classes. Will retain attributes found for stream_Z.

                    f.write('N stream' + '\n')
                    array_N = Array(array_stream_N[a], event, rsample, str(sta_name)+str(a))
                    array_N.apply_Z_array_attributes(array_Z) 
                    array_N.array_to_vespagram(data_dir, ev_dir, vel_model, trim_start, trim_end, populate_array_metadata=False, beampack=False)
                    #array_N.QC_vespagram(trim_start, trim_end, phases=['S','sS'])
                    
                            
                    # ------------- E ---------------   
                    f.write('E stream' + '\n')
                    array_E = Array(array_stream_E[a], event, rsample, str(sta_name)+str(a))
                    array_E.apply_Z_array_attributes(array_Z) 
                    array_E.array_to_vespagram(data_dir, ev_dir, vel_model, trim_start, trim_end, populate_array_metadata=False, beampack=False)
                    #array_E.QC_vespagram(trim_start, trim_end, phases=['S','sS'])'''
                    
                    
                    # ------------- T ---------------
                    # Copy stream_Z class, and replace stream data with a different component to make stream_N and stream_E classes. Will retain attributes found for stream_Z.

                    f.write('T stream' + '\n')
                    array_T = Array(array_stream_ZNE[a], event, rsample, str(sta_name)+str(a))
                    array_T.apply_Z_array_attributes(array_Z) 
                    array_T.format_stream()
                    array_T.rotate_stream_to_transverse(data_dir, array_Z) 
                    array_T.array_to_vespagram(data_dir, ev_dir, vel_model, trim_start, trim_end, populate_array_metadata=False, beampack=True, phase='S')                       
                    array_T.QC_vespagram(trim_start, trim_end, phases=['S','sS'])              
                    array_T.get_picks(trim_start, trim_end, phases=['S','sS'])  
                    array_T.id_phases(trim_start, trim_end, phases=['S','sS'])
                    #array_T.relative_to_absolute_conversion(trim_start, phases=['S','sS'])
                    array_T.finalise_peaks()
                    print('Final Picks:', array_T.phase_id_picks)
                    print('T component has been processed for Array %s' %str(sta_name)+str(a))

                # --------- Plotting ZNE Vespagrams & Beams ---------
                f.write('Plotting array ' + str(sta_name)+str(a) + '\n')
                print('TRIM', trim_start, trim_end)
                
                '''fig = array_Z.plot_1_component(trim_start, trim_end, start_phase = 'P', end_phase = 'sP', picks=array_Z.phase_id_picks)
                fig.savefig(ev_dir +'/pw_beam_vespagram_Z_%s.png' %a, dpi=300)            
                
                if component == 'ZNE':
                    fig = Array.plot_3_components(array_Z, array_N, array_E, trim_start, trim_end, rsample)
                    fig.savefig(ev_dir +'/pw_beams_ZNE_%s.png' %a, dpi=300)
                    
                    fig = Array.plot_3_components(array_Z, array_N, array_E, trim_start, trim_end, rsample, beams=False, vespagrams=True)
                    fig.savefig(ev_dir +'/pw_vespagrams_ZNE_%s.png' %a, dpi=300)
                    
                    fig = array_T.plot_1_component(trim_start, trim_end, start_phase = 'S', end_phase = 'sS', picks=array_T.phase_id_picks)
                    fig.savefig(ev_dir +'/pw_beam_vespagram_T_%s.png' %a, dpi=300)'''

                # append array to list of array output classes
                f.write('Saving out array classes' + '\n')
                f.write('\n')
                
                if component == 'ZNE':
                    #array_N_list.append(array_N)
                    #array_E_list.append(array_E)
                    if np.sum(array_T.phase_id_picks) != 0: # only save out array if it has picks
                        array_T_list.append(array_T)
                        array_T.output_array_stations(ev_dir, str(sta_name)+str(a), 'successful_S')
                        array_T.array_statistics = [no_arrays, failed_counter-failed_counter_Z, QC_failed_counter-QC_failed_counter_Z]

                TIME_S = time.time()
                TIME_Sp = time.ctime(TIME_S)
                print ('Loop end time is ', TIME_Sp)
                f.write('Loop end time is ' + str(TIME_Sp) + '\n')
                f.write('\n')
            
            except RuntimeError as e: 
                print('loop %s failed due to poor vespagram quality' %str(sta_name)+str(a))
                print(e)
                QC_failed_counter += 1
                f = open(outputs_txt, 'a+')
                f.write('Array ' + str(sta_name)+str(a) + ' failed, corresponding error: ' + str(e) + '\n')
                f.close()
                continue
            except AssertionError as e: 
                print('loop %s failed due to array having less than 8 traces' %str(sta_name)+str(a))
                print(e)
                QC_failed_counter += 1
                f = open(outputs_txt, 'a+')
                f.write('Array ' + str(sta_name)+str(a) + ' failed, corresponding error: ' + str(e) + '\n')
                f.close()
                continue
            except ConnectionError as e: 
                print('loop %s failed due to another phase being close to sS' %str(sta_name)+str(a))
                print(e)
                QC_failed_counter += 1
                f = open(outputs_txt, 'a+')
                f.write('Array ' + str(sta_name)+str(a) + ' failed, another phase is close to sS: ' + str(e) + '\n')
                f.close()
                continue
            except Exception as e:
                print('loop %s failed' %str(sta_name)+str(a))
                print(e)
                print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno))
                failed_counter += 1
                print('Failed Counter:', failed_counter)
                failed_percentage = (failed_counter/no_arrays)*100
                print(failed_percentage)
                f = open(outputs_txt, 'a+')
                f.write('Array ' + str(sta_name)+str(a) + ' failed, corresponding error: ' + str(e)+ ' Error on line {}'.format(sys.exc_info()[-1].tb_lineno) + '\n')
                f.close()
                if failed_percentage > 50:
                    txt_file_name = 'Failed_Events_from_Array_Processing.txt'
                    failed_txt = os.path.join(results_parent_dir, txt_file_name)
                    f = open(failed_txt, 'a+')
                    f.write('No. of failed arrays = ' + str(failed_counter) + '\n')
                    f.write(str(evname) + '\t' + 'More than 50% of arrays failed' + '\t' + str(e) + '\t' + 'Error on line {}'.format(sys.exc_info()[-1].tb_lineno) + '\n')
                    raise Exception('No. of Failed Bins > 50%. Loop Terminated.')
                    break         
                continue      

        # ------ Save out Array classes (with beams) ------
        print('Saving out array classes as .npys' + '\n') 
        try:
            np.save(ev_dir + '/array_Z.npy', array_Z_list, allow_pickle=True)

            if component == 'ZNE':
                #np.save(ev_dir + '/array_N.npy', array_N_list, allow_pickle=True)
                #np.save(ev_dir + '/array_E.npy', array_E_list, allow_pickle=True)
                np.save(ev_dir + '/array_T.npy', array_T_list, allow_pickle=True)
        except Exception as e:
            print('Event %s can not be processed' %event.evname)
            print(e)
            print('Array saving failed')
            txt_file_name = 'Failed_Events_from_Array_Processing.txt'
            failed_txt = os.path.join(results_parent_dir, txt_file_name)
            f = open(failed_txt, 'a+')
            f.write(str(event.evname) + '\t' + 'Array saving failed' + '\t' + str(e) + '\n')
            f.close()
            exit()
    #sys.exit()
    # ===== DEPTH CONVERSION =====

    # -------- Note the TIME ----------
    TIME = time.time()
    TIME = time.ctime(TIME)
    print('Starting Depth Modelling:', TIME)
    f = open(outputs_txt, 'a+')
    f.write('Starting Depth Modelling:' + str(TIME) + '\n')
    f.write('\n')

    if iscloc == False:
        try:
            if depth_conversion == True or (len(glob.glob(ev_dir + '/all_arrays.npy'))==0):
                # -------- Consider Results from all Arrays -------
                
                # Set depth conversion flag to True
                depth_conversion = True
                
                # Load saved output arrays
                array_Z_list = np.load(ev_dir + '/array_Z.npy', allow_pickle=True)
                
                # If array_T_list is empty, revert to only array_Z relocation
                if component == 'ZNE':
                    array_T_list = np.load(ev_dir + '/array_T.npy', allow_pickle=True) 
                    if array_T_list.size > 0: 
                        all_arrays = Global(event, array_Z_list, array_T_list)
                    else:
                        component = 'Z'
                        all_arrays = Global(event, array_Z_list)
                else:
                    all_arrays = Global(event, array_Z_list)
                
                #all_arrays.remove_empty_arrays(component) # remove arrays with no picks, shouldn't be saved out anyway but just in case!
                all_arrays.create_metadata_dataframe(component)

                # Forward model depth from differential times
                all_arrays.forward_model_depth(vel_model, ['P','pP'])
                all_arrays.forward_model_depth(vel_model, ['P','sP'])
                all_arrays.find_cleaning_filter(phases = ['P','pP','sP'])
                all_arrays.forward_depth_modelling_P_coda()

                if component == 'ZNE':
                    all_arrays.forward_model_depth(vel_model, ['S','sS'])
                    all_arrays.find_cleaning_filter(phases = ['S','sS'])
                    all_arrays.forward_depth_modelling_P_S_coda()
                    
                # ------ Save out Global class ------
                #np.save(ev_dir + '/all_arrays.npy', all_arrays, allow_pickle=True)                
                with open(ev_dir + '/all_arrays.npy', 'wb') as outfile:
                    pickle.dump(all_arrays, outfile)

            # --------- Write out results ---------- nb. Jack knifing is not included here.
            #================================================
            # If event already has an entry into the final catalogue, break routine.
            if glob.glob(results_parent_dir + '/Final_1D_Catalogue.txt') and do_array_processing == False and depth_conversion == False:
                if str(event.origin_time)[:19] in open(results_parent_dir + '/Final_1D_Catalogue.txt').read():
                    print('Event already relocated')
                    #sys.exit()
                     
            #all_arrays = np.load(ev_dir + '/all_arrays.npy', allow_pickle=True)
            with open (ev_dir + '/all_arrays.npy', 'rb') as infile:
                all_arrays = pickle.load(infile)
            
            all_arrays.write_out_final_outputs(ev_dir, component)
            catalogue_name = 'Final_1D_Catalogue.txt'
            all_arrays.write_out_catalogue(results_parent_dir, catalogue_name, component)
            catalogue_name = 'Final_1D_Catalogue_detailed.txt'
            all_arrays.write_out_catalogue_detailed(results_parent_dir, catalogue_name, component)

        except Exception as e:
            #sys.exit()
            print(str(event.evname) + '\t' + str(event.event_id) + '\t' + str(e) + '\t' + 'Error on line {}'.format(sys.exc_info()[-1].tb_lineno) + '\n')
            txt_file_name = 'Failed_1D_Events.txt'
            failed_txt = os.path.join(results_parent_dir, txt_file_name)
            f = open(failed_txt, 'a+')
            f.write(str(event.evname) + '\t' + str(event.event_id) + '\t' + str(e) + '\t' + 'Error on line {}'.format(sys.exc_info()[-1].tb_lineno) + '\n')
            f.close()

    if iscloc == True:
        try:
            for subdir in ['ISCloc/inputs','ISCloc/stations', 'ISCloc/outputs']:
                path = Path(results_dir) / subdir
                path.mkdir(parents=True, exist_ok=True)

            iscloc_inputs_dir = results_dir + '/ISCloc/inputs'
            station_list_loc = results_dir + '/ISCloc/stations'

            phase_relocation = ISCloc()
            phase_relocation.load_event(event)
            
            if os.path.exists(iscloc_inputs_dir+"ISF2_" + str(phase_relocation.event.event_id) + ".dat"):
                pass
            else:
                phase_relocation.create_input_ISFs_ISC(iscloc_inputs_dir, event_id=phase_relocation.event.event_id, append=False)
            
            # Load in ad-hoc array metadata
            phase_relocation.load_array_metadata(results_parent_dir, phase_relocation.event.evname)
            
            # Extract P arrivals per real station in ad-hoc array, from ISF 2.1 files
            phase_relocation.extract_ISC_P_arrivals(iscloc_inputs_dir, event.event_id)
            if component == 'ZNE':
                phase_relocation.extract_ISC_S_arrivals(iscloc_inputs_dir, event.event_id) # Extract S picks
                if phase_relocation.array_ISF_S_stations == []:
                    print('No ISC reported S picks in array')
                    component = 'Z'            

            # Determine ad-hoc arrays failure rate (number without a manual P pick in the ISC catalogue)
            phase_relocation.determine_array_failure_rate_for_P(results_parent_dir)
            if component == 'ZNE':
                phase_relocation.determine_array_failure_rate_for_S(results_parent_dir) # For ISC catalogue S picks
                
            # Correct ISF/ISC P onsets to array centre, and median time (throw out stations with more than 2.5 degree coordinate difference)
            phase_relocation.correct_P_onsets_to_array_centre()
            #print(phase_relocation.P_ISF)
            if component == 'ZNE':
                phase_relocation.correct_S_onsets_to_array_centre()
                
            # Load saved output arrays
            array_Z_list = np.load(ev_dir + '/array_Z.npy', allow_pickle=True)
            phase_relocation.extract_peaks_in_utc(array_Z_list)
            phase_relocation.find_timeshift()
            
            if component == 'ZNE':
                array_T_list = np.load(ev_dir + '/array_T.npy', allow_pickle=True)
                phase_relocation.extract_S_peaks_in_utc(array_T_list, array_Z_list)
                phase_relocation.find_S_timeshift()
            
            # Create table of station network, name, longitude and latitude for array
            # Sorted dataframes
            if component == 'Z':
                data= {
                    'Array No': phase_relocation.array_names,
                    'Magnitude': phase_relocation.event.evm,
                    'Depth': phase_relocation.event.evdp,
                    'Dist': phase_relocation.array_gcarc,
                    'No_ISF_Phases': phase_relocation.no_used_ISF_P_phases,
                    'No_adhoc_arrays': len(phase_relocation.array_names),
                    'No_traces': phase_relocation.trace_no,
                    'P_ISF': phase_relocation.P_ISF,
                    'Peaks_AB': phase_relocation.peaks,
                    'Timeshift': phase_relocation.P_peak_diff}

            elif component == 'ZNE':
                data= {
                    'Array No': phase_relocation.array_names,
                    'Magnitude': phase_relocation.event.evm,
                    'Depth': phase_relocation.event.evdp,
                    'Dist': phase_relocation.array_gcarc,
                    'No_ISF_P_Phases': phase_relocation.no_used_ISF_P_phases,
                    'No_ISF_S_Phases': phase_relocation.no_used_ISF_S_phases,
                    'No_adhoc_arrays': len(phase_relocation.array_names),
                    'No_traces': phase_relocation.trace_no,
                    'P_ISF': phase_relocation.P_ISF,
                    'Peaks_AB': phase_relocation.peaks,
                    'Timeshift': phase_relocation.P_peak_diff,
                    'S_ISF': phase_relocation.S_ISF,
                    'S_Peaks_AB': phase_relocation.S_peaks,
                    'S_Timeshift': phase_relocation.S_peak_diff}
                    

            df=pd.DataFrame(data)
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print(df.sort_values(by=['Dist'], ascending=True))

            sta, dist, back, phase, times, lat, lon, elev = phase_relocation.apply_timeshift_to_peaks()
            if component =='ZNE':
                sta, dist, back, phase, times, lat, lon, elev = phase_relocation.apply_timeshift_to_S_peaks(sta, dist, back, phase, times, lat, lon, elev, use_P=False)

            if sta.size == 0:
                print('No outputs for %s' %event.event_id)
                txt_file_name = 'Failed_ISCloc_ISF_events.txt'
                txt = os.path.join(results_parent_dir, txt_file_name)
                f = open(txt, 'a+')
                f.write(str(event) + '\t' + str(event.event_id) + '\n')
                f.close()
                
            # Create final ISF inputs with new phases
            phase_relocation.create_input_ISFs_ISC(iscloc_inputs_dir, event.event_id, sta, dist, back, phase, times, lat, lon, elev, append=True)
            print()
            
            # Make station list per event using stations in ISF input
            print('Making station list')
            phase_relocation.make_event_station_list(iscloc_inputs_dir, event.event_id, station_list_loc, new_list=True, augmented=True)
            
        except Exception as e:
            #sys.exit()
            print('No outputs for %s' %event.event_id)
            print(str(event.evname) + '\t' + str(event.event_id) + '\t' + str(e) + '\t' + 'Error on line {}'.format(sys.exc_info()[-1].tb_lineno) + '\n')
            txt_file_name = 'Failed_Events_from_Array_Processing.txt'
            failed_txt = os.path.join(results_parent_dir, txt_file_name)
            f = open(failed_txt, 'a+')
            f.write(str(event.evname) + '\t' + str(event.event_id) + '\t' + str(e) + '\t' + 'Error on line {}'.format(sys.exc_info()[-1].tb_lineno) + '\n')
            f.close()
                    

    # --------- Record Time ----------
    TIME_S = time.time()
    TIME_Sp = time.ctime(TIME_S)
    print ('Script end time is ', TIME_Sp)
    f = open(outputs_txt, 'a+')
    f.write('Script end time is ' + str(TIME_Sp) + '\n')
    f.write('\n')
    f.close()
    
    print('Array processing script complete.')
    return





