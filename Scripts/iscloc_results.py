#!/usr/bin/env python

import os
import sys
import numpy as np
import math
import re
import random
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.patches import Ellipse
from obspy.geodetics import kilometers2degrees
import scipy.stats as stats
import seaborn as sns

def main(input_txt, iscloc_inputs):
    # open output file lines
    with open(input_txt) as file:
         iscloc_lines = [line.rstrip() for line in file]
    file.close()
    
    # open input file lines
    txt = input_txt.split('/')[-1]
    txt = txt.split('.')[0]
    with open(iscloc_inputs + txt + '.in') as file:
         isf_input_lines = [line.rstrip() for line in file]
    file.close()
    
    print(input_txt)  
    
    n=0    # no. events (total)
    fixed = 0 # fixed depth (AB)

    n_new_input_dp = 0 # number of input phases from AB (taken from input file)  - 'no_original_input_depth_phases'
    n_og_input_dp = 0  # number of original input phases (taken from input file) - 'no_new_input_depth_phases'    

    n_phases = 0 # number of phases in output file (AB + Everyone)   - 'no.phases'
    n_td_phases = 0 # number of time defining phases (AB + Everyone) - 'no.time_defining_phases'
    n_phases_AB = 0 # no. new phases (AB)                            - 'no.phases_AB'
    n_td_phases_AB = 0 # no. new time defining phases (AB)           - 'no.time_defining_phases_AB'
    
    n_dp = 0 # no of total depth phases (AB + Everyone)                - 'no.total_dp'
    n_td_dp = 0 # no. total time defining depth phases (AB + Everyone) - 'no.total_time_defining_dp'
    n_dp_AB = 0 # no. new depth phases (AB)                            - 'no.total_dp_AB'
    n_td_dp_AB = 0 # no new time defining depth phases (AB)            - 'no.total_time_defining_dp_AB'
    
    n_P = 0 # no. P (AB + Everyone)   - 'no.P'
    n_pP = 0 # no. pP (AB + Everyone) - 'no.pP'
    n_sP = 0 # no. sP (AB + Everyone) - 'no.sP'
    n_S = 0 # no. S (AB + Everyone)   - 'no.S'
    n_sS = 0 # no. sS (AB + Everyone) -' no.sS'
    
    n_P_AB = 0 # no. P (AB)   - 'no.P_AB'
    n_pP_AB = 0# no. pP (AB)  - 'no.pP_AB'
    n_sP_AB = 0 # no. sP (AB) - 'no.sP_AB'
    n_S_AB = 0 # no. S (AB)   - 'no.S_AB'
    n_sS_AB = 0 # no. sS (AB) - 'no.sS_AB'
    
    n_td_P = 0 # No. time defining P phases (AB + Everyone)   - 'no.time_defining_P'
    n_td_pP = 0 # No. time defining pP phases (AB + Everyone) - 'no.time_defining_pP'
    n_td_sP = 0 # No. time defining sP phases (AB + Everyone) - 'no.time_defining_sP'
    n_td_S = 0 # No. time defining S phases (AB + Everyone)   - 'no.time_defining_S'
    n_td_sS = 0 # No. time defining sS phases (AB + Everyone) - 'no.time_defining_S'
    
    n_td_P_AB = 0 # no. time defining P (AB)   - 'no.time_defining_P_AB'
    n_td_pP_AB = 0 # no. time defining pP (AB) - 'no.time_defining_pP_AB'
    n_td_sP_AB = 0 # no. time defining sP (AB) - 'no.time_defining_sP_AB'
    n_td_S_AB = 0 # no. time defining S (AB)   - 'no.time_defining_S_AB'
    n_td_sS_AB = 0 # no. time defining sS (AB) - 'no.time_defining_sS_AB'
     
    n_other_AB = 0 # no other phases (e.g. sP converted to PcP) (AB) - 'no.other_AB'

    mag = np.nan
    output = []
    phases = []
    
    ISC_outputs = [np.nan]*16
    EHB_outputs = [np.nan]*16
    NEIC_outputs = [np.nan]*16
    GCMT_outputs = [np.nan]*16
    AB_outputs = [np.nan]*17
    pub_pP_depth = np.nan   
    pub_pP_depth_err = np.nan
    EHB_dp_Q = np.nan
    
    for line in iscloc_lines:
        #print(line)
        
        # EVENT
        if re.search('Event', line):
            evid = line.split()[1]
            #print(evid)
            n += 1

        if re.search('^mb\s+[0-9]\.[0-9]\s', line) and 'ISC' in line:
            mag = line.split()[1]
            
        # ============== STRIPPING OUT CATALOGUE RESULTS ===============
        # depth phase depth and error (DPdep, Err)  
        elif re.search('AB\s+[0-9]\s+AB ', line):
        
            date = line[0:10].strip()
            origin_time = line[11:22].strip()
            time_err = line[25:29].strip()
            RMS = line[31:35].strip()
            hyp_lat = line[36:45].strip()
            hyp_lon = line[46:54].strip()
            Smaj = line[55:60].strip()
            Smin = line[61:66].strip()
            Az = line[68:71].strip()
            depth = line[71:76].strip()
            depth_err = line[77:82].strip()
            Ndef = line[83:87].strip()
            Nsta = line[88:93].strip()
            gap = line[93:97].strip()
            pP_depth = line[146:151].strip()       
            depth_error = line[153:157].strip()
            
            if 'f' in line[76:77]:
                fixed = 1
                #print('FIXED')
            
            #print(evid, date, origin_time, time_err, RMS, hyp_lat, hyp_lon, Smaj, Smin, Az, depth, depth_err, Ndef, Nsta, gap, pP_depth, depth_error)

            AB_outputs = [date, origin_time, time_err, RMS, hyp_lat, hyp_lon, Smaj, Smin, Az, depth, fixed, depth_err, Ndef, Nsta, gap, pP_depth, depth_error]
            
        elif re.search('^[0-9]{4}\/[0-9]{2}\/[0-9]{2}\s[0-9]{2}\:[0-9]{2}\:[0-9]{2}\.[0-9]{2}\s', line) and re.search('\sISC\s',line):
        
            date = line[0:10].strip()
            origin_time = line[11:22].strip()
            time_err = line[25:29].strip()
            RMS = line[31:35].strip()
            hyp_lat = line[36:45].strip()
            hyp_lon = line[46:54].strip()
            Smaj = line[55:60].strip()
            Smin = line[61:66].strip()
            Az = line[68:71].strip()
            depth = line[71:76].strip()
            depth_err = line[77:82].strip()
            Ndef = line[83:87].strip()
            Nsta = line[88:93].strip()
            gap = line[93:97].strip()
            pP_depth = line[146:151].strip()       
            depth_error = line[153:157].strip()
            
            #print(evid, date, origin_time, time_err, RMS, hyp_lat, hyp_lon, Smaj, Smin, Az, depth, depth_err, Ndef, Nsta, gap, pP_depth, depth_error)

            ISC_outputs = [date, origin_time, time_err, RMS, hyp_lat, hyp_lon, Smaj, Smin, Az, depth, depth_err, Ndef, Nsta, gap, pP_depth, depth_error]
            
            # Search for ISC published pP depth and error in input file            
            for line in isf_input_lines:
                if re.search('#PARAM pP_DEPTH=', line):
                    print(line)
                    values = line.split('=')[-1]
                    if '+' in values:
                        pub_pP_depth = values.split('+')[0]   
                        pub_pP_depth_err = values.split('+')[1][:-1]
                    else:
                        pass
            
        elif re.search('^[0-9]{4}\/[0-9]{2}\/[0-9]{2}\s[0-9]{2}\:[0-9]{2}\:[0-9]{2}\.[0-9]{2}\s', line) and re.search('\sISC-EHB\s',line):
        
            date = line[0:10].strip()
            origin_time = line[11:22].strip()
            time_err = line[25:29].strip()
            RMS = line[31:35].strip()
            hyp_lat = line[36:45].strip()
            hyp_lon = line[46:54].strip()
            Smaj = line[55:60].strip()
            Smin = line[61:66].strip()
            Az = line[68:71].strip()
            depth = line[71:76].strip()
            depth_err = line[77:82].strip()
            Ndef = line[83:87].strip()
            Nsta = line[88:93].strip()
            gap = line[93:97].strip()
            pP_depth = line[146:151].strip()       
            depth_error = line[153:157].strip()
            
            #print(evid, date, origin_time, time_err, RMS, hyp_lat, hyp_lon, Smaj, Smin, Az, depth, depth_err, Ndef, Nsta, gap, pP_depth, depth_error)

            EHB_outputs = [date, origin_time, time_err, RMS, hyp_lat, hyp_lon, Smaj, Smin, Az, depth, depth_err, Ndef, Nsta, gap, pP_depth, depth_error]
            
            # Search for EHB 'level' for quality in input file            
            for line in isf_input_lines:
                if re.search('#PARAM DEPTH_QUALITY=', line):
                    EHB_dp_Q = line.split('=')[-1][1]          
            
        elif re.search('^[0-9]{4}\/[0-9]{2}\/[0-9]{2}\s[0-9]{2}\:[0-9]{2}\:[0-9]{2}\.[0-9]{2}\s', line) and re.search('\sGCMT\s',line):
        
            date = line[0:10].strip()
            origin_time = line[11:22].strip()
            time_err = line[25:29].strip()
            RMS = line[31:35].strip()
            hyp_lat = line[36:45].strip()
            hyp_lon = line[46:54].strip()
            Smaj = line[55:60].strip()
            Smin = line[61:66].strip()
            Az = line[68:71].strip()
            depth = line[71:76].strip()
            depth_err = line[77:82].strip()
            Ndef = line[83:87].strip()
            Nsta = line[88:93].strip()
            gap = line[93:97].strip()
            pP_depth = line[146:151].strip()       
            depth_error = line[153:157].strip()
            
            #print(evid, date, origin_time, time_err, RMS, hyp_lat, hyp_lon, Smaj, Smin, Az, depth, depth_err, Ndef, Nsta, gap, pP_depth, depth_error)

            GCMT_outputs = [date, origin_time, time_err, RMS, hyp_lat, hyp_lon, Smaj, Smin, Az, depth, depth_err, Ndef, Nsta, gap, pP_depth, depth_error]
            
        elif re.search('^[0-9]{4}\/[0-9]{2}\/[0-9]{2}\s[0-9]{2}\:[0-9]{2}\:[0-9]{2}\.[0-9]{2}\s', line) and re.search('\sNEIC\s',line):
            try:
                if float(line[55:60]) > 0 and float(line[61:66])>0:
                    
                    date = line[0:10].strip()
                    origin_time = line[11:22].strip()
                    time_err = line[25:29].strip()
                    RMS = line[31:35].strip()
                    hyp_lat = line[36:45].strip()
                    hyp_lon = line[46:54].strip()
                    Smaj = line[55:60].strip()
                    Smin = line[61:66].strip()
                    Az = line[68:71].strip()
                    depth = line[71:76].strip()
                    depth_err = line[77:82].strip()
                    Ndef = line[83:87].strip()
                    Nsta = line[88:93].strip()
                    gap = line[93:97].strip()
                    pP_depth = line[146:151].strip()       
                    depth_error = line[153:157].strip()
                    
                    #print(evid, date, origin_time, time_err, RMS, hyp_lat, hyp_lon, Smaj, Smin, Az, depth, depth_err, Ndef, Nsta, gap, pP_depth, depth_error)

                    NEIC_outputs = [date, origin_time, time_err, RMS, hyp_lat, hyp_lon, Smaj, Smin, Az, depth, depth_err, Ndef, Nsta, gap, pP_depth, depth_error]
                    
            except:
                pass
        
    # ============= STRIPPING OUT PHASE STATISTICS ===============              
    # Find time defining depth phases (AB)           
    for line in iscloc_lines:
        if re.search('[A-z0-9]+\s+[0-9]+\.[0-9]+\s+[0-9]+\.+[0-9]\s',line):
            if 'FDSN  ZZ' in line:
                n_phases_AB += 1 
                if 'T__' in line:
                    n_td_phases_AB += 1  
                    if re.search('[A-z0-9]+\s+[0-9]+\.[0-9]+\s+[0-9]+\.+[0-9]\sP',line): # P
                        n_td_P_AB += 1 
                    if re.search('[A-z0-9]+\s+[0-9]+\.[0-9]+\s+[0-9]+\.+[0-9]\spP',line): # pP
                        n_td_pP_AB += 1
                    if re.search('[A-z0-9]+\s+[0-9]+\.[0-9]+\s+[0-9]+\.+[0-9]\ssP',line): # sP
                        n_td_sP_AB += 1
                    if re.search('[A-z0-9]+\s+[0-9]+\.[0-9]+\s+[0-9]+\.+[0-9]\sP',line): # S
                        n_td_S_AB += 1
                    if re.search('[A-z0-9]+\s+[0-9]+\.[0-9]+\s+[0-9]+\.+[0-9]\ssS',line): # sS
                        n_td_sS_AB += 1                    
                else:
                    if re.search('[A-z0-9]+\s+[0-9]+\.[0-9]+\s+[0-9]+\.+[0-9]\sP',line): # P
                        n_P_AB += 1 
                    if re.search('[A-z0-9]+\s+[0-9]+\.[0-9]+\s+[0-9]+\.+[0-9]\spP',line): # pP
                        n_pP_AB += 1
                    if re.search('[A-z0-9]+\s+[0-9]+\.[0-9]+\s+[0-9]+\.+[0-9]\ssP',line): # sP
                        n_sP_AB += 1
                    if re.search('[A-z0-9]+\s+[0-9]+\.[0-9]+\s+[0-9]+\.+[0-9]\sP',line): # S
                        n_S_AB += 1
                    if re.search('[A-z0-9]+\s+[0-9]+\.[0-9]+\s+[0-9]+\.+[0-9]\ssS',line): # sS
                        n_sS_AB += 1
                            
    n_dp_AB = n_pP_AB + n_sP_AB + n_sS_AB    
    n_td_dp_AB = n_td_pP_AB + n_td_sP_AB + n_td_sS_AB
    
    
    # Find time defining depth phases (Everyone)           
    for line in iscloc_lines:
        if re.search('[A-z0-9]+\s+[0-9]+\.[0-9]+\s+[0-9]+\.+[0-9]\s',line):
            n_phases += 1
            if 'T__' in line:
                n_td_phases += 1  
                if re.search('[A-z0-9]+\s+[0-9]+\.[0-9]+\s+[0-9]+\.+[0-9]\sP',line): # P
                    n_td_P += 1 
                if re.search('[A-z0-9]+\s+[0-9]+\.[0-9]+\s+[0-9]+\.+[0-9]\spP',line): # pP
                    n_td_pP += 1
                if re.search('[A-z0-9]+\s+[0-9]+\.[0-9]+\s+[0-9]+\.+[0-9]\ssP',line): # sP
                    n_td_sP += 1
                if re.search('[A-z0-9]+\s+[0-9]+\.[0-9]+\s+[0-9]+\.+[0-9]\sP',line): # S
                    n_td_S += 1
                if re.search('[A-z0-9]+\s+[0-9]+\.[0-9]+\s+[0-9]+\.+[0-9]\ssS',line): # sS
                    n_td_sS += 1                    
            else:
                if re.search('[A-z0-9]+\s+[0-9]+\.[0-9]+\s+[0-9]+\.+[0-9]\sP',line): # P
                    n_P += 1 
                if re.search('[A-z0-9]+\s+[0-9]+\.[0-9]+\s+[0-9]+\.+[0-9]\spP',line): # pP
                    n_pP += 1
                if re.search('[A-z0-9]+\s+[0-9]+\.[0-9]+\s+[0-9]+\.+[0-9]\ssP',line): # sP
                    n_sP += 1
                if re.search('[A-z0-9]+\s+[0-9]+\.[0-9]+\s+[0-9]+\.+[0-9]\sP',line): # S
                    n_S += 1
                if re.search('[A-z0-9]+\s+[0-9]+\.[0-9]+\s+[0-9]+\.+[0-9]\ssS',line): # sS
                    n_sS += 1
    
    n_dp = n_pP + n_sP + n_sS
    n_td_dp = n_td_pP + n_td_sP + n_td_sS
    
    # Checking input ISF
    n_new_input_dp = 0 
    n_new_input_P = 0
    n_new_input_pP = 0
    n_new_input_sP = 0
    n_new_input_S = 0
    n_new_input_sS = 0
    
    n_og_input_dp = 0 
    n_og_input_P = 0
    n_og_input_pP = 0
    n_og_input_sP = 0
    n_og_input_S = 0
    n_og_input_sS = 0
                   
    for line in isf_input_lines:       
        if 'AEB   ZZ' in line:
            if re.search('[A-z0-9]+\s+[0-9]+\.[0-9]+\s+[0-9]+\.+[0-9]\sP',line):
                n_new_input_P += 1 #'no.original_input_P',
            elif re.search('[A-z0-9]+\s+[0-9]+\.[0-9]+\s+[0-9]+\.+[0-9]\spP',line):
                n_new_input_pP += 1  
            elif re.search('[A-z0-9]+\s+[0-9]+\.[0-9]+\s+[0-9]+\.+[0-9]\ssP',line):
                n_new_input_sP += 1
            elif re.search('[A-z0-9]+\s+[0-9]+\.[0-9]+\s+[0-9]+\.+[0-9]\sS',line):
                n_new_input_S += 1 
            elif re.search('[A-z0-9]+\s+[0-9]+\.[0-9]+\s+[0-9]+\.+[0-9]\ssS',line):
                n_new_input_sS += 1 
        
        elif re.search('[A-z0-9]+\s+[0-9]+\.[0-9]+\s+[0-9]+\.+[0-9]\sP',line):
            n_og_input_P += 1 #'no.new_input_P',
        elif re.search('[A-z0-9]+\s+[0-9]+\.[0-9]+\s+[0-9]+\.+[0-9]\spP',line):
            n_og_input_pP += 1  
        elif re.search('[A-z0-9]+\s+[0-9]+\.[0-9]+\s+[0-9]+\.+[0-9]\ssP',line):
            n_og_input_sP += 1
        elif re.search('[A-z0-9]+\s+[0-9]+\.[0-9]+\s+[0-9]+\.+[0-9]\sS',line):
            n_og_input_S += 1
        elif re.search('[A-z0-9]+\s+[0-9]+\.[0-9]+\s+[0-9]+\.+[0-9]\ssS',line):
            n_og_input_sS += 1

    n_new_input_dp = n_new_input_pP + n_new_input_sP + n_new_input_sS # - 'no_original_input_depth_phases'
    n_og_input_dp = n_og_input_pP + n_og_input_sP + n_og_input_sS     # - 'no_new_input_depth_phases'
    
    n_other_AB = n_new_input_dp - n_dp_AB    # checking to see how many depth phases change phase from input to output files
             
    output = [evid, mag, n_og_input_dp, n_og_input_P, n_og_input_pP, n_og_input_sP, n_og_input_S, n_og_input_sS, n_new_input_dp, n_new_input_P, n_new_input_pP, n_new_input_sP, n_new_input_S, n_new_input_sS, n_phases, n_td_phases, n_phases_AB, n_td_phases_AB, n_dp, n_td_dp, n_dp_AB, n_td_dp_AB, n_P, n_pP, n_sP, n_S, n_sS, n_P_AB, n_pP_AB, n_sP_AB, n_S_AB, n_sS_AB, n_td_P, n_td_pP, n_td_sP, n_td_S, n_td_sS, n_td_P_AB, n_td_pP_AB, n_td_sP_AB, n_td_S_AB, n_td_sS_AB, n_other_AB]  
    
    output.extend(ISC_outputs)
    output.append(pub_pP_depth)   
    output.append(pub_pP_depth_err)
    output.extend(EHB_outputs)
    output.append(EHB_dp_Q)
    output.extend(GCMT_outputs)
    output.extend(NEIC_outputs)
    output.extend(AB_outputs)
    
    #print(output)
    #print('No. of Events:', n)
    return output
                
#=======================================              

def strip_iscloc_results(final_3D_cat_name, analysis_only, iscloc_inputs, iscloc_outputs, include_original_phase_results):

    if analysis_only != True:
        
        name = '%s_detailed.csv' %final_3D_cat_name
        
        # work through output files from ISF input plus AB phases
        file_flag = '[0-9]+\.out'  # added AB phases 

        output= []
        for file in os.listdir(iscloc_outputs):
            print(file)  
            if re.search(file_flag,str(file)):
                try:
                    output.append(main(iscloc_outputs+file, iscloc_inputs))
                except Exception as e:
	                print('error', e)
	                output.append([np.nan]*33)

        plot_data=pd.DataFrame(output, columns=['evid', 'magnitude', 'no_original_input_depth_phases', 'no.original_input_P', 'no.original_input_pP', 'no.original_input_sP', 'no.original_input_S', 'no.original_input_sS', 'no_new_input_depth_phases', 'no.new_input_P', 'no.new_input_pP', 'no.new_input_sP', 'no.new_input_S', 'no.new_input_sS', 'no.phases', 'no.time_defining_phases', 'no.phases_AB', 'no.time_defining_phases_AB', 'no.total_dp', 'no.total_time_defining_dp',
        'no.total_dp_AB', 'no.total_time_defining_dp_AB', 'no.P', 'no.pP', 'no.sP', 'no.S', 'no.sS', 'no.P_AB', 'no.pP_AB', 'no.sP_AB', 'no.S_AB', 'no.sS_AB', 'no.time_defining_P', 'no.time_defining_pP', 'no.time_defining_sP', 'no.time_defining_S', 'no.time_defining_sS',
        'no.time_defining_P_AB', 'no.time_defining_pP_AB', 'no.time_defining_sP_AB', 'no.time_defining_S_AB', 'no.time_defining_sS_AB', 'no.other_AB',
        'ISC_date', 'ISC_origin_time', 'ISC_time_err', 'ISC_RMS', 'ISC_hyp_lat', 'ISC_hyp_lon', 'ISC_Smaj', 'ISC_Smin', 'ISC_Az', 'ISC_depth', 'ISC_depth_err', 'ISC_Ndef', 'ISC_Nsta', 'ISC_gap', 'ISC_pP_depth', 'ISC_depth_error', 'ISC_pub_pP_depth', 'ISC_pub_pP_depth_err',
        'EHB_date', 'EHB_origin_time', 'EHB_time_err', 'EHB_RMS', 'EHB_hyp_lat', 'EHB_hyp_lon', 'EHB_Smaj', 'EHB_Smin', 'EHB_Az', 'EHB_depth', 'EHB_depth_err', 'EHB_Ndef', 'EHB_Nsta', 'EHB_gap', 'EHB_pP_depth', 'EHB_depth_error', 'EHB_dp_quality',
        'GCMT_date', 'GCMT_origin_time', 'GCMT_time_err', 'GCMT_RMS', 'GCMT_hyp_lat', 'GCMT_hyp_lon', 'GCMT_Smaj', 'GCMT_Smin', 'GCMT_Az', 'GCMT_depth', 'GCMT_depth_err', 'GCMT_Ndef', 'GCMT_Nsta', 'GCMT_gap', 'GCMT_pP_depth', 'GCMT_depth_error', 
        'NEIC_date', 'NEIC_origin_time', 'NEIC_time_err', 'NEIC_RMS', 'NEIC_hyp_lat', 'NEIC_hyp_lon', 'NEIC_Smaj', 'NEIC_Smin', 'NEIC_Az', 'NEIC_depth', 'NEIC_depth_err', 'NEIC_Ndef', 'NEIC_Nsta', 'NEIC_gap', 'NEIC_pP_depth', 'NEIC_depth_error',
        'AB_date', 'AB_origin_time', 'AB_time_err', 'AB_RMS', 'AB_hyp_lat', 'AB_hyp_lon', 'AB_Smaj', 'AB_Smin', 'AB_Az', 'AB_depth', 'AB_fixed', 'AB_depth_err', 'AB_Ndef', 'AB_Nsta', 'AB_gap', 'AB_pP_depth', 'AB_pP_depth_err' ])
        

        # Change %s to match input type:
        plot_data.to_csv(name)
        
        if include_original_phase_results == True:
            # Add ISF original results to catalogue, and jack knifed results
            evid = plot_data['evid'].to_numpy()
            print(evid, len(evid))
            
            # Match evid to find ISF without added phases results
            output= []
            for ev in evid: 
                file = str(ev) + '.ISFout' 
                try:
                    output.append(main(iscloc_outputs+file, iscloc_inputs)[-17:])
                except Exception as e:
                    print('error', e)
                    output.append([np.nan]*17)
               
            plot_data[['ISF_date', 'ISF_origin_time', 'ISF_time_err', 'ISF_RMS', 'ISF_hyp_lat', 'ISF_hyp_lon', 'ISF_Smaj', 'ISF_Smin', 'ISF_Az', 'ISF_depth', 'ISF_fixed', 'ISF_depth_err', 'ISF_Ndef', 'ISF_Nsta', 'ISF_gap', 'ISF_pP_depth', 'ISF_pP_depth_err']] = output
            
            # Change %s to match input type:
            plot_data.to_csv(name)
          
        # Create final (nice) catalogue from ISCloc results
        name = str(final_3D_cat_name) + '.txt'
        
        cols=['AB_date', 'AB_origin_time', 'evid', 'magnitude', 'AB_hyp_lat', 'AB_hyp_lon', 'AB_Smaj', 'AB_Smin', 'AB_Az', 'ISC_depth', 'AB_depth', 'AB_depth_err']
        
        plot_data = plot_data[(pd.to_numeric(plot_data['ISC_depth'], errors='coerce')>=40)] # remove events which were initially <40 km deep
        plot_data = plot_data[(pd.to_numeric(plot_data['ISC_depth'], errors='coerce')<=350)]
        plot_data = plot_data[(plot_data['no.total_time_defining_dp_AB']>0)] # only eents where phases were added
        plot_data = plot_data[(plot_data['AB_fixed']==0)] # no depth fixed events relocations
        final_cat_df = plot_data[cols].to_numpy()
        
        output_file = name
        f = open(output_file, 'w')
        f.write('Date'.ljust(12) + '\t' + 'Time'.ljust(12) + '\t' + 'Event_id'.ljust(12) + '\t' + 'mb'.ljust(12) + '\t' + 'Lat'.ljust(12) + '\t' + 'Lon'.ljust(12) + '\t' + 'Smaj_Err'.ljust(12) + '\t' + 'Smin_Err'.ljust(12) + '\t' + 'Az'.ljust(12) + '\t' + 'ISC_Depth'.ljust(12) + '\t' + 'R_Depth'.ljust(12) + '\t' + 'Error'.ljust(12) + '\n')
        for i in range (len(final_cat_df)):
            for j in range (len(final_cat_df[i])):
                f.write(str(final_cat_df[i][j]).ljust(12) + '\t')
            f.write('\n')
        f.close()


    # Analysis =====================

    # ======== LOAD DF ========
    # load dataframe from csv file
    df = pd.read_csv('%s_detailed.csv' %final_3D_cat_name, dtype=float, converters={'ISC_date':str, 'ISC_origin_time':str, 'EHB_date':str, 'EHB_origin_time':str, 'GCMT_date':str, 'GCMT_origin_time':str, 'NEIC_date':str, 'NEIC_origin_time':str, 'AB_date':str, 'AB_origin_time':str, 'ISF_date':str, 'ISF_origin_time':str, 'noS_date':str, 'noS_origin_time':str})   

    # filter df to remove events <40 km, >350 km
    df = df[df['ISC_depth']>=40]
    df = df[df['ISC_depth']<=350]

    # ========= Statistics ==========
    print()
    # No. events with additional phases
    print('No. events with additional phases:', len(df[df['no.phases_AB']>0]))
    print('No. additional phases:', np.sum(df['no.phases_AB']))
    print('Mean phases added:', np.sum(df['no.phases_AB'])/len(df['no.phases_AB']))
    print('Min. (except 0) phases added:', np.min(df['no.phases_AB'][df['no.phases_AB']!=0]))
    print('Max. phases added', np.max(df['no.phases_AB']))
    print()
    print('No. depth phases added:', np.sum(df['no_new_input_depth_phases']))
    print('Mean depth phases added:', np.sum(df['no_new_input_depth_phases'])/len(df['no_new_input_depth_phases']))
    print('Min. (except 0) depth phases added:', np.min(df['no_new_input_depth_phases'][df['no_new_input_depth_phases']!=0]))
    print('Max depth phases added:', np.max(df['no_new_input_depth_phases']))
    print('No. original depth phases:', np.sum(df['no_original_input_depth_phases']))
    print('Percentage increase in depth phases:', (np.sum(df['no_new_input_depth_phases'])/np.sum(df['no_original_input_depth_phases']))*100)
    print()
    print('Total td depth phases:', np.sum(df['no.total_time_defining_dp']))
    print('Total new td depth phases (AB):', np.sum(df['no.total_time_defining_dp_AB']))
    print('Percentage increase in td depth phases:', (np.sum(df['no.total_time_defining_dp_AB'])/np.sum(df['no.total_time_defining_dp']))*100)
    print()
    
    if include_original_phase_results==True:
        og_sub5 = np.sum((df['no.total_dp']-df['no.total_dp_AB']).to_numpy()<5)
        sub5 = np.sum((df['no.total_dp']).to_numpy()<5)
        print('No. events originally with <5 depth phases:', og_sub5)
        print('No events with <5 depth phases, post additions:', sub5)
        print('No events which now have >5 depth phases:', og_sub5-sub5)
        print('Percentage decrease in events <5 depth phases:', ((og_sub5-sub5)/og_sub5)*100)
        print('No. events no longer fixed:', np.nansum(df['ISF_fixed'].to_numpy())-np.nansum(df['AB_fixed'].to_numpy()))
        print()

    print('Number original input P:', np.sum(df['no.original_input_P']))
    print('Number original input pP:', np.sum(df['no.original_input_pP']))
    print('Number original input sP:', np.sum(df['no.original_input_sP']))
    print('Number original input S:', np.sum(df['no.original_input_S']))
    print('Number original input sS:', np.sum(df['no.original_input_sS']))
    print()
    print('Number new input P:', np.sum(df['no.new_input_P']))
    print('Number new input pP:', np.sum(df['no.new_input_pP']))
    print('Number new input sP:', np.sum(df['no.new_input_sP']))
    print('Number new input S:', np.sum(df['no.new_input_S']))
    print('Number new input sS:', np.sum(df['no.new_input_sS']))
    print()
    print('Number output P:', np.sum(df['no.P']))
    print('Number output pP:', np.sum(df['no.pP']))
    print('Number output sP:', np.sum(df['no.sP']))
    print('Number output S:', np.sum(df['no.S']))
    print('Number output sS:', np.sum(df['no.sS']))
    print()
    print('Number new output P, %:', np.sum(df['no.P_AB']), (np.sum(df['no.P_AB'])/np.sum(df['no.P']))*100)
    print('Number new output pP, %:', np.sum(df['no.pP_AB']), (np.sum(df['no.pP_AB'])/np.sum(df['no.pP']))*100)
    print('Number new output sP, %:', np.sum(df['no.sP_AB']), (np.sum(df['no.sP_AB'])/np.sum(df['no.sP']))*100)
    print('Number new output S, %:', np.sum(df['no.S_AB']), (np.sum(df['no.S_AB'])/np.sum(df['no.S']))*100)
    print('Number new output sS, %:', np.sum(df['no.sS_AB']), (np.sum(df['no.sS_AB'])/np.sum(df['no.sS']))*100)
    print()
    print('Number time defining P:', np.sum(df['no.time_defining_P']))
    print('Number time defining pP:', np.sum(df['no.time_defining_pP']))
    print('Number time defining sP:', np.sum(df['no.time_defining_sP']))
    print('Number time defining S:', np.sum(df['no.time_defining_S']))
    print('Number time defining sS:', np.sum(df['no.time_defining_sS']))
    print()
    print('Number new time defining P, %:', np.sum(df['no.time_defining_P_AB']), (np.sum(df['no.time_defining_P_AB'])/np.sum(df['no.time_defining_P']))*100)
    print('Number new time defining pP, %:', np.sum(df['no.time_defining_pP_AB']), (np.sum(df['no.time_defining_pP_AB'])/np.sum(df['no.time_defining_pP']))*100)
    print('Number new time defining sP, %:', np.sum(df['no.time_defining_sP_AB']), (np.sum(df['no.time_defining_sP_AB'])/np.sum(df['no.time_defining_sP']))*100)
    print('Number new time defining S, %:', np.sum(df['no.time_defining_S_AB']), (np.sum(df['no.time_defining_S_AB'])/np.sum(df['no.time_defining_S']))*100)
    print('Number new time defining sS, %:', np.sum(df['no.time_defining_sS_AB']), (np.sum(df['no.time_defining_sS_AB'])/np.sum(df['no.time_defining_sS']))*100)

    print('No. events with additional td phases:', len(df[df['no.total_time_defining_dp_AB']>0]))

    if include_original_phase_results==True:
        # From AB's perspective, e.g. larger means AB error is larger
        counter_larger = 0
        counter_same = 0
        counter_smaller = 0
        for ev in range (len(df_unfixed['ISF_depth_err'])):
            #print(ev, df_unfixed['ISF_depth_err'].to_numpy()[ev], df_unfixed['AB_depth_err'].to_numpy()[ev])
            if df_unfixed['ISF_depth_err'].to_numpy()[ev] < df_unfixed['AB_depth_err'].to_numpy()[ev]:
                #print(df_unfixed['ISF_depth_err'].to_numpy()[ev], df_unfixed['AB_depth_err'].to_numpy()[ev])
                counter_larger += 1
            if df_unfixed['ISF_depth_err'].to_numpy()[ev] == df_unfixed['AB_depth_err'].to_numpy()[ev]:
                #print(df_unfixed['ISF_depth_err'].to_numpy()[ev], df_unfixed['AB_depth_err'].to_numpy()[ev])
                counter_same += 1
            if df_unfixed['ISF_depth_err'].to_numpy()[ev] > df_unfixed['AB_depth_err'].to_numpy()[ev]:
                #print(df_unfixed['ISF_depth_err'].to_numpy()[ev], df_unfixed['AB_depth_err'].to_numpy()[ev])
                counter_smaller += 1
        #print(counter, len(df_unfixed['ISF_depth_err']))
        #print('No. Events with depth error reduction, %:', counter, ((len(df_unfixed['ISF_depth_err'])-counter)/len(df_unfixed['ISF_depth_err'])*100))

        print('AB err larger:', counter_larger, (counter_larger/len(df_unfixed['ISF_depth_err'])*100))
        print('AB err equal:', counter_same, (counter_same/len(df_unfixed['ISF_depth_err'])*100))
        print('AB err less:', counter_smaller, (counter_smaller/len(df_unfixed['ISF_depth_err'])*100))
        #print('Sum:', counter_larger + counter_same + counter_smaller)
        print('No. AB has new err for (previously NaN):', (len(df_unfixed['ISF_depth_err']) - (counter_larger + counter_same + counter_smaller)), ((len(df_unfixed['ISF_depth_err']) - (counter_larger + counter_same + counter_smaller))/len(df_unfixed['ISF_depth_err']))*100) 


