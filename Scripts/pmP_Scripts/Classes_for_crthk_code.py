#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Tue Jul 11 15:57:32 2023

Author: Hanna-Riia Allas (earha@leeds.ac.uk)

Defining classes to use in the Crustal_thickness_code.

"""
# Importing modules

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import copy
import obspy
import os
from scipy.signal import find_peaks, peak_prominences, peak_widths

import matplotlib.pyplot as plt

from scipy.signal import find_peaks

#==============================================================================

class Cr_subarray:
    def __init__(self, subarray_outputs, event_depth_improved, pmP_pick_exists=False):
        """
        Defining a class to deal with all of the crustal thickness operations.
        A Cr_subarray object stores data from a single sub-array, loaded in from
        the corresponding Subarray class object in Alice's code.
        
        Cr_subarray inputs:
            
        subarray_outputs: contains all the parameters from the associated Subarray class object.
        event_depth_improved: the updated event depth from Alice's results
        
        List of data stored in Alice's Subarray class that I can access and use within the Cr_subarray class using self.outputs):
            
        outputs.binned_stream                  # initial traces in subarray
        outputs.tpratio                        # taper ratio used
        outputs.frqmin                         # minimum freq pass
        outputs.frqmax                         # maximum freq pass
        outputs.evname                         # event name
        outputs.evla                           # event latitude
        outputs.evlo                           # event longitude
        outputs.ev_depth                       # event depth
        outputs.origin_time                    # origin time
        outputs.trim_start                     # trim start time (s)
        outputs.trim_interval                  # trim interval/end time (s)
        outputs.rsample                        # re-sampling rate (10 Hz typically)
        outputs.subarray_no                    # subarray "station" name
        outputs.fig_dir                        # file pathway to figures directory
        outputs.ev_subarray_gcarc              # event to subarray great circle distance
        outputs.subarray_P_time                # modelled P arrival time
        outputs.subarray_pP_time               # modelled pP arrival time
        outputs.subarray_sP_time               # modelled sP arrival time
        outputs.subarray_slowness              # calculated subarray slowness
        outputs.beampack_backazimuth           # beampacked found backazimuth
        outputs.beampack_slowness              # beampack found slowness
        outputs.backazimuth_range              # backazimuth test range
        outputs.slowness_range                 # slowness test range
        outputs.max_P_envelope_grd             # beampacking grid
        outputs.slowness_index                 # array index for optimum beampacked slowness within tested slowness range
        outputs.vespa_grid                     # vespagram data
        outputs.repeated_loop                  # x-correlation QC check has made the subarray function run again - True/False
        outputs.final_picks                    # TE picks (array of three arrivals given as pts on trimmed trace, 0 if no pick)
        outputs.stream                         # intitially saved out binned stream, normalised and resampled
        outputs.stname                         # station names
        outputs.stla                           # station latitudes
        outputs.stlo                           # station longitudes
        outputs.stel                           # station elevations
        outputs.stnet                          # station networks
        outputs.ev_st_gcarc                    # event to station great circle distance
        outputs.st_baz                         # station backazimuth
        outputs.subarray_longitude             # subarray centre longitude
        outputs.subarray_latitude              # subarray centre latitude
        outputs.subarray_baz                   # calculated subarray backazimuth
        outputs.confidence                     # subarray confidence
        outputs.peaks                          # peaks???
        outputs.yyyy                           # origin time year
        outputs.mn                             # origin time month
        outputs.dd                             # origin time day
        outputs.hh                             # origin time hour
        outputs.mm                             # origin time minutes
        outputs.ss                             # origin time seconds
        outputs.timeshifted_stream             # time-shifted stream using optimum backazimuth and slowness values
        outputs.beam                           # all vespagram beams created at a range of slownesses
        outputs.phase_weighted_beam            # all vespagram phase weighted beams at a range of slownesses
        outputs.relative_time                  # relative time for x-axis
        outputs.optimum_beam                   # optimum beam formed at best fit slowness and backazimuth
        outputs.PW_optimum_beam                # optimum phase weighted beam formed at best fit slowness and backazimuth
        outputs.lamda                          # pseudo wavelength
        outputs.envelope                       # envelope of phase weighted optimum beam
        outputs.picking_threshold              # picking threshold
        outputs.x_corr_lag                     # cross correlation lag 
        outputs.x_corr                         # cross correlations 
        outputs.x_corr_shift                   # cross correlation shift
        outputs.x_corr_trimmed_PW_beam         # trimmed phase weighted beams used for x-correlation
        outputs.x_corr_trimmed_traces          # trimmed traces used for x-correlation
        outputs.vespa_QC_ccx                   # vespagram QC amplitude clusters, core centre x coordinate
        outputs.vespa_QC_ccy                   # vespagram QC amplitude clusters, core centre y coordinate
        outputs.vespa_QC_cc_std                # vespagram QC amplitude clusters, core centre std
        outputs.vespa_QC_cc_mean               # vespagram QC amplitude clusters, core centre mean
        outputs.vespa_QC_cores                 # vespagram QC amplitude clusters, cores
        outputs.vespa_QC_npt                   # vespagram QC amplitude clusters, no points in each core/cluster
        outputs.subarray_calculated_PW_beam    # phase weighted beam created with calculated slowness and backazimuth values
        outputs.pick_weights                   # pick weightings
        outputs.TE_P_rel_onset                 # absolute onset of P
        outputs.TE_pP_rel_onset                # absolute onset of pP
        outputs.TE_sP_rel_onset                # absolute onset of sP
        outputs.dt_pP_TE                       # differential time between pP-P
        outputs.dt_sP_TE                       # differential time between sP-P
        outputs.epicentral_dist_pP_TE          # epicentral distances of subarrays with found pP-P results
        outputs.epicentral_dist_sP_TE          # epicentral distances of subarrays with found sP-P results
        
        List of (new) Cr_subarray attributes:
        
        self.crthk                             # Crust1 first-pass crustal thickness value at event coordinates
        self.evdepth                           # event depth from relocation code results
        
        self.pP_ray_parameter
        self.pmP_ray_parameter
        
        self.pP_bounce_lat_m
        self.pP_bounce_lon_m
        self.pmP_bounce_lat_m
        self.pmP_bounce_lon_m
        
        The following attributes exist only for subarrays with one pmP pick
        
        pP_pmP_delay_time
        
        self.new_crthk_FM                      # crustal thickness value obtained from pmP-pP delay time for the subarray using the forward modelling method
        self.new_crthk_RP                      # crustal thickness value calculated from pmP-pP delay time for the subarray using pmP ray parameter (method currently not called in the main code)
        
        self.pP_bounce_lat
        self.pP_bounce_lon
        self.pmP_bounce_lat
        self.pmP_bounce_lon
        
        
        
            
        """
        
        self.outputs = subarray_outputs
        self.evdepth = event_depth_improved
        
        self.pmP_pick_exists = False
                
#==============================================================================
    def predict_arrivals(self, model, crustal_thickness):
        
        """
        This method predicts the modelled travel times for the pmP and smP phases
        using a 1D velocity model. It also recalculates P, pP and sP model arrivals.
        
        The method also saves ray parameters for the predicted phases into the class object,
        to use in crustal thickness calculations later.
        
        Inputs:
            outputs.ev_subarray_gcarc: event to subarrray centre epicentral distance (degrees)
            crustal_thickness: crustal thickness value from Crust1
            velocity_model: the custom model used in Taup traveltime calculation
            event_depth: event depth (from Alice's improved depth catalogue)
            
        Returns:
            self.subarray_pmP_time: modelled pmP arrival for subarray centre, relative time w.r.t. event (s)
            self.subarray_smP_time: modelled smP arrival for subarray centre, relative time w.r.t. event (s)
            self.subarray_P_revised: P arrival modelled using custom model and revised depth
            self.subarray_pP_revised: pP arrival modelled using custom model and revised depth
            self.subarray_sP_revised = sP arrival modelled using custom model and revised depth
            
            self.pP_ray_parameter: pP ray parameter
            self.pmP_ray_parameter: pmP ray parameter
        """
        
        # Import variables
        
        outputs = self.outputs
        crthk = crustal_thickness
        
        # Save Crust1 first-pass crustal thickness value into the class object for later
        
        self.crthk = crthk
        
        # Predict traveltimes
        
        arrivals = model.get_travel_times(source_depth_in_km=self.evdepth, 
                                          distance_in_degree=outputs.ev_array_gcarc, 
                                          phase_list=["P", "p^"+str(int(crthk))+"P", "s^"+str(int(crthk))+"P", "pP", "sP"])
                
        #print("Predicted arrivals for subarray using custom model", outputs.array_no, ":", arrivals)
          
        if (len(arrivals) < 0.5):
            print("No predicted arrivals for subarray at", outputs.array_latitude, outputs.array_longitude)
        else:
            pass
        
        # Save arrival times wrt event
        
        for i in range(5):
            if arrivals[i].name == "P":
                P_time = arrivals[i].time
            elif arrivals[i].name == ("p^"+str(int(crthk))+"P"):
                pmP_time = arrivals[i].time
            elif arrivals[i].name == "pP":
                pP_time = arrivals[i].time
            elif arrivals[i].name == "sP":
                sP_time = arrivals[i].time
            else:
                smP_time = arrivals[i].time
        
#        ak135 = taup.TauPyModel(model='ak135')
#        ak135_arrivals = ak135.get_travel_times(source_depth_in_km=self.evdepth, distance_in_degree=outputs.ev_array_gcarc, phase_list=["P", "pP", "sP", "p^mP"])
#        print("For comparison, these are some phase arrivals predicted from the unmodified ak135 model:", ak135_arrivals)
                
        self.subarray_pmP_time = pmP_time
        self.subarray_smP_time = smP_time
        self.subarray_P_revised = P_time    # save P, pP and sP modelled arrivals as revised values, since these have now been modelled with the revised event depth and the custom crustal model
        self.subarray_pP_revised = pP_time
        self.subarray_sP_revised = sP_time
        
        # Save ray parameters for pmP and pP phases to use in crustal thickness calculations later
        
        for i in range(5):
            if arrivals[i].name == ("p^"+str(int(crthk))+"P"):
                pmP_ray_parameter = arrivals[i].ray_param_sec_degree
            elif arrivals[i].name == "pP":
                pP_ray_parameter = arrivals[i].ray_param_sec_degree
            else:
                pass
                
        self.pP_ray_parameter = pP_ray_parameter
        self.pmP_ray_parameter = pmP_ray_parameter
                
        return  
#==============================================================================
    def predict_pmP_arrival_relative_to_DP(self):
        
        """
        This method predicts the relative arrival time for the pmP phase by first calculating 
        the difference between the modelled pmP and pP arrivals, then subtracting this
        difference from Alice's actual pP pick. This is done because the pmP/pP
        difference is sensitive only to the crust and not the event depth or any
        far away velocity structure, so is potentially better for defining a search window
        for pmP. 
        
        Inputs:
            self.subarray_pmP_time: modelled pmP arrival time wrt event
            self.subarray_pP_revised: modelled pP arrival time wrt event using the improved event depth

            outputs.final_picks: Alice's final depth phase arrival picks as an array ([1] - pP, [2] - sP; values are zero if no phase picked)
            
        Returns:
            self.pmP_pP_diff: modelled pmP-pP arrival time difference
            self.subarray_pmP_rel_time: modelled pmP arrival calculated relative to Alice's depth phase pick, converted to time relative to event (s)
        """
        
        pmP_m = self.subarray_pmP_time
        pP_m = self.subarray_pP_revised
        outputs = self.outputs
        
        DP_picks = outputs.phase_id_picks
        
        # Convert pP pick from pts into arrival time wrt event
        
        trim_start = self.outputs.trim_start
        rsample = self.outputs.resample
        
        pP_pick_sec = (DP_picks[1]/rsample)+trim_start
        P_pick_sec = (DP_picks[0]/rsample)+trim_start
        
        # Calculate the modelled arrival time difference between depth phase and precursor
        
        pmP_pP_diff_m = pP_m - pmP_m
            
        self.pmP_pP_diff_m = pmP_pP_diff_m
                
        # Predict pmP arrivals from depth phase arrivals
        
        if DP_picks[1] != 0:    # Check that pP pick exists; if yes, use it to calculate pmP_rel
            pmP_rel = pP_pick_sec - pmP_pP_diff_m            
        else:                   # If no pP pick exists, set pmP_rel to directly modelled pmP. SHOULD NOT HAPPEN IF pP HAS BEEN CHECKED ALREADY
            pmP_rel = pmP_m
            print("No pmP arrival time could be predicted relative to depth phase.")
               
        self.subarray_pmP_rel_time = pmP_rel
        
        # Calculate difference between P pick and pmP_rel model arrival for defining the search window later
        
        pmP_P_diff_m = pmP_rel - P_pick_sec
        
        self.pmP_P_diff_m = pmP_P_diff_m
        
        return

#==============================================================================
    def predict_smP_arrival_relative_to_DP(self):
        
        """
        This method predicts the relative arrival time for the smP phase by first calculating 
        the difference between the modelled smP and pP arrivals, then subtracting this
        difference from Alice's actual pP pick. This is done because the smP/pP
        difference is sensitive only to the crust and not the event depth or any
        far away velocity structure, so is potentially better for defining a search window
        for pmP. 
        
        Inputs:
            self.subarray_smP_time: modelled pmP arrival time wrt event
            self.subarray_pP_revised: modelled pP arrival time wrt event using the improved event depth

            outputs.final_picks: Alice's final depth phase arrival picks as an array ([1] - pP, [2] - sP; values are zero if no phase picked)
            
        Returns:
            self.smP_pP_diff: modelled smP-pP arrival time difference
            self.subarray_smP_rel_time: modelled pmP arrival calculated relative to Alice's depth phase pick, converted to time relative to event (s)
        """
        
        smP_m = self.subarray_smP_time
        pP_m = self.subarray_pP_revised
        outputs = self.outputs
        
        DP_picks = outputs.phase_id_picks
        
        # Convert pP pick from pts into arrival time wrt event
        
        trim_start = self.outputs.trim_start
        rsample = self.outputs.resample
        
        pP_pick_sec = (DP_picks[1]/rsample)+trim_start
        P_pick_sec = (DP_picks[0]/rsample)+trim_start
        
        # Calculate the modelled arrival time difference between depth phase and precursor
        
        smP_pP_diff_m = pP_m - smP_m
            
        self.smP_pP_diff_m = smP_pP_diff_m
                
        # Predict pmP arrivals from depth phase arrivals
        
        if DP_picks[1] != 0:    # Check that pP pick exists; if yes, use it to calculate pmP_rel
            smP_rel = pP_pick_sec - smP_pP_diff_m            
        else:                   # If no pP pick exists, set pmP_rel to directly modelled pmP. SHOULD NOT HAPPEN IF pP HAS BEEN CHECKED ALREADY
            smP_rel = smP_m
            print("No smP arrival time could be predicted relative to depth phase.")
               
        self.subarray_smP_rel_time = smP_rel
        
        '''# Calculate difference between P pick and smP_rel model arrival for defining the search window later
        
        smP_P_diff_m = smP_rel - P_pick_sec
        
        self.smP_P_diff_m = smP_P_diff_m'''
        
        return
#==============================================================================          
    def find_pmP_picks_window(self):
        """
        This method picks potential pmP signals using a search window around the predicted pmP arrival
        to identify peaks of interest on envelope data.
        
        Inputs:
            
        self.outputs.phase_weighted_beam: phase weighted beamforms per slowness tested (in obspy stream format)     
        self.outputs.slowness_index = position in the test slowness array where the slownesss found via beampacking exists
        self.outputs.trim_start: starting time for trimming traces (s)
        self.outputs.rsample: sample rate (Hz)
        self.outputs.final_picks: Alice's final picks for P, pP and sP (zero if no picked phase)
          
        self.outputs.final_picks: Alice's P, pP and sP picks
        self.subarray_pmP_time: predicted pmP arrival from direct modelling
        self.subarray_pmP_rel_time: predicted pmP arrival from back-calculating from pmP
        
        Returns:
            
        self.pmP_window_picks: potential pmP picks in the search window
        self.pmP_window_picks_time: potential pmP picks converted to time in s from event origin time
        
        """
        # Extract variables
        
        outputs = self.outputs
        
        stream = self.outputs.phase_weighted_beams
        slow_index = self.outputs.slowness_index
        trim_start = self.outputs.trim_start
        rsample = self.outputs.resample
        DP_picks = self.outputs.phase_id_picks
        
        pmP_m = self.subarray_pmP_time
        pmP_rel_m = self.subarray_pmP_rel_time
        pmP_pP_diff = self.pmP_pP_diff_m
        pmP_P_diff = self.pmP_P_diff_m
        
        evdepth = self.evdepth
        crthk = self.crthk
       
# ---------- PRELIMINARIES ----------
                
        stream.normalize()

        # Convert Alice's P, pP picks from pts into time from event origin
        
        P_time = (DP_picks[0]/rsample)+trim_start
        pP_time = (DP_picks[1]/rsample)+trim_start
                
        # Trim data, to only consider peaks in the area of interest (between P and pP arrivals)

        starttime = self.outputs.event.origin_time + (P_time * 0.98)
        endtime = self.outputs.event.origin_time + (pP_time * 1.02)
        x_axis_time_addition = ((P_time * 0.98) - trim_start)*rsample

        beam_trimmed = copy.deepcopy(stream[slow_index])
        beam_trimmed.trim(starttime, endtime, pad=True, fill_value=0)                   # Trim the beamforms in the stream to only look at the time interval of interest

        envelope_trimmed = obspy.signal.filter.envelope(beam_trimmed.data)
        
        max_peak = np.max(envelope_trimmed)                                             # Finds maximum peak in enveloped stream (should be P)
        peaks_tmp, properties = find_peaks(envelope_trimmed, prominence=0.05*max_peak)  # Use find_peaks function to detect peaks in the beamforms that have a prominence of at least 0.05 of the max peak (should be P)
        width = peak_widths(envelope_trimmed, peaks_tmp, rel_height=0.9) # Find widths of peaks to use as lower bound window buffer later
        #print('WIDTH', width)

        '''# Briefly plot peaks and widths for check
        x = np.arange(0, len(envelope_trimmed),1)
        plt.plot(x, envelope_trimmed)
        plt.scatter(peaks_tmp, envelope_trimmed[peaks_tmp])
        plt.hlines(*width[1:])
        plt.savefig('env.png')'''

        self.window_picking_threshold = 0.05*max_peak                                   # Save the window picking threshold value into Cr_subarray for plotting
        #print('f')         
# ---------- FINDING THE PICKS ----------     

        if len(peaks_tmp) == 0:
            #print("No peaks were identified in trace for subarray ", outputs.array_no)
            self.pmP_window_picks = np.zeros(1)            
            self.pmP_window_picks_time = np.zeros(1)
            return

        peaks = [0] * len(peaks_tmp)          
        for i in range (len(peaks)):                                                    # Convert trimmed x-axis peak locations to the untrimmed data x-axis
            peaks[i] = peaks_tmp[i] + x_axis_time_addition
        # Find closest peak to P pick, extract its width/wavelength in samples
        #print(DP_picks, peaks)
        pmp_candidates = [0]*len(peaks)
        for i in range (len(peaks)):
            pmp_candidates[i] = np.abs(peaks[i] - DP_picks[0])
        idx = np.argmin(pmp_candidates)
        P_lamda = width[0][idx]
        self.P_lamda = P_lamda
        
# ---------- DEFINING SEARCH WINDOW --------- 
        
        # Search window defined as 3s (3 s window is based off traveltime tables and corresponds to crustal thickness range of ca +-10km)
        window_interval = 2.5 * rsample
        #window = 2 * (3 * rsample) # playing with this...
        window = 2 * (window_interval)       

        # Define window upper and lower bounds using additional constrains (3s pad with pP and P arrivals)
                
        # Upper bound  - 3s of pP pick
        pmP_end = ((pmP_rel_m-trim_start)*rsample) + window/2

        if pmP_end > (DP_picks[1] - 0.5*P_lamda):
            pmP_end = (DP_picks[1] - 0.5*P_lamda) #3 seconds in sample pts
        else:
            pass

        # Lower bound  - 1/2 of time between pP and P [change to 1/2 P wavelength]
        # Find wavelength of P 
            
        #print('P_lamda', P_lamda)

        pmP_start = ((pmP_rel_m-trim_start)*rsample) - window/2
        
        if pmP_start < (DP_picks[0] + 0.5*P_lamda):
            pmP_start = (DP_picks[0] + 0.5*P_lamda)
            print('pmP Start', pmP_start, DP_picks[0] + 0.5*P_lamda, )
        else:
            pass
      
        if pmP_end <= pmP_start or pmP_start > ((pmP_rel_m-trim_start)*rsample): #or if pmP arrival is before window start, too close to P...
            print("Search window could not be defined.")
            self.pmP_window_picks = np.zeros(1)            
            self.pmP_window_picks_time = np.zeros(1)
            self.window_upper = np.zeros(1)
            self.window_lower = np.zeros(1)
            self.upper_window_condition = np.zeros(1)
            self.lower_window_condition = np.zeros(1)
            return

        # Convert window bounds to relative time and save for plotting
        pmP_end_time = ((pmP_end/rsample) + trim_start)
        pmP_start_time = ((pmP_start/rsample) + trim_start)
       
        self.window_upper = pmP_end_time
        self.window_lower = pmP_start_time
        self.upper_window_condition = DP_picks[0] + 0.5*P_lamda
        self.lower_window_condition = DP_picks[1] - window_interval

        # Initialise list for potential picks within the window        
        pmP_picks = []                                                                         
        if DP_picks[1] != 0:
            for i in range (len(peaks)):                                                # Check peaks are in the window 
                if peaks[i] > pmP_start and peaks[i] < pmP_end:
                    pmP_picks = np.append(pmP_picks, peaks[i])
                    
        if len(pmP_picks) == 0:
            #print("No signals picked within window for subarray ", outputs.array_no)
            self.pmP_window_picks = np.zeros(1)            
            self.pmP_window_picks_time = np.zeros(1)
            return
  
        pmP_picks_rel_time = [0] * len(pmP_picks)                                       # Convert pmP picks to arrival times relative to event
        for i in range(0, len(pmP_picks)):
            pmP_picks_rel_time[i] = (pmP_picks[i]/rsample)+trim_start
        
        #print("Potential pmP arrivals picked within window for subarray ", outputs.array_no, " :", pmP_picks_rel_time)

        self.pmP_window_picks = pmP_picks
        self.pmP_window_picks_time = pmP_picks_rel_time
        
        return
#==============================================================================
    def signal_to_noise_check(self):
        
        """
        This method checks that the pmP window picks are above some reasonable
        noise threshold. The threshold used here is background noise taken from trace
        before the P-arrival.
        
        Returns:
            self.snr_filtered_picks: window picks that passed the signal-to-noise filter
            self.snr_filtered_picks_time: window picks that passed the signal-to-noise filter, converted to time wrt event origin
        """
        
        # Extract variables
        
        outputs = self.outputs
        
        stream = self.outputs.phase_weighted_beams
        slow_index = self.outputs.slowness_index
        trim_start = self.outputs.trim_start
        rsample = self.outputs.resample
        envelope = self.outputs.PW_optimum_beam_envelope
        
        DP_picks = self.outputs.phase_id_picks
        P_time = (DP_picks[0]/rsample)+trim_start
        
        picks = self.pmP_window_picks
        
# ---------- DEFINE BACKGROUND NOISE THRESHOLD ----------

        # Trim region in front of P arrival to use in defining background noise level; use a 40-second interval before P onset
        
        endtime = outputs.event.origin_time + (P_time * 0.98)
        starttime = endtime - 40

        # Save the mean noise level 

        pre_arrival_noise = copy.deepcopy(stream[slow_index])
        pre_arrival_noise.trim(starttime, endtime, pad=True, fill_value=0)

        noise_std = np.std(abs(pre_arrival_noise.data))                         # calculate standard deviation of the background noise
        
        mean_noise = np.mean(abs(pre_arrival_noise.data)+ (noise_std*2))        # calculate mean noise as absolute value of pre-P arrival noise + two standard deviations
        
# ---------- FILTER PICKS AND SAVE PICKS THAT PASS THE SNR FILTER
                
        snr = [0] * len(picks) 
        snr_filtered_picks = []
        
        # Check that the picks have a signal-to-noise ratio of at least 8, then save picks that pass the test into snr_filtered_picks
        
        for i in range (len(picks)):                            
            snr[i] = envelope[int(picks[i])]/mean_noise
            if snr[i] >= 8:
                snr_filtered_picks.append(picks[i])

        # Convert the filtered picks into absolute time from event origin
        
        snr_filtered_picks_time = [0] * len(snr_filtered_picks)
        for i in range(0, len(snr_filtered_picks)):
            snr_filtered_picks_time[i] = (snr_filtered_picks[i]/rsample)+trim_start
        
        #print('Remaining picks after signal-to-noise filtering: ', snr_filtered_picks_time)

        # Save the snr of picks that passed the filter

        snr_of_filtered_picks = []
        for i in range(len(snr)):
            if snr[i] >= 8:
                snr_of_filtered_picks.append(snr[i])

        #print("Signal-to-noise ratio(s) of the remaining pick(s): ", snr_of_filtered_picks)
                
        self.snr_filtered_picks = snr_filtered_picks
        self.snr_filtered_picks_time = snr_filtered_picks_time

#------------------------------------------------------------------------------
        
        # If only one pick passed the SNR filter, save it as the pmP pick for the subarray
        if len(snr_filtered_picks) == 1:
            self.pmP_pick = snr_filtered_picks[0]
            self.pmP_pick_time = snr_filtered_picks_time[0]
            self.pmP_pick_exists = True
        
        # If multiple picks passed the SNR filter, save the one with the highest signal-to-noise ratio as the pmP pick. ONLY FOR TESTING! 
        elif len(snr_filtered_picks) > 1:
            max_idx = snr_of_filtered_picks.index(max(snr_of_filtered_picks))
            self.pmP_pick = snr_filtered_picks[max_idx]
            self.pmP_pick_time = snr_filtered_picks_time[max_idx]
            self.pmP_pick_exists = True
            
        else:
            return
        
        # Calculate the delay time between pmP and pP picks, save into subarray
                
        pP_pmP_delay_time = (DP_picks[1] - self.pmP_pick)/rsample
        self.pP_pmP_delay_time = pP_pmP_delay_time  
               
        return
#==============================================================================

    def extract_amplitudes_PW(self):

        outputs = self.outputs
        envelope = self.outputs.PW_optimum_beam_envelope
        
        DP_picks = self.outputs.phase_id_picks

        P_index = DP_picks[0]
        pmP_index = self.pmP_pick
        pP_index = DP_picks[1]
        
        P_amplitude = envelope[int(P_index)]
        pmP_amplitude = envelope[int(pmP_index)]
        pP_amplitude = envelope[int(pP_index)]
        
        self.P_pw_amplitude = P_amplitude
        self.pmP_pw_amplitude = pmP_amplitude
        self.pP_pw_amplitude = pP_amplitude        
        
        return

    def extract_amplitudes(self):

        outputs = self.outputs
        slow_index = self.outputs.slowness_index
        beam = self.outputs.beams[slow_index]
        envelope = obspy.signal.filter.envelope(beam.data)
        
        DP_picks = self.outputs.phase_id_picks

        P_index = DP_picks[0]
        pmP_index = self.pmP_pick
        pP_index = DP_picks[1]
        
        P_amplitude = envelope[int(P_index)]
        pmP_amplitude = envelope[int(pmP_index)]
        pP_amplitude = envelope[int(pP_index)]
        
        self.P_amplitude = P_amplitude
        self.pmP_amplitude = pmP_amplitude
        self.pP_amplitude = pP_amplitude        
        
        return


#==============================================================================
    def get_model_pierce_points(self, crustal_thickness, model):
        
        """
        Returns bounce points of modelled pP and pmP raypaths for the event-subarray pair.
        Uses the first-pass crustal thickness extracted from Crust1, and the custom
        velocity model built earlier.
        
        Inputs:
            self: event and subarray coordinates, event depth
            crustal_thickness: Crust1 first pass crustal thickness
            model: the custom model calculated earlier in the code using the first-pass crustal thickness
            
        Outputs:
            self.pmP_bounce_lat_m = pmP bounce point latitude, first pass
            self.pmP_bounce_lon_m = pmP bounce point longitude, first pass
            self.pP_bounce_lat_m = pP bounce point latitude, first pass
            self.pP_bounce_lon_m = pP bounce point longitude, first pass
            
        """
        
        # Extract variables
        
        outputs = self.outputs
        
        evla = outputs.event.evla
        evlo = outputs.event.evlo
        arr_la = outputs.array_latitude
        arr_lo = outputs.array_longitude
        
        evdepth = self.evdepth
        
        crthk = crustal_thickness
        model=model
    
        # Calculate pierce points for the event-subarray pair
            
        arrivals = model.get_pierce_points_geo(source_depth_in_km = evdepth,
                                              source_latitude_in_deg = evla,
                                              source_longitude_in_deg = evlo,
                                              receiver_latitude_in_deg = arr_la,
                                              receiver_longitude_in_deg = arr_lo,
                                              phase_list=["P", "p^"+str(int(crthk))+"P", "pP"])
        pmP_arr = arrivals[1]
        pP_arr = arrivals[2]
        
        # Extract bounce point coordinates for pmP
        
        for i in range(len(pmP_arr.pierce)):
            if pmP_arr.pierce[i]['depth'] == crthk:
                print(pmP_arr.pierce[i], hasattr(pmP_arr,'pierce_points'))
                pmP_bounce_lat = pmP_arr.pierce[i]['lat']
                pmP_bounce_lon = pmP_arr.pierce[i]['lon']
                #print("Coordinates of the pmP bounce point at the Moho are ", pmP_bounce_lat, ", ", pmP_bounce_lon)
                break
        
        self.pmP_bounce_lat_m = pmP_bounce_lat
        self.pmP_bounce_lon_m = pmP_bounce_lon
        
        # Extract bounce point coordinates for pP
        
        for i in range(len(pP_arr.pierce)):
            if pP_arr.pierce[i]['depth'] == 0:
                pP_bounce_lat = pP_arr.pierce[i]['lat']
                pP_bounce_lon = pP_arr.pierce[i]['lon']
                #print("Coordinates of the pP bounce point at the surface are ", pP_bounce_lat, ", ", pP_bounce_lon)
                break
        
        self.pP_bounce_lat_m = pP_bounce_lat
        self.pP_bounce_lon_m = pP_bounce_lon
        
        return
#==============================================================================        
    def get_improved_pierce_points(self, crustal_velocity_model, vel_dir, build_velocity_model_function = None):
        
        """
        Returns bounce points of pP and pmP raypaths for the event-subarray pair.
        Uses the crustal thickness value calculated from the pmP-pP delay times, stored
        in the Cr_subarray object; it builds a custom velocity model with this new calculated crustal
        thickness value, and predicts bounce points using that velocity model.
        
        Inputs:
            self: event and subarray coordinates, event depth
            self.new_crthk_FM: improved crustal thickness estimate from forward modelling approach
            
        Outputs:
            self.pmP_bounce_lat = pmP bounce point latitude, improved
            self.pmP_bounce_lon = pmP bounce point longitude, improved
            self.pP_bounce_lat = pP bounce point latitude, improved
            self.pP_bounce_lon = pP bounce point longitude, improved
        """
        
        # Extract variables
        
        outputs = self.outputs
        
        evla = outputs.event.evla
        evlo = outputs.event.evlo
        arr_la = outputs.array_latitude
        arr_lo = outputs.array_longitude
        
        evdepth = self.evdepth
        crthk = self.new_crthk_FM # CHANGE BACK TO FM
        
        print("Now finding new bounce point locations for successful subarrays.")
        
        # Build new velocity model with the improved crustal thickness value, calling a function from the custom functions script to do it (use lazy import to get around circularity issue here)
        
        if build_velocity_model_function is None:
            from Functions_for_crthk_code import Build_velocity_model
            
        model = Build_velocity_model(crthk, self.outputs.event.evname, crustal_velocity_model, vel_dir)
    
        # Calculate pierce points for the event-subarray pair
            
        arrivals = model.get_pierce_points_geo(source_depth_in_km = evdepth,
                                              source_latitude_in_deg = evla,
                                              source_longitude_in_deg = evlo,
                                              receiver_latitude_in_deg = arr_la,
                                              receiver_longitude_in_deg = arr_lo,
                                              phase_list=["P", "p^"+str(int(crthk))+"P", "pP"])
        pmP_arr = arrivals[1]
        pP_arr = arrivals[2]
        
        # Extract bounce point coordinates for pmP
        
        for i in range(len(pmP_arr.pierce)):
            if pmP_arr.pierce[i]['depth'] == crthk:
                pmP_bounce_lat = pmP_arr.pierce[i]['lat']
                pmP_bounce_lon = pmP_arr.pierce[i]['lon']
                #print("Coordinates of the pmP bounce point at the Moho are ", pmP_bounce_lat, ", ", pmP_bounce_lon)
                break
        
        self.pmP_bounce_lat = pmP_bounce_lat
        self.pmP_bounce_lon = pmP_bounce_lon
        
        # Extract bounce point coordinates for pP
        
        for i in range(len(pP_arr.pierce)):
            if pP_arr.pierce[i]['depth'] == 0:
                pP_bounce_lat = pP_arr.pierce[i]['lat']
                pP_bounce_lon = pP_arr.pierce[i]['lon']
                #print("Coordinates of the pP bounce point at the surface are ", pP_bounce_lat, ", ", pP_bounce_lon)
                break
        
        self.pP_bounce_lat = pP_bounce_lat
        self.pP_bounce_lon = pP_bounce_lon
        
        return
#==============================================================================
    def calculate_crustal_thickness_ray_param(self):
        """
        Calculates crustal thickness from the pmP-pP arrival time difference
        using the approach from the 2008 McGlashan et al paper.
        
        Returns:
            self.new_crthk_RP: new improved crustal thickness value from ray parameter calculation
        
        """
        # Extract variables
        
        rsample = self.outputs.resample
        
        pP_rp = self.pP_ray_parameter
        pmP_rp = self.pmP_ray_parameter
        
        DP_picks = self.outputs.phase_id_picks
        pmP_pick = self.pmP_pick
        
        # Specify uniform crustal velocity to be used in the calculation
        
        Vp_cr = 6.3                                              
        
        # Check that pmP pick exists
        
        if pmP_pick == 0:
            print("Cannot calculate crustal thickness, no pmP pick is stored in subarray.")
            return
        else: 
            pass
        
        # Calculate the delay time between pmP and pP
        
        pP_pmP_delay_time = (DP_picks[1] - pmP_pick)/rsample
        print("pP-pmP delay time:", pP_pmP_delay_time)
        
        # Convert ray parameters from s/deg to s/km
        
        pP_rp = pP_rp * (1/(111))
        pmP_rp = pmP_rp * (1/(111))
        
        # Convert the delay time into crustal thickness. Use the ray parameter of pmP in the calculation
        
        improved_crthk_pmP = (pP_pmP_delay_time/2) * (1/(np.sqrt((1/Vp_cr)*(1/Vp_cr)-pmP_rp*pmP_rp)))
        
        # Convert the delay time into crustal thickness. Use the ray parameter of pP in the calculation
        
        improved_crthk_pP = (pP_pmP_delay_time/2) * (1/(np.sqrt((1/Vp_cr)*(1/Vp_cr)-pP_rp*pP_rp)))
        
        print("Crustal thickness calculated from pmP-pP delay, with crustal P-velocity", Vp_cr, "km/s:", "\nUsing pmP ray parameter: ", improved_crthk_pmP, "\nUsing pP ray parameter: ", improved_crthk_pP)
        
        self.new_crthk_RP = improved_crthk_pmP 
        
        return
    
#==============================================================================    
    def plot_pmP_window_picking_figure(self, fig_dir):
        
        """
        Returns plots of vespagram and optimum slowness beamformed trace with
        preliminary pmP window picks marked within the search window, along with P and
        depth phase picks and predicted pmP (and smP) arrivals.
        
        Returns:
        
        self.window_picking_figure: figure with potential pmP picks within window around the modelled arrival
    
        """
        # Extract variables
        
        P_time = self.subarray_P_revised
        sP_time = self.subarray_sP_revised
        pmP_m = self.subarray_pmP_time
        pmP_rel_m = self.subarray_pmP_rel_time
        pmP_picks = self.pmP_window_picks
        smP_rel_m = self.subarray_smP_rel_time
        
        smP_m = self.subarray_smP_time
        window_u = self.window_upper
        window_l = self.window_lower
        window_uc = self.upper_window_condition
        window_lc = self.lower_window_condition
        outputs = self.outputs
                
        vespa_grd = outputs.vespa_grd
        stream = outputs.phase_weighted_beams
        relative_time = outputs.relative_time 
        trim_start = outputs.trim_start
        #trim_interval = outputs.trim_interval
        rsample = outputs.resample
        slow_index = outputs.slowness_index
        slow = outputs.slowness_range
        envelope = outputs.PW_optimum_beam_envelope
        
        threshold = self.window_picking_threshold                               # Picking threshold for the preliminary picks within window

        DP_picks = outputs.phase_id_picks
        
        DP_picks_time = [0] * len(DP_picks)                                     # Convert DP picks to arrival times relative to event, to use for plotting
        for i in range(0, len(DP_picks)):
            DP_picks_time[i] = (DP_picks[i]/rsample)+trim_start
        
        P_pick_time = DP_picks_time[0]
        pP_pick_time = DP_picks_time[1]
        sP_pick_time = DP_picks_time[2]

# ---------- PRELIMINARIES ----------
        
        print("Now plotting window picking figure for subarray ", outputs.array_no)
        
        x_axis_time_addition = int(((P_time * 0.98) - trim_start)*rsample)
        x_axis_end = int(((sP_time * 1.02) - trim_start)*rsample)
        time = np.arange(0, len(vespa_grd[0].data), 1)
        
        slowness_array = []
        for i in range (len(pmP_picks)):
            slowness_array.append(slow_index)
        
        x_tick_interval = 20
        x_ticks = np.arange(0, len(envelope), x_tick_interval*rsample)
        relative_time = np.arange(relative_time[0], relative_time[-1]+(x_tick_interval*2), x_tick_interval) 
        relative_time = np.round(relative_time,0)
        relative_time = relative_time[:len(x_ticks)]
        relative_time = [int(t) for t in relative_time]

        #xaxis_p = np.arange(0, len(vespa_grd[0]), 20*rsample)
        #relative_time = np.arange(trim_start, (trim_start + trim_interval)+40, 20) 
        #relative_time = relative_time[:len(xaxis_p)]
        yaxis_p = np.linspace(0, len(slow), 5)
        yaxis_l = np.around(np.linspace(slow[0], slow[-1], 5), 2)               # set y-axis labels
                            
        vmax = np.max(vespa_grd)                                                # find maximum value on the vespagram grid for plotting
        vmin = np.min(vespa_grd)
        if abs(vmin) >= abs(vmax):
            vmax= abs(vmin)
                
        xlim = ((P_time-trim_start)*rsample) - 400                              # set x-axis limits
        xlim_interval = xlim + 1600
             
# ---------- PLOT STARTS HERE ----------
                
        window_picking_figure, ax = plt.subplots(3, 1, sharex=True, figsize=(25,15))   # initialise figure object and a 2D array of axes (subplot) objects (2x1 array of axes, for two subplots in one column)
        plt.subplots_adjust(hspace=0.2, wspace=0.2)                                    # adjust spacing between subplots in the figure; hspace - vertical space (as fraction of subplot height), wspace - horizontal space
        
        axes = [ax[0], ax[1], ax[2]]
        
        # UPPER PLOT
        
        ax[0].set_title('Event ' + str(outputs.event.evname) + ' Picking pmP %s' %outputs.ev_array_gcarc, fontsize=16)
        
        ax[0].pcolormesh(vespa_grd, cmap = 'seismic', vmin = (-vmax*0.2), vmax = (vmax*0.2), rasterized=True)
        ax[0].axhline(slow_index, linestyle = '--', color = 'k', zorder = 1, label = 'Optimum Beam')
        
        ax[0].set_ylabel('Slowness (s/km)', fontsize = 20)
        ax[0].set_xticks(x_ticks)
        ax[0].set_xlim(xlim, xlim_interval)
        ax[0].set_ylim(np.min(slow), np.max(slow))
        ax[0].set_yticks(yaxis_p)
        ax[0].set_yticklabels(yaxis_l, fontsize=16)
                
        if len(pmP_picks) >= 1:    
            ax[0].scatter(pmP_picks, slowness_array , color='gold', edgecolor = 'k',linewidth = 1, s=100, label='pmP window picks', zorder = 2)

        # MIDDLE PLOT

        vmax = np.max(envelope)
       
        ax[1].plot(time, stream[slow_index], color = 'k', linewidth = '1', linestyle = '--', zorder = 0.5, label = 'Optimum Beam')
        ax[1].plot(time, envelope, linewidth = '2', zorder = 1, color='k', label = 'Envelope')
        
        ax[1].set_ylabel('Velocity (m/s)', fontsize = 20)
              
        yaxis_l = [-1, -0.5, 0, 0.5, 1]
        ax[1].set_ylim([-vmax*1.5, vmax*1.5])
        ax[1].set_yticks(np.linspace(-vmax, vmax, 5))
        ax[1].set_yticklabels(yaxis_l, fontsize=16)    
        ax[1].set_xticklabels(relative_time, rotation=-45, ha='left', fontsize=16)
        
        ax[1].axhline(threshold, linestyle = '--', color='k', label='Window picking threshold')
        
        if len(pmP_picks) >= 1:   
            ax[1].scatter(pmP_picks, envelope[pmP_picks.astype(int)], color='gold', edgecolor = 'k',linewidth = 1, s=100, label='pmP window picks', zorder = 2)
        
        # LOWER PLOT

        # enhance the amplitude of the beampacked trace for plotting
        scaling_factor = 20
        
        st = stream[slow_index].copy()
        st.data = st.data * scaling_factor
        
        # enhance the amplitude of the envelope for plotting
        en = envelope.copy()
        en = en * scaling_factor
       
        ax[2].plot(time, st, color = 'k', linewidth = '2.5', zorder = 1, label = 'Optimum Beam, enhanced amplitude')
        ax[2].plot(time, en, linewidth = '2', linestyle = '--', zorder = 0.5, color='k', label = 'Envelope')
        ax[2].set_ylabel('Velocity (m/s)', fontsize = 20)

        yaxis_l = [-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75]

        ax[2].set_xticklabels(relative_time, rotation=-45, ha='left', fontsize=16)

        vmax = np.max(st.data) * 0.5
        ax[2].set_ylim([-vmax, vmax])
        ax[2].set_yticks(np.linspace(-vmax, vmax, 7))
        ax[2].set_yticklabels(yaxis_l, fontsize=16)

        if len(pmP_picks) >= 1:   
            ax[2].scatter(pmP_picks, en[pmP_picks.astype(int)], color='gold', edgecolor = 'k',linewidth = 1, s=100, label='pmP window picks', zorder = 2)
        
        # ALL PLOTS
      
        for ax in axes:
            
            # Phase arrival time markers
            ax.scatter(((P_pick_time-trim_start)*rsample), 0 , marker = '^', s = 400, color = 'k', label='P, pP and sP picks', transform=ax.get_xaxis_transform()) # modelled arrival locations on plot
            ax.scatter(((pP_pick_time-trim_start)*rsample), 0, marker = '^', s = 400, color = 'k', transform=ax.get_xaxis_transform())
            ax.scatter(((sP_pick_time-trim_start)*rsample), 0, marker = '^', s = 400, color = 'k', transform=ax.get_xaxis_transform())
            ax.scatter(((pmP_rel_m-trim_start)*rsample), 0, marker = '^', s = 400, color = 'r', zorder = 2, label='Modelled pmP arrival, relative to DP', transform=ax.get_xaxis_transform())
            ax.scatter(((pmP_m-trim_start)*rsample), 0, marker = '^', s = 400, color = 'y', zorder = 1.5, label='Modelled pmP arrival, direct', transform=ax.get_xaxis_transform())
            ax.scatter(((smP_m-trim_start)*rsample), 0, marker = '^', s = 400, color = 'm', label='Modelled smP arrival, direct', transform=ax.get_xaxis_transform())
            ax.scatter(((smP_rel_m-trim_start)*rsample), 0, marker = '^', s = 400, color = 'm', zorder = 2, label='Modelled smP arrival, relative to DP', transform=ax.get_xaxis_transform())
        
            # Search window
            ax.axvline(((window_u-trim_start)*rsample), color = 'gray', linestyle = '--', label = 'Picking window')
            ax.axvline(((window_l-trim_start)*rsample), color = 'gray', linestyle = '--')
            ax.axvline(window_uc, color = 'red', linestyle = '--')
            ax.axvline(window_lc, color = 'red', linestyle = '--')
      
            # Tick parameters
            ax.tick_params(width=1, length = 8)

            # Legends
            ax.legend(loc='upper right', fontsize=15)

        
        # SAVE FIGURE
        
        #fig_dir = os.path.join(os.path.join(res_dir, '%s' %outputs.event.evname), 'Crust_code_figures')
        
        fig_name = str(outputs.event.evname) + '_Picking_pmP_%s.png' %outputs.ev_array_gcarc
        path = os.path.join(fig_dir, fig_name)
        window_picking_figure.savefig(path, dpi=500)
        
        plt.close()
        
        return window_picking_figure

#============================================================================== 

#==============================================================================    
    def plot_pmP_window_picking_figure_paper(self, fig_dir):
        
        """
        Returns plots of vespagram and optimum slowness beamformed trace with
        preliminary pmP window picks marked within the search window, along with P and
        depth phase picks and predicted pmP (and smP) arrivals.
        
        Returns:
        
        self.window_picking_figure: figure with potential pmP picks within window around the modelled arrival
    
        """
        # Extract variables
        
        P_time = self.subarray_P_revised
        sP_time = self.subarray_sP_revised
        pmP_m = self.subarray_pmP_time
        pmP_rel_m = self.subarray_pmP_rel_time
        pmP_picks = self.pmP_window_picks
        smP_rel_m = self.subarray_smP_rel_time
        smP_m = self.subarray_smP_time

        window_u = self.window_upper
        window_l = self.window_lower
        window_uc = self.upper_window_condition
        window_lc = self.lower_window_condition
        outputs = self.outputs
                
        vespa_grd = outputs.vespa_grd
        stream = outputs.phase_weighted_beams
        relative_time = outputs.relative_time 
        trim_start = outputs.trim_start
        #trim_interval = outputs.trim_interval
        rsample = outputs.resample
        slow_index = outputs.slowness_index
        slow = outputs.slowness_range
        envelope = outputs.PW_optimum_beam_envelope
        
        threshold = self.window_picking_threshold                               # Picking threshold for the preliminary picks within window

        DP_picks = outputs.phase_id_picks
        
        DP_picks_time = [0] * len(DP_picks)                                     # Convert DP picks to arrival times relative to event, to use for plotting
        for i in range(0, len(DP_picks)):
            DP_picks_time[i] = (DP_picks[i]/rsample)+trim_start
        
        P_pick_time = DP_picks_time[0]
        pP_pick_time = DP_picks_time[1]
        sP_pick_time = DP_picks_time[2]

# ---------- PRELIMINARIES ----------
        
        print("Now plotting window picking figure for subarray ", outputs.array_no)
        dt = np.round(self.pP_pmP_delay_time,2)
        thickness = np.round(self.new_crthk_FM,2)
        
        x_axis_time_addition = int(((P_time * 0.98) - trim_start)*rsample)
        x_axis_end = int(((sP_time * 1.02) - trim_start)*rsample)
        time = np.arange(0, len(vespa_grd[0].data), 1)
        
        slowness_array = []
        for i in range (len(pmP_picks)):
            slowness_array.append(slow_index)

        x_tick_interval = 20
        x_ticks = np.arange(0, len(envelope), x_tick_interval*rsample)
        relative_time = np.arange(relative_time[0], relative_time[-1]+(x_tick_interval*2), x_tick_interval) 
        relative_time = np.round(relative_time,0)
        relative_time = relative_time[:len(x_ticks)]
        relative_time = [int(t) for t in relative_time]

        #xaxis_p = np.arange(0, len(vespa_grd[0]), 20*rsample)
        #relative_time = np.arange(trim_start, (trim_start + trim_interval)+40, 20) 
        #relative_time = relative_time[:len(xaxis_p)]
        yaxis_p = np.linspace(0, len(slow), 5)
        yaxis_l = np.around(np.linspace(slow[0], slow[-1], 5), 2)               # set y-axis labels
                            
        vmax = np.max(vespa_grd)                                                # find maximum value on the vespagram grid for plotting
        vmin = np.min(vespa_grd)
        if abs(vmin) >= abs(vmax):
            vmax= abs(vmin)
                
        xlim = ((P_time-trim_start)*rsample) - 200                              # set x-axis limits
        xlim_interval = xlim + 800
             
# ---------- PLOT STARTS HERE ----------
                
        window_picking_figure, ax = plt.subplots(2, 1, sharex=True, figsize=(8,10))   # initialise figure object and a 2D array of axes (subplot) objects (2x1 array of axes, for two subplots in one column)
        plt.subplots_adjust(hspace=0.2, wspace=0.2)                                    # adjust spacing between subplots in the figure; hspace - vertical space (as fraction of subplot height), wspace - horizontal space
        
        axes = [ax[0], ax[1]]
        
        # UPPER PLOT
        #ax[0].set_title('Event ' + str(outputs.event.evname) + ' Picking pmP %s' %outputs.ev_array_gcarc, fontsize=16)
        
        cb = ax[0].pcolormesh(vespa_grd, cmap = 'seismic', vmin = (-vmax), vmax = (vmax), rasterized=True)
        ax[0].axhline(slow_index, linestyle = '--', color = 'k', zorder = 1, label = 'Optimum Beam')
        
        ax[0].set_ylabel('Slowness (s/km)', fontsize = 20)
        ax[0].set_xticks(x_ticks)
        ax[0].set_xlim(xlim, xlim_interval)
        ax[0].set_ylim(np.min(slow), np.max(slow))
        ax[0].set_yticks(yaxis_p)
        ax[0].set_yticklabels(yaxis_l, fontsize=16)
        #window_picking_figure.colorbar(cb)
        if len(pmP_picks) >= 1:    
            ax[0].scatter(pmP_picks, slowness_array , color='gold', edgecolor = 'k',linewidth = 1, s=50, label='pmP window picks', zorder = 2)
        ax[0].text(0.01, 0.92, 'pP-pmP: %s seconds' %(dt), fontsize = 16, transform=ax[0].transAxes)
        ax[0].text(0.01, 0.85, 'Crustal Thickness: %s km' %(thickness), fontsize = 16, transform=ax[0].transAxes)

        # MIDDLE PLOT

        vmax = np.max(envelope)
       
        ax[1].plot(time, stream[slow_index], color = 'k', linewidth = '1', linestyle = '--', zorder = 0.5, label = 'Optimum Beam')
        ax[1].plot(time, envelope, linewidth = '2', zorder = 1, color='k', label = 'Envelope')
        
        ax[1].set_ylabel('Velocity (m/s)', fontsize = 20)
              
        yaxis_l = [-1, -0.5, 0, 0.5, 1]
        ax[1].set_ylim([-vmax*1.1, vmax*1.1])
        ax[1].set_yticks(np.linspace(-vmax, vmax, 5))
        ax[1].set_yticklabels(yaxis_l, fontsize=16)    
        ax[1].set_xticklabels(relative_time, rotation=-45, ha='left', fontsize=16)
        
        ax[1].axhline(threshold, linestyle = '--', color='k', label='Window picking threshold')
        ax[1].set_xlabel('Time (s)', fontsize=20)
        if len(pmP_picks) >= 1:   
            ax[1].scatter(pmP_picks, envelope[pmP_picks.astype(int)], color='gold', edgecolor = 'k',linewidth = 1, s=50, label='pmP window picks', zorder = 2)
        #ax[1].legend(bbox_to_anchor=(1.05,1),loc='upper left')

        '''
        # LOWER PLOT

        # enhance the amplitude of the beampacked trace for plotting
        scaling_factor = 20
        
        st = stream[slow_index].copy()
        st.data = st.data * scaling_factor
        
        # enhance the amplitude of the envelope for plotting
        en = envelope.copy()
        en = en * scaling_factor
       
        ax[2].plot(time, st, color = 'k', linewidth = '2.5', zorder = 1, label = 'Optimum Beam, enhanced amplitude')
        ax[2].plot(time, en, linewidth = '2', linestyle = '--', zorder = 0.5, color='k', label = 'Envelope')
        ax[2].set_ylabel('Velocity (m/s)', fontsize = 20)

        yaxis_l = [-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75]

        ax[2].set_xticklabels(relative_time, rotation=-45, ha='left', fontsize=16)

        vmax = np.max(st.data) * 0.5
        ax[2].set_ylim([-vmax, vmax])
        ax[2].set_yticks(np.linspace(-vmax, vmax, 7))
        ax[2].set_yticklabels(yaxis_l, fontsize=16)

        if len(pmP_picks) >= 1:   
            ax[2].scatter(pmP_picks, en[pmP_picks.astype(int)], color='gold', edgecolor = 'k',linewidth = 1, s=100, label='pmP window picks', zorder = 2)
        '''
        # ALL PLOTS
      
        for ax in axes:
            
            # Phase arrival time markers
            ax.scatter(((P_pick_time-trim_start)*rsample), 0.02 , marker = '^', s = 100, color = 'k', label='P, pP and sP picks', transform=ax.get_xaxis_transform()) # modelled arrival locations on plot
            ax.scatter(((pP_pick_time-trim_start)*rsample), 0.02, marker = '^', s = 100, color = 'k', transform=ax.get_xaxis_transform())
            ax.scatter(((sP_pick_time-trim_start)*rsample), 0.02, marker = '^', s = 100, color = 'k', transform=ax.get_xaxis_transform())
            ax.scatter(((pmP_rel_m-trim_start)*rsample), 0.02, marker = '^', s = 100, color = 'r', zorder = 2, label='Modelled pmP arrival, relative to P', transform=ax.get_xaxis_transform())
            #ax.scatter(((pmP_m-trim_start)*rsample), 0.02, marker = '^', s = 100, color = 'y', zorder = 1.5, label='Modelled pmP arrival, direct', transform=ax.get_xaxis_transform())
            #ax.scatter(((smP_m-trim_start)*rsample), 0.02, marker = '^', s = 100, color = 'm', label='Modelled smP arrival, direct', transform=ax.get_xaxis_transform())
            ax.scatter(((smP_rel_m-trim_start)*rsample), 0.02, marker = '^', s = 100, color = 'm', zorder = 2, label='Modelled smP arrival, relative to P', transform=ax.get_xaxis_transform())
        
            # Search window
            ax.axvline(((window_u-trim_start)*rsample), label = 'Picking window')
            ax.axvline(((window_l-trim_start)*rsample))
            ax.axvline(window_uc, color = 'red', linestyle = '--')
            ax.axvline(window_lc, color = 'red', linestyle = '--')
            print(outputs.ev_array_gcarc, ((window_u-trim_start)*rsample), window_uc)
                    
            # Tick parameters
            ax.tick_params(width=1, length = 8)
            ax.legend(bbox_to_anchor=(1.02,1),loc='upper left')

        # Legends
        #window_picking_figure.legend(bbox_to_anchor=(1.05,1),loc='upper left', fontsize=15)
        #ax[1].legend(bbox_to_anchor=(1.05,1),loc='upper left')

        # SAVE FIGURE
        #fig_dir = os.path.join(os.path.join(res_dir, '%s' %outputs.event.evname), 'Crust_code_figures') 
        fig_name = str(outputs.event.evname) + '_Picking_pmP_paper_%s.png' %outputs.ev_array_gcarc
        path = os.path.join(fig_dir, fig_name)
        
        window_picking_figure.savefig(path, dpi=500, bbox_inches='tight')
        #plt.show()
        plt.close()
        
        return window_picking_figure

#==============================================================================                          
                                              

