#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 15:24:16 2023

@author: ee18ab
"""

# ## IMPORT MODULES
import warnings
warnings.filterwarnings("ignore")

import math
import obspy
import os
import statistics
import sys
import numpy as np
import pandas as pd
import scipy
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import matplotlib.cm as cm

from obspy.core.stream import Stream
from obspy.signal.filter import bandpass
from obspy.geodetics import locations2degrees, degrees2kilometers
from obspy.imaging.cm import obspy_sequential
from obspy.core.util import AttribDict
from obspy import taup

from scipy.signal import hilbert
from scipy.fftpack import fft,ifft
from scipy.signal import find_peaks, peak_prominences, peak_widths

from sklearn.cluster import DBSCAN
from datetime import datetime
from copy import deepcopy

from sklearn.neighbors import BallTree
from sklearn.cluster import DBSCAN

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

mpl.rcParams['text.usetex'] = False

# IMPORT CLASSES
from classes import Earthquake_event
from classes import Stations
from classes import Array
from classes import Global
from classes import Figures
  
def make_figures(catalogue, event, component, data_file, results_file):

    input_no = event

    event = catalogue[int(input_no)-1]
    evla = event.origins[0].latitude
    evlo = event.origins[0].longitude    
    evdp = event.origins[0].depth/1000
    evm = event.magnitudes[0].mag
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

    mn = add_zeroes(mn)
    dd = add_zeroes(dd)
    hh = add_zeroes(hh)
    mm = add_zeroes(mm)
    ss = add_zeroes(ss)
    
    evname_obspyDMT = str(yyyy)+str(mn)+str(dd)+'_'+str(hh)+str(mm)+str(ss)
    evname = str(yyyy)+str(mn)+str(dd)+str(hh)+str(mm)+str(ss)
    print("Event names is:",evname)
    
    file_path = results_file + '/' + evname
    data_file = data_file + '/' + evname

    print('Making Figures for %s' %evname)

    if component == 'Z':
	    # Load saved output arrays
	    name = 'array_Z.npy'
	    path = os.path.join(file_path, name)
	    array_classes = np.load(path, allow_pickle=True)
	    phase = 'P'
	    print(len(array_classes))

    if component == 'T':
	    # Load saved output arrays
	    name = 'array_T.npy'
	    path = os.path.join(file_path, name)
	    array_classes = np.load(path, allow_pickle=True)
	    phase = 'S'
	    print(len(array_classes))
	    #end
	    
    # Figures directory 
    try:
        # Create folder
        directory = 'Array_Processing_Figures'
        fig_dir = os.path.join(file_path, directory)
        os.mkdir(fig_dir)
        print('Directory %s created' %directory )
            
    except FileExistsError:
        directory = 'Array_Processing_Figures'
        fig_dir = os.path.join(file_path, directory)
        pass

    if component == 'Z':
        try:
            # Create folder
            directory = 'Array_Processing_Figures/P'
            fig_dir = os.path.join(file_path, directory)
            os.mkdir(fig_dir)
            print('Directory %s created' %directory )
                
        except FileExistsError:
            directory = 'Array_Processing_Figures/P'
            fig_dir = os.path.join(file_path, directory)
            pass

    if component == 'T':
        try:
            # Create folder
            directory = 'Array_Processing_Figures/S'
            fig_dir = os.path.join(file_path, directory)
            os.mkdir(fig_dir)
            print('Directory %s created' %directory )
                
        except FileExistsError:
            directory = 'Array_Processing_Figures/S'
            fig_dir = os.path.join(file_path, directory)
            pass

    # Initiate Figures class
    for i in range (len(array_classes)):
        array_class = array_classes[i]
        figures = Figures(array_class=array_class)
       
        # Make Figures
        try:
            #beampacking_figure = figures.BP_polar_plot(phase=phase)       
            #timeshifted_traces = figures.Timeshift_Traces(phase=phase)
            #vespagram_fig = figures.Plain_Vespagram(phase=phase)
            #threshold = figures.Plot_picking_threshold(phase=phase)
            picking_figure = figures.Picking_Vespagram(phase=phase) 
            #comparison_grid = figures.beampack_vs_calculated_grid(phase=phase)
            #corr_fig = figures.x_corr(phase=phase) 
            #QC_fig = figures.QC_Vespagram(phase=phase)
            #final_vespa = figures.Final_Picks(phase=phase)
            beampack_beams = figures.Beams_and_Beampacking(phase=phase)
            vespas_combined = figures.Vespagrams(phase=phase)
            #QC_vespas_combined = figures.Vespagrams_and_QC(phase=phase)

        except Exception as e:
            print(e, ' Error on line {}'.format(sys.exc_info()[-1].tb_lineno))
            pass
        
        # Save Figures
        try:
            fig_name = 'Timeshifted_traces_%s.png' %array_class.ev_array_gcarc
            path = os.path.join(fig_dir, fig_name)
            timeshifted_traces.savefig(path, dpi=500, bbox_inches = 'tight')
        except Exception as e:
            pass
          
        try:
            fig_name = 'X_corr_%s.png' %array_class.ev_array_gcarc
            path = os.path.join(fig_dir, fig_name)
            corr_fig.savefig(path, dpi=500, bbox_inches = 'tight')
        except Exception as e:
            pass
      
        try: 
            fig_name = 'Beampacking_%s.png' %array_class.ev_array_gcarc
            path = os.path.join(fig_dir, fig_name)
            beampacking_figure.savefig(path, dpi=500, bbox_inches = 'tight')
        except Exception as e:
            print(e, ' Error on line {}'.format(sys.exc_info()[-1].tb_lineno))
            pass
          
        try: 
            fig_name = 'Compare_grid_%s.png' %array_class.ev_array_gcarc
            path = os.path.join(fig_dir, fig_name)
            comparison_grid.savefig(path, dpi=500, bbox_inches = 'tight')
        except Exception as e:
            print(e, ' Error on line {}'.format(sys.exc_info()[-1].tb_lineno))
            pass
        
        try:
            fig_name = 'Vespagram_%s.png' %array_class.ev_array_gcarc
            path = os.path.join(fig_dir, fig_name)
            vespagram_fig.savefig(path, dpi=500, bbox_inches = 'tight')
        except Exception as e:
            print(e, ' Error on line {}'.format(sys.exc_info()[-1].tb_lineno))
            pass    
          
        try:
            fig_name = 'Threshold_%s.png' %array_class.ev_array_gcarc
            path = os.path.join(fig_dir, fig_name)
            threshold.savefig(path, dpi=500, bbox_inches = 'tight')
        except Exception as e:
            print(e, ' Error on line {}'.format(sys.exc_info()[-1].tb_lineno))
            pass 
        
        try:
            fig_name = 'Picking_%s.png' %array_class.ev_array_gcarc
            path = os.path.join(fig_dir, fig_name)
            picking_figure.savefig(path, dpi=500, bbox_inches = 'tight')
        except Exception as e:
            print(e, ' Error on line {}'.format(sys.exc_info()[-1].tb_lineno))
            pass
        
        try:
            fig_name = 'Vespa_QC_%s.png' %array_class.ev_array_gcarc
            path = os.path.join(fig_dir, fig_name)
            QC_fig.savefig(path, dpi=500, bbox_inches='tight')
        except Exception as e:
            print(e, ' Error on line {}'.format(sys.exc_info()[-1].tb_lineno))
            pass
        
        try:
            fig_name = 'Final_Vespa_%s.png' %array_class.ev_array_gcarc
            path = os.path.join(fig_dir, fig_name)
            final_vespa.savefig(path, dpi=500, bbox_inches = 'tight')
        except Exception as e:
            print(e, ' Error on line {}'.format(sys.exc_info()[-1].tb_lineno))
            pass
        
        try:
            fig_name = 'Beampack_Beam_Combo_%s.svg' %array_class.ev_array_gcarc
            path = os.path.join(fig_dir, fig_name)
            beampack_beams.savefig(path, dpi=500, bbox_inches = 'tight')
        except Exception as e:
            print(e, ' Error on line {}'.format(sys.exc_info()[-1].tb_lineno))
            pass
        
        try:
            fig_name = 'Vespa_Combo_%s.png' %array_class.ev_array_gcarc
            path = os.path.join(fig_dir, fig_name)
            vespas_combined.savefig(path, dpi=500, bbox_inches = 'tight')
        except Exception as e:
            print(e, ' Error on line {}'.format(sys.exc_info()[-1].tb_lineno))
            pass
        
        try:
            fig_name = 'Vespa_QC_Combo_%s.png' %array_class.ev_array_gcarc
            path = os.path.join(fig_dir, fig_name)
            QC_vespas_combined.savefig(path, dpi=500, bbox_inches = 'tight')
        except Exception as e:
            print(e, ' Error on line {}'.format(sys.exc_info()[-1].tb_lineno))
            pass
            
    # Array making figure
    #Load ad-hoc array npys
    stream = np.load(data_file + '/Arrays/array_stream_Z.npy', allow_pickle=True)
    centroids = np.load(data_file + '/Arrays/array_centroids.npy', allow_pickle=True)
    stlats = np.load(data_file + '/Arrays/array_stlats.npy',  allow_pickle=True)
    stlons = np.load(data_file + '/Arrays/array_stlons.npy', allow_pickle=True)
    stlo = np.load(data_file + '/Arrays/stlo.npy', allow_pickle=True)
    stla = np.load(data_file + '/Arrays/stla.npy', allow_pickle=True)

    if component == 'Z':
        figures = Figures()
        global_fig, US_fig = figures.Plot_Arrays(array_classes[0].event, stlons, stlats, centroids, stlo, stla)

        '''# Find successful P arrays!

        #--------- Set Event ---------------

        # Select event from catalogue and define attributes
        event = Earthquake_event(catalogue[input_no-1])
        event.define_event()

        print("Event names is:",event.evname)

        # P arrays
        P_stla = []
        P_stlo = []
        name = 'array_Z.npy'
        path = os.path.join(file_path, name)
        array_classes = np.load(path, allow_pickle=True)
        print(len(array_classes))
        for i in range (len(array_classes)):
            P_stla.append(array_classes[i].stations.stla)
            P_stlo.append(array_classes[i].stations.stlo) #For plotting successful arrays

        # Load in processed Z,N,E data
        #print(data_file)
        stream = obspy.read(data_file + "/Data/" + "*.MSEED")
        stream_Z = stream.select(component='Z')

        streamZ = stream_Z

        # Find attributes
        stations = Stations(streamZ, event, data_file)
        stations.get_station_attributes()

        #centroids = np.load('centroids.npy', allow_pickle=True) # one off to match paper arrays

        figures = Figures()
        global_fig, US_fig = figures.Plot_Arrays(array_classes[0].event, stlons, stlats, centroids, S_stlo, S_stla) #NEEDS US FIXING...
        #global_fig, US_fig = figures.Plot_Arrays(array_classes[0].event, P_stlo, P_stla, centroids, stations.stlo, stations.stla) #NEEDS US FIXING...
        #global_fig, US_fig = figures.Plot_Arrays_success(array_classes[0].event, P_stlo, P_stla, centroids, stations.stlo, stations.stla) #NEEDS US FIXING...

        fig_name = 'Global_arrays_success_P.png'
        path = os.path.join(fig_dir, fig_name)
        global_fig.savefig(path, dpi=500, bbox_inches = 'tight')

        fig_name = 'US_arrays_success_P.png'
        path = os.path.join(fig_dir, fig_name)
        US_fig.savefig(path, dpi=500, bbox_inches = 'tight')'''

    if component == 'T':

        #--------- Set Event ---------------
        event = Earthquake_event(catalogue[int(input_no)-1])
        event.define_event()
    
        # S arrays
        S_stla = []
        S_stlo = []
        name = 'array_T.npy'
        path = os.path.join(file_path, name)
        array_classes = np.load(path, allow_pickle=True)
        print(len(array_classes))
        for i in range (len(array_classes)):
            S_stla.append(array_classes[i].stations.stla)
            S_stlo.append(array_classes[i].stations.stlo) #For plotting successful arrays
            
        '''binned_N_stream = np.load(data_file + 'array_stream_T.npy', allow_pickle=True) # for plotting all S related arrays, pre-processing
        print(len(binned_N_stream))
        
        for i in range (len(binned_N_stream)):
            if binned_N_stream[i] == []:
                continue
            ar = Array(binned_N_stream[i], event, 10, 1)
            ar.format_stream()
            ar.populate_station_metadata(data_file)
        
            S_stla.append(ar.stations.stla)
            S_stlo.append(ar.stations.stlo)'''
            
        # Load in processed Z,N,E data
        #print(data_file)
        stream = obspy.read(data_file + "/Data/" + "*.MSEED")
        stream_N = stream.select(component='N')
        stream_E = stream.select(component='E')
        stream_Z = stream.select(component='Z')
        
        streamZNE = stream_N + stream_E + stream_Z
            
        # Create stream with only stations which have Z,N,E components
        ststring = [0] * len(streamZNE)
        for i in range(0, len(streamZNE)):
            ststring[i] = streamZNE[i].stats.network + '.' + streamZNE[i].stats.station
            #print('ID', ststring[i])
        
        stations = np.unique(ststring)
        #print(len(stations)) 

        final_stream = Stream()
        for j in range (len(stations)):
            #print(stations[j])
            st = stream_Z.select(network = stations[j][:2], station = stations[j][3:])
            #print(st)
            #if len(st) == 3:
            final_stream.extend(st)
        
        streamZNE = final_stream
        
        # Find attributes
        stations = Stations(streamZNE, event, data_file)
        stations.get_station_attributes()

        figures = Figures()
        global_fig, US_fig = figures.Plot_Arrays(array_classes[0].event, S_stlo, S_stla, centroids, stations.stlo, stations.stla)
        #global_fig, US_fig = figures.Plot_Arrays_success(array_classes[0].event, S_stlo, S_stla, centroids, stations.stlo, stations.stla)


    fig_name = 'Global_arrays.png'
    path = os.path.join(fig_dir, fig_name)
    global_fig.savefig(path, dpi=500, bbox_inches = 'tight')

    fig_name = 'US_arrays.png'
    path = os.path.join(fig_dir, fig_name)
    US_fig.savefig(path, dpi=500, bbox_inches = 'tight')

    print('Figures script complete.')
    return



