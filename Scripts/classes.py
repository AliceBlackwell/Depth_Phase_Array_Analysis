#!/usr/bin/env python3

'''
MODULE: for relocating intermediate-earthquakes automatically using array processing
Written by Alice Blackwell.
Date: 24th September 2023
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
import subprocess
import itertools
import statistics
import matplotlib

from sklearn.neighbors import BallTree
from sklearn.cluster import DBSCAN

from obspy.core.stream import Stream
from obspy.geodetics import locations2degrees, degrees2kilometers
from obspy.signal.filter import envelope
from obspy.core import UTCDateTime
from obspy.imaging.cm import obspy_sequential

from copy import deepcopy
import requests

from scipy.signal import hilbert
from scipy.fftpack import fft,ifft
from scipy.signal import find_peaks, peak_prominences, peak_widths
from scipy.stats import median_abs_deviation 
from scipy.stats import percentileofscore

from bs4 import BeautifulSoup

import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

class Earthquake_event:
    def __init__(self, event):
        ''' initialise catalogue entry as Event class'''
        self.event = event
        
    def add_zeroes(self, mn):
        ''' creates 2 digit date inputs, e.g. 3/10/1998 --> 03/10/1998 '''
        if len(str(mn)) == 2:
            pass
        else:
            mn = '0'+ str(mn)
        return mn
        
    def define_event(self):
        ''' populate event attributes'''
        
        self.event_id = re.sub("[^0-9]", "", str(self.event.resource_id))
        self.evla = self.event.origins[0].latitude
        self.evlo = self.event.origins[0].longitude    
        self.evm = self.event.magnitudes[0].mag
        self.evdp = self.event.origins[0].depth/1000
        self.yyyy = int(self.event.origins[0].time.year)
        self.mn = int(self.event.origins[0].time.month)
        self.dd = int(self.event.origins[0].time.day)
        self.hh = int(self.event.origins[0].time.hour)
        self.mm = int(self.event.origins[0].time.minute)
        self.ss = int(self.event.origins[0].time.second)
        self.utc_name = str(self.event.origins[0].time)[:19] #string for origin time
        self.origin_time = obspy.UTCDateTime(self.yyyy,self.mn,self.dd,self.hh,self.mm,self.ss) # UTC format origin time

        mn_0 = self.add_zeroes(self.mn)
        dd_0 = self.add_zeroes(self.dd)
        hh_0 = self.add_zeroes(self.hh)
        mm_0 = self.add_zeroes(self.mm)
        ss_0 = self.add_zeroes(self.ss)

        #evname_obspyDMT = str(yyyy)+str(mn)+str(dd)+'_'+str(hh)+str(mm)+str(ss)
        self.evname = str(self.yyyy)+str(mn_0)+str(dd_0)+str(hh_0)+str(mm_0)+str(ss_0)        
        return
        
class Stations:
    def __init__(self, stream, event, data_file):
        self.stream = stream   # Obspy Stream object
        self.event = event     # Earthquake_event class object
        self.data = data_file  # file pathway to station .xml files
        
    def get_station_attributes(self):
        '''Populate station attributes using Obspy stream'''
        
        ntraces = len(self.stream)
        # Extract metadata from the stream
        stname = [0] * ntraces # Station name
        ststring = [0] * ntraces # Station SEED ID
        stla = [0] * ntraces # Station latitude
        stlo = [0] * ntraces # Station longitude
        stel = [0] * ntraces # Station elevation
        stnet = [0] * ntraces # Station network 
        nsamples = [0] * ntraces #no. sample points per trace
        gcarc = [0] * ntraces # Event to station distance
        az = [0] * ntraces # Azimuth
        radaz = [0] * ntraces # Azimuth in radians
        baz = [0] * ntraces # Backazimuth
        radbaz = [0] * ntraces # Backazimuth in radians
        pathname_xml = self.data + '/Stations'
        
        for i in range(0,ntraces):
            stname[i] = self.stream[i].stats.station
            ststring[i] = self.stream[i].get_id()   #array containing SEED identifiers for each trace with network, station, location & channel code
        
        if ststring[0][-1] != 'T':     # if not a transverse stream...  
            for i in range(0,ntraces):
                rname = pathname_xml+ '/' + self.stream[i].stats.network + '.' + self.stream[i].stats.station+ '.' + self.stream[i].stats.channel + '.xml'
                #print(i, 'out of', ntraces-1,':', rname)
                inv=obspy.read_inventory(rname) #creates a one trace inventory 
                stnet[i]=self.stream[i].stats.network
                nsamples[i] = (self.stream[i].stats.npts)
                statcoords = inv.get_coordinates(ststring[i],self.event.origin_time)
                stla[i] = (statcoords[u'latitude'])
                stlo[i] = (statcoords[u'longitude'])
                stel[i] = (statcoords[u'elevation'])
                
                gcarc[i] = obspy.geodetics.locations2degrees(self.event.evla, self.event.evlo, stla[i], stlo[i])
                rayline=obspy.geodetics.gps2dist_azimuth(self.event.evla, self.event.evlo, stla[i], stlo[i])
                az[i] = rayline[1] # Event to station, in degrees
                radaz[i] = math.radians(az[i])
                baz[i]=rayline[2]
                radbaz[i] = math.radians(baz[i])
                
        else:  # use Z component metadata for T component stream
            for i in range(0,ntraces):
                rname = pathname_xml+ '/' + self.stream[i].stats.network + '.' + self.stream[i].stats.station+ '.' + '*HZ' + '.xml'
                #print(i, 'out of', ntraces-1,':', rname)
                inv=obspy.read_inventory(rname) #creates a one trace inventory 
                stnet[i]=self.stream[i].stats.network
                nsamples[i] = (self.stream[i].stats.npts)
                statcoords = inv.get_coordinates(ststring[i][:-1]+'Z',self.event.origin_time)
                stla[i] = (statcoords[u'latitude'])
                stlo[i] = (statcoords[u'longitude'])
                stel[i] = (statcoords[u'elevation'])
                
                gcarc[i] = obspy.geodetics.locations2degrees(self.event.evla, self.event.evlo, stla[i], stlo[i])
                rayline=obspy.geodetics.gps2dist_azimuth(self.event.evla, self.event.evlo, stla[i], stlo[i])
                az[i] = rayline[1] # Event to station, in degrees
                radaz[i] = math.radians(az[i])
                baz[i]=rayline[2]
                radbaz[i] = math.radians(baz[i])
                
        self.stname = stname
        self.ststring = ststring
        self.stla = stla
        self.stlo = stlo
        self.stel = stel
        self.stnet = stnet
        self.nsamples = nsamples
        self.ev_st_gcarc = gcarc
        self.st_baz = baz
        self.pathname_xml = pathname_xml
        return
        
    @staticmethod
    def get_tauP_arrivals(evla, evlo, stla, stlo, evdp, phase_list, vel_model):
        ''' Method to calculate arrival times using TauP with event and station coordinates'''
        ''' Can be called without intialising Stations class, Stations.get_tauP_arrivals() '''
        ''' example phase_list=["P","pP","sP","S","sS"]'''
        gcarc = obspy.geodetics.locations2degrees(evla, evlo, stla, stlo)
        
        arrivals = vel_model.get_travel_times(source_depth_in_km=evdp, distance_in_degree=gcarc ,phase_list=phase_list)
        if (len(arrivals) < 0.5):
            print("No predicted arrivals for station at", stla, stlo)
        else:
            pass
        
        arr_times = []
        slowness = []
        for i in range (len(arrivals)):
            arr_times.append(arrivals[i].time)
            slowness.append((arrivals[i].ray_param_sec_degree)/111) # in km
        return arr_times, slowness
   
    def make_arrays(self, min_array_diameter, min_stations, print_sub_arrays, stream=None, evla=None, evlo=None, stla=None, stlo=None):
        '''Assemble stations into ad-hoc arrays using DBSCAN and Ball-Tree machine learning algorithms from Sci-kit Learn. Adapted from James Ward. Can feed in own inputs without class dependance.'''
        
        if stream == None:  # unlikely due to the intialisation of the Stations class needing a stream
            stream = self.stream
        if evla == None:   # unlikely due to the intialisation of the Stations class needing an event
            evla = self.event.evla
        if evlo == None:   # unlikely due to the intialisation of the Stations class needing an event
            evlo = self.event.evlo
        if stla == None:
            stla = self.stla
        if stlo == None:
            stlo = self.stlo
        
        # Remind of stream, stations, and lat/lon
        lat_lons = np.array(list(zip(stla,stlo)))
            
        # Convert lat/lon to radians (consider zip function)
        lat_lons_rad = np.array(list(zip(np.deg2rad(stla),np.deg2rad(stlo))))
        
        # Decide DBSCAN Parameters
        min_radius_deg = min_array_diameter/2
        min_radius = np.deg2rad(min_radius_deg)
        print('Minimum radius in radians: ',min_radius)
        min_samples = min_stations
        
        # DBSCAN============================================================================
        dbscan = DBSCAN(eps = min_radius, min_samples = min_samples, metric = 'haversine').fit(lat_lons_rad)
        core_samples_mask = np.zeros_like(dbscan.labels_, dtype=bool)
        core_samples_mask[dbscan.core_sample_indices_] = True
        labels = dbscan.labels_
        
        print('Number of stations: ',len(labels))
        core_samples = dbscan.core_sample_indices_
        print('Estimated number of cores: ', len(core_samples))
        
        if len(core_samples) == 0:
            print('No sub-arrays can be created, station distribution not appropriate')
            return 0, 0, 0

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        print("Estimated number of clusters: %d" % n_clusters_)
        print("Estimated number of noise points: %d" % n_noise_)
        
        # BALL-TREE========================================================================
        stlo = np.asarray(stlo)
        stla = np.asarray(stla)

        lons_core = stlo[core_samples]
        lats_core = stla[core_samples]

        ## Store the usable stations in a 2D array
        lons_use = stlo[np.where(labels >= 0)[0]]
        lats_use = stla[np.where(labels >= 0)[0]]
        lats_lons_use = np.array(list(zip(np.deg2rad(lats_use), np.deg2rad(lons_use))))

        # store lats and lons of core points
        lats_lons_core = np.array(list(zip(np.deg2rad(lats_core), np.deg2rad(lons_core))))

        # make a tree of these lats and lons
        tree = BallTree(lats_lons_core, leaf_size=int(np.round(lats_lons_core.shape[0]/2,0)), metric="haversine")

        # make a copy of the lats and lons of the core points as reference
        core_points_as_centroids = np.copy(lats_lons_core)
        core_samples_cp = np.copy(core_samples)
        
        # create list for the final centroids
        final_centroids = []
        centroid_index = []
        while core_points_as_centroids.size != 0:
            # first get all the core points within min_radius of the first core point in the
            sub_array, distances = tree.query_radius(
                X=np.array([core_points_as_centroids[0]]), r=min_radius, return_distance=True
            )
            # add the first point to the centroid list
            final_centroids.append(core_points_as_centroids[0])
            centroid_index.append(core_samples_cp[0])
            # for every value not within the spacing distance of the centroid,
            # apply a mask and keep them
            # i.e. remove all core points within the spacing distance of the
            # current centroid.

            for s in sub_array[0]:
                value = lats_lons_core[s]
                core_samples_cp =  core_samples_cp[np.logical_not(np.logical_and(core_points_as_centroids[:,0]==value[0],
                                                                     core_points_as_centroids[:,1]==value[1]))]
                core_points_as_centroids =  core_points_as_centroids[np.logical_not(np.logical_and(core_points_as_centroids[:,0]==value[0],
                                                                     core_points_as_centroids[:,1]==value[1]))]

        # SAVE OUT CENTROIDS AND ASSOCIATED STATIONS -----------------------------------------
        
        # Create tree to query later
        use_tree = BallTree(lats_lons_use, leaf_size=int(np.round(lats_lons_core.shape[0]/2,0)), metric='haversine')

        binned_stations_lat = [0] * len(final_centroids) # station coordinates found for each array will be stored here
        binned_stations_lon = [0] * len(final_centroids)
        centroids = [0] * len(final_centroids) # array centre station stored here
        sub_array = [0] * len(final_centroids) # results of tree query stored here

        # Query ball-tree and populate arrays with stations/centroid
        for i in range (0,len(final_centroids)):
            lat_centre = np.around(final_centroids[i][0], 2)
            lon_centre = np.around(final_centroids[i][1], 2)
            sub_array[i] = use_tree.query_radius(X=[final_centroids[i]], r=min_radius)[0] # searches for stations in tree (made earlier) within min radius of centroid
            binned_stations_lon[i] = np.degrees(lats_lons_use[:,1][sub_array[i]]) # saving binned stations
            binned_stations_lat[i] = np.degrees(lats_lons_use[:,0][sub_array[i]])
            centroids[i] = (np.degrees(lon_centre), np.degrees(lat_centre))    # saving centroid stations

        # Print sub-array results (flag) --------------------------------------------------------------
        if print_sub_arrays == 1:
            binned_st_coords = [0] * len(binned_stations_lon)
            for i in range (len(binned_stations_lon)):
                binned_st_coords[i] = list(zip(binned_stations_lon[i], binned_stations_lat[i]))

            for i in range (len(centroids)):
                print('Centroid Station:',centroids[i])
                print('Binned Stations:', len(binned_stations_lat[i]))
                print()
                print(binned_st_coords[i])
                print()
                plt.scatter(binned_stations_lon[i], binned_stations_lat[i])
                plt.scatter(centroids[i][0], centroids[i][1])
                print()

        # Split Stream into array streams ================================================
  
        sub_array_stream = []
        for i in range (len(centroids)):
            sub_array_stream.append([])

        # ==== Binned Stations ======
        mask = np.where(labels >= 0)[0]
        stream_use = [0] * len(mask)
     
        for i in range (len(mask)):
            stream_use[i] = stream[mask[i]]  #stations remaining post DBSCAN masking
        
        # ==== Centroid Stations =====
        stream_centroids = [0] * len(centroid_index)
        
        for i in range (len(centroid_index)):
            stream_centroids[i] = stream[centroid_index[i]]
             
        # ==== Bin Stream ====
        for i in range (len(centroids)):
            for j in range (len(sub_array[i])):
                sub_array_stream[i].append(stream_use[sub_array[i][j]]) # apply ball-tree query to stream to bin traces into arrays
                
        print('No. of ad-hoc arrays:', len(sub_array_stream))
        print('No. of stations used:', len(lats_lons_use))
        return sub_array_stream, centroids, binned_stations_lat, binned_stations_lon
        
    def recreate_arrays_for_other_components(self, array_stream, component_stream):
        ''' Use a stream which has already been split into arrays (i.e. [[list of traces per array][...][...]]) to make the same arrays for another component'''
        # Per array, go through traces and append to new component array list
        
        component_array_stream = []
        
        for i in range (len(array_stream)):
            array = array_stream[i]
            component_array_stream.append([])
            
            for j in range (len(array)):
                trace = array[j]
                ststring = trace.stats.network + '.' + trace.stats.station
                st = component_stream.select(network = ststring[:2], station = ststring[3:])
                if len(st) == 1:
                    component_array_stream[i].append(st[0])
                else:
                    component_array_stream[i].extend(st)
       
        return component_array_stream
        

class Array:
    def __init__(self, stream, event, resample, array_no):
        self.stream = stream
        self.event = event
        self.resample = resample
        self.array_no = array_no
    
    def get_class_variables(self):
        '''Return variables saved to class instance.'''
        members = [attr for attr in vars(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]
        print(members)
        return members

    def format_stream(self):
        ''' format list of traces into an Obspy Stream'''
        stream_name = Stream()
        for i in range(len(self.stream)):
            stream_name.append(self.stream[i])

        #stream_name = Stream(traces=[self.stream]) # should work, but doesn't
        stream_name.resample(self.resample)
        stream_name.normalize()
        self.stream = stream_name
        
    def populate_station_metadata(self, data_dir):
        ''' set up Station class for array stations (class is accessible through self.stations) '''
        array_stations = Stations(self.stream, self.event, data_dir)
        array_stations.get_station_attributes()
        #print(array_stations.stla, array_stations.stlo)
        self.stations = array_stations
        return
        
    def get_array_attributes(self):  
        ''' Determine array centre, backazimuth from the centre and average station elevation''' 
        average_array_stla = np.sum(self.stations.stla)/len(self.stations.stla)
        average_array_stlo = np.sum(self.stations.stlo)/len(self.stations.stlo) 
        average_coords = ('%.3f' % average_array_stlo, '%.3f' % average_array_stla)
        rayline_ref=obspy.geodetics.gps2dist_azimuth(self.event.evla,self.event.evlo, average_array_stla, average_array_stlo)
        baz_ref=rayline_ref[2]
        average_elevation = np.sum(self.stations.stel)/len(self.stations.stel)
        
        self.ev_array_gcarc = obspy.geodetics.locations2degrees(self.event.evla, self.event.evlo, average_array_stla, average_array_stlo)
        
        self.array_latitude = average_array_stla
        self.array_longitude = average_array_stlo
        self.array_baz = baz_ref
        self.array_elevation = average_elevation
        return
        
    def apply_Z_array_attributes(self, array_Z):
        # attributes to add to class from another (Z) array        
        self.event = array_Z.event 
        self.resample =  array_Z.resample 
        #self.stations = array_Z.stations
        self.ev_array_gcarc = array_Z.ev_array_gcarc
        self.array_latitude = array_Z.array_latitude
        self.array_longitude = array_Z.array_longitude
        self.array_baz = array_Z.array_baz
        self.array_elevation = array_Z.array_elevation
        #self.taup_P_time = array_Z.taup_P_time
        #self.taup_slowness = array_Z.taup_slowness
        #self.taup_pP_time = array_Z.taup_pP_time
        #self.taup_sP_time = array_Z.taup_sP_time
        #self.taup_S_time = array_Z.taup_S_time
        #self.taup_sS_time = array_Z.taup_sS_time
        self.beampack_backazimuth = array_Z.beampack_backazimuth
        self.beampack_slowness = array_Z.beampack_slowness
        self.slowness_index = array_Z.slowness_index
        self.backazimuth_range = array_Z.backazimuth_range
        self.slowness_range = array_Z.slowness_range
        return
    
    def output_array_stations(self, ev_dir, array_no, handle, beampack=True):
        ''' Saves out ad-hoc array metadata.'''
        if beampack==False:
            for i in range (len(self.stream)):
                if os.path.isfile(ev_dir + '/array_stations_' + handle + '.txt') == False:
                    f = open(ev_dir + '/array_stations_' + handle + '.txt', 'w')
                    f.write(str(array_no) + '\t' + str(self.array_latitude) + '\t'+ str(self.array_longitude) + '\t' + str(self.stream[i].get_id()) + '\t' + str(self.array_elevation) + '\t'+ str(self.ev_array_gcarc)+'\n')
                    f.close()
                    
                else:
                    f = open(ev_dir + '/array_stations_' + handle + '.txt', 'a+')
                    f.write(str(array_no) + '\t' + str(self.array_latitude) + '\t'+ str(self.array_longitude) + '\t' + str(self.stream[i].get_id()) + '\t' + str(self.array_elevation) + '\t'+ str(self.ev_array_gcarc)+'\n')
                    f.close()
        
        if beampack==True:      
            for i in range (len(self.stream)):
                if os.path.isfile(ev_dir + '/array_stations_' + handle + '.txt') == False:
                    f = open(ev_dir + '/array_stations_' + handle + '.txt', 'w')
                    f.write(str(array_no) + '\t' + str(self.array_latitude) + '\t'+ str(self.array_longitude) + '\t' + str(self.stream[i].get_id()) + '\t' + str(self.array_elevation) + '\t'+ str(self.ev_array_gcarc)+ '\t' + str(self.beampack_backazimuth) + '\t'+ str(self.beampack_slowness) + '\n')
                    f.close()
                    
                else:
                    f = open(ev_dir + '/array_stations_' + handle + '.txt', 'a+')
                    f.write(str(array_no) + '\t' + str(self.array_latitude) + '\t'+ str(self.array_longitude) + '\t' + str(self.stream[i].get_id()) + '\t' + str(self.array_elevation) + '\t'+ str(self.ev_array_gcarc) + '\t' + str(self.beampack_backazimuth) + '\t'+ str(self.beampack_slowness)+ '\n')
                    f.close()
        return
        
    def save_P_sS_tauP_arrivals(self, vel_model):
        ''' Calculate model arrival times using TauP for P, pP, sP, S, sS'''
        gcarc = obspy.geodetics.locations2degrees(self.event.evla, self.event.evlo, self.array_latitude, self.array_longitude)
        
        arrivals = vel_model.get_travel_times(source_depth_in_km=self.event.evdp, distance_in_degree=gcarc ,phase_list=["P","pP","sP","S","sS"])
        if (len(arrivals) < 0.5):
            print("No predicted arrivals for station at", self.array_latitude, self.array_longitude)
        else:
            pass
        
        self.taup_P_time = arrivals[0].time
        slowness = arrivals[0].ray_param_sec_degree
        self.taup_slowness = slowness/111 # P slowness in s/km
        self.taup_S_slowness = arrivals[3].ray_param_sec_degree/111 # S slowness in s/km
        
        self.taup_pP_time = arrivals[1].time
        self.taup_sP_time = arrivals[2].time
        self.taup_S_time = arrivals[3].time
        self.taup_sS_time = arrivals[4].time  
        return 
        
    def rotate_stream_to_transverse(self, data_file, array_Z):
        ''' Rotate ZNE data into RT, and save out tranverse stream.'''
        # P_backazimuth from Z component array
        try:
            P_backazimuth = array_Z.beampack_backazimuth
        except:
            P_backazimuth = np.nan
        
        print('P baz', P_backazimuth)
        
        # Resample, interpolate sample points to same points in time and trim traces       
        self.stream.resample(self.resample)
        starttime=(self.event.origin_time + (5*60) + 1)
        endtime=(self.event.origin_time + (5*60) + 1200 - 1)

        stream = Stream()
        for i in range (len(self.stream)):
            try:
                tr = self.stream[i]
                tr.interpolate(sampling_rate=10, method="lanczos", starttime=starttime, a=12)
                stream.append(tr)
            except:
                print('FAILED', i, tr)

        ZNE_array_stream = stream
        ZNE_array_stream.trim(starttime, endtime)
        
        # replace P_backazimuth with calculated values if np.nan (array failed during Z component processing), rotate components
        if np.isnan(P_backazimuth):
            try:
                # Calculate backazimuth
                print('USING CALCULATED BAZ')
                rayline_ref=obspy.geodetics.gps2dist_azimuth(self.event.evla, self.event.evlo, self.array_latitude, self.array_longitude)
                P_backazimuth=rayline_ref[2]
                print('Calculated backazimuth: ', P_backazimuth)
                ZNE_array_stream.rotate(method='NE->RT',back_azimuth=P_backazimuth)
            except Exception as e:
                print('Failed: ',e)

        else:
            try:
                print('Using P beampack found backazimuth')
                ZNE_array_stream.rotate(method='NE->RT',back_azimuth=P_backazimuth)
            except Exception as e:
                print('Failed:', e)
                
        # make stream of just transverse components
        print(ZNE_array_stream)
        stream_T = ZNE_array_stream.select(component='T')
        print(stream_T)
        for tr in stream_T:
                tr.normalize() # Normalise waveforms
                tr.write(data_file + "/Data/"+tr.stats.network+"."+tr.stats.station + "." + tr.stats.channel + '.MSEED', format = 'MSEED')
        
        self.stream = stream_T
        return 
    
    def beampack_P(self, vel_model):
        '''Beampack the direct phase, and extract best-fitting backazimuth and slowness for the array'''     
        # Define test backazimuth range
        back_max = self.array_baz+15
        if back_max > 360:
            back_max = back_max - 360
  
        back_min = self.array_baz-15
        if back_min < 0:
            back_min = 360 + back_min
        
        if back_min > back_max:
            back_min_tmp = back_min
            back_min = back_max
            back_max = back_min_tmp 
        backazimuth_range = np.linspace(back_min, back_max, 30)

        # Define test slowness range
        print('taup slw', self.taup_slowness)
        dt_dx = round(self.taup_slowness,3)       
        slowness_range = np.linspace(dt_dx-0.04, dt_dx+0.04, 81)
        min_slowness = min(slowness_range);   max_slowness = max(slowness_range)
        nbins = int((max_slowness - min_slowness)/0.001)
        exp_slw_index = int((self.taup_slowness - min_slowness)*(nbins)/(max_slowness - min_slowness))
        
        #Setting up the trim parameters (seconds)
        trima = 7 
        trimb = 10
        
        # Define trace trims
        starttime = self.event.origin_time + self.taup_P_time - trima
        endtime = self.event.origin_time + self.taup_P_time + trimb
        
        # Finding TauP P arrival time for each station in array, 
        # and calculating the time difference to the array centre
        '''timeshift = [0] * len(self.stream)
        for i in range (len(self.stream)):
            arrivals = vel_model.get_travel_times(source_depth_in_km=self.event.evdp, distance_in_degree=self.stations.ev_st_gcarc[i], phase_list=["P"])
            arr=arrivals[0]
            station_P_time = arr.time
            timeshift[i] = self.taup_P_time - station_P_time # NOT USED'''
        
        
        # --- Beamforming ---
        # Set up empty 3d array for the timeshift needed to align station data to array centre,
        # [slowness][backazimuth][stations]
        timeshift_m = np.zeros((len(slowness_range), len(backazimuth_range), len(self.stations.stla)))
        
        # Convert backazimuth into radians
        baz_range_rad = [0] * len(backazimuth_range)
        for i in range (len(backazimuth_range)):
            baz_range_rad[i] = math.radians(backazimuth_range[i])
        
        # Set up empy 2d array for distance to array centre along backazimuth, per station
        baz_dist = np.zeros((len(baz_range_rad), len(self.stations.stla)))
        
        # Populate arrays for baz_dist and timeshift
        for i in range (len(slowness_range)):
            for j in range(len(baz_range_rad)):
                for k in range(len(self.stations.stla)):
                    baz_dist[j][k] = degrees2kilometers((self.array_longitude - self.stations.stlo[k])*math.sin(baz_range_rad[j]) + (self.array_latitude - self.stations.stla[k])*math.cos(baz_range_rad[j]))
                    timeshift_m[i][j][k] = (slowness_range[i]*baz_dist[j][k])*-1
        
        # Set up empty 2d array to store the maximum beam value per slow/baz test pair
        max_env_grd = np.zeros((len(slowness_range), len(backazimuth_range)))  # maximum envelope 2D grid
        
        # beamform for each test baz and slowness, store max beam envelope amplitude
        for i in range (len(slowness_range)):
            for j in range(len(baz_range_rad)):
                beams_tmp = Stream()
                stream_tmp = deepcopy(self.stream)
                for k in range (len(self.stream)):
                    tr = stream_tmp[k]
                    data = np.roll(tr.data,int(np.round(timeshift_m[i][j][k]*self.resample)))
                    tr.data = data
                    tr.trim(starttime,endtime,pad=True,fill_value=0)
                    tr.normalize()
                    
                    if k == 0:
                        sum_data = tr.data
    
                    if k>0 and k<(len(self.stream)-1):
                        sum_data = sum_data + tr.data
    
                    if k == (len(self.stream)-1):
                        sum_data = sum_data + tr.data
                        beams_tmp.data = sum_data/len(self.stream) #normalise
                        
                        from obspy.signal.filter import envelope
                        env = envelope(beams_tmp.data)
                        max_env_grd[i][j] = np.max(env)
            
        # Find highest coherency [array index], which indicates best-fit backazimuth and slowness
        # (find the maximum of the beam envelope maximums)
        max_beampack = np.max(max_env_grd)
        max_beampack_index = np.where(max_env_grd==max_beampack)
        
        # Normalise grid
        max_env_grd = max_env_grd/max_beampack
        
        # Convert Index to Backazimuth and Slowness
        beampack_slow = slowness_range[0] + (max_beampack_index[0]*(slowness_range[1]-slowness_range[0]))
        if len(beampack_slow) > 1:
            beampack_slow = [beampack_slow[0]]
        beampack_baz = max_beampack_index[1]+backazimuth_range[0]
        if len(beampack_baz) > 1:
            beampack_baz = np.asarray([beampack_baz[0]])
        slowness_index = int((beampack_slow[0]-slowness_range[0])/(slowness_range[1]-slowness_range[0]))
        print('Slowness =', beampack_slow, 'Backazimuth =', beampack_baz)
               
        #Save values
        self.beampack_backazimuth = beampack_baz[0].tolist()
        self.beampack_slowness = beampack_slow[0].tolist()
        self.slowness_index = slowness_index
        self.backazimuth_range = backazimuth_range
        self.slowness_range = slowness_range
        self.max_P_envelope_grd = max_env_grd      
        return

    def beampack_S(self, vel_model):
        '''Beampack the direct phase, and extract best-fitting backazimuth and slowness for the array'''     
        # Define test backazimuth range
        back_max = self.array_baz+15
        if back_max > 360:
            back_max = back_max - 360
  
        back_min = self.array_baz-15
        if back_min < 0:
            back_min = 360 + back_min
        
        if back_min > back_max:
            back_min_tmp = back_min
            back_min = back_max
            back_max = back_min_tmp 
        backazimuth_range = np.linspace(back_min, back_max, 30)

        # Define test slowness range
        print('taup slw', self.taup_S_slowness)        
        dt_dx = round(self.taup_S_slowness,3)       
        slowness_range = np.linspace(dt_dx-0.04, dt_dx+0.04, 81)
        min_slowness = min(slowness_range);   max_slowness = max(slowness_range)
        nbins = int((max_slowness - min_slowness)/0.001)
        exp_slw_index = int((self.taup_S_slowness - min_slowness)*(nbins)/(max_slowness - min_slowness))
    
        #Setting up the trim parameters (seconds)
        trima = 7 
        trimb = 10
        
        # Define trace trims
        starttime = self.event.origin_time + self.taup_S_time - trima
        endtime = self.event.origin_time + self.taup_S_time + trimb
        
        # --- Beamforming ---
        # Set up empty 3d array for the timeshift needed to align station data to array centre,
        # [slowness][backazimuth][stations]
        timeshift_m = np.zeros((len(slowness_range), len(backazimuth_range), len(self.stations.stla)))
        
        # Convert backazimuth into radians
        baz_range_rad = [0] * len(backazimuth_range)
        for i in range (len(backazimuth_range)):
            baz_range_rad[i] = math.radians(backazimuth_range[i])
        
        # Set up empy 2d array for distance to array centre along backazimuth, per station
        baz_dist = np.zeros((len(baz_range_rad), len(self.stations.stla)))
        
        # Populate arrays for baz_dist and timeshift
        for i in range (len(slowness_range)):
            for j in range(len(baz_range_rad)):
                for k in range(len(self.stations.stla)):
                    baz_dist[j][k] = degrees2kilometers((self.array_longitude - self.stations.stlo[k])*math.sin(baz_range_rad[j]) + (self.array_latitude - self.stations.stla[k])*math.cos(baz_range_rad[j]))
                    timeshift_m[i][j][k] = (slowness_range[i]*baz_dist[j][k])*-1
        
        # Set up empty 2d array to store the maximum beam value per slow/baz test pair
        max_env_grd = np.zeros((len(slowness_range), len(backazimuth_range)))  # maximum envelope 2D grid
        
        # beamform for each test baz and slowness, store max beam envelope amplitude
        for i in range (len(slowness_range)):
            for j in range(len(baz_range_rad)):
                beams_tmp = Stream()
                stream_tmp = deepcopy(self.stream)
                for k in range (len(self.stream)):
                    tr = stream_tmp[k]
                    data = np.roll(tr.data,int(np.round(timeshift_m[i][j][k]*self.resample)))
                    tr.data = data
                    tr.trim(starttime,endtime,pad=True,fill_value=0)
                    tr.normalize()
                    
                    if k == 0:
                        sum_data = tr.data
    
                    if k>0 and k<(len(self.stream)-1):
                        sum_data = sum_data + tr.data
    
                    if k == (len(self.stream)-1):
                        sum_data = sum_data + tr.data
                        beams_tmp.data = sum_data/len(self.stream) #normalise
                        
                        from obspy.signal.filter import envelope
                        env = envelope(beams_tmp.data)
                        max_env_grd[i][j] = np.max(env)
            
        # Find highest coherency [array index], which indicates best-fit backazimuth and slowness
        # (find the maximum of the beam envelope maximums)
        max_beampack = np.max(max_env_grd)
        max_beampack_index = np.where(max_env_grd==max_beampack)
        
        # Normalise grid
        max_env_grd = max_env_grd/max_beampack
        
        # Convert Index to Backazimuth and Slowness
        beampack_slow = slowness_range[0] + (max_beampack_index[0]*(slowness_range[1]-slowness_range[0]))
        if len(beampack_slow) > 1:
            beampack_slow = [beampack_slow[0]]
        beampack_baz = max_beampack_index[1]+backazimuth_range[0]
        if len(beampack_baz) > 1:
            beampack_baz = np.asarray([beampack_baz[0]])
        slowness_index = int((beampack_slow[0]-slowness_range[0])/(slowness_range[1]-slowness_range[0]))
        print('Slowness =', beampack_slow, 'Backazimuth =', beampack_baz)
               
        #Save values
        self.beampack_backazimuth = beampack_baz[0].tolist()
        self.beampack_slowness = beampack_slow[0].tolist()
        self.slowness_index = slowness_index
        self.backazimuth_range = backazimuth_range
        self.slowness_range = slowness_range
        self.max_P_envelope_grd = max_env_grd      
        return


    def create_vespagram_data(self, trim_start, trim_end):
        ''' Calculate beam per test slowness, using best-fitting baz from beampack_P, and assemble into vespagram''' 

        # Process and trim stream
        self.stream.detrend(type='demean')
        self.stream.normalize()
        stream_vespa = self.stream.copy()
        trima_time = self.event.origin_time + trim_start
        trimb_time = self.event.origin_time + trim_start + trim_end
        
        # Copy stream for trimming
        stream_vespa.trim(trima_time, trimb_time, pad=True, fill_value=0)     
        stream_vespa.normalize()
        
        # Calculate relative time to save out
        rel_time = (np.arange(0,len(stream_vespa[0]),1)/self.resample) + trim_start
        
        # Find distance along backazimuth to the array centre, per station in array
        baz_rad = math.radians(self.beampack_backazimuth)
        baz_dist = [0] * len(self.stations.stla)
        baz_dist_km = [0] * len(self.stations.stla)
        
        for i in range (len(self.stations.stla)):
            baz_dist[i] = (self.array_longitude - self.stations.stlo[i])*math.sin(baz_rad) + (self.array_latitude - self.stations.stla[i])*math.cos(baz_rad)
            baz_dist_km[i] = degrees2kilometers(baz_dist[i])
        
        # Find time difference between station trace and array centre
        dt = np.random.random((len(self.slowness_range), len(baz_dist_km)))     
        for i in range (len(self.slowness_range)):
            for j in range(len(baz_dist_km)):
                dt[i][j] = (self.slowness_range[i]*baz_dist_km[j])*-1
                
                
        # ==== BEAMFORMING PER SLOWNESS ====
        npts = stream_vespa[0].stats.npts # sample points of trace
        phi = np.zeros((npts,),dtype=complex) # empty complex array, length of trace sample points
        phi_app = [] # array to store phi per beam in
        st_beam = Stream()
        
        # Beampack value beams for plotting
        stream_plt = Stream()
        st_plt = Stream()
        st_plt_beam = Stream()
        st_plt_beampws = Stream()
        
        for i in range (0, len(self.slowness_range)):   # slowness range
            st_tmp =  stream_vespa.copy()
            st_bu =  stream_vespa[0].copy() # trace to replace data with, and beamform onto
            
            for j in range (0, len(baz_dist_km)):    # station number from 0          
                #Timeshift traces to geometric centre of bin
                tr = st_tmp[j]
                data = np.roll(tr.data,int(np.round(dt[i][j]*self.resample)))
                tr.data = data
               
                if i == self.slowness_index:
                    st_plt.append(tr)  # time shifted traces at array slowness from beampacking
                    
                # for phase weighting beams later
                phi = phi+np.exp(1j*np.angle(hilbert(data)))           
                
                if j == 0:
                    st_bu.data = tr.data
                elif j > 0 and j < (len(baz_dist_km)-1):
                    st_bu.data = st_bu.data + tr.data  # Beamforming data into one stream trace
                elif j == (len(baz_dist_km)-1):
                    st_bu.data = st_bu.data + tr.data
                    st_bu.data = st_bu.data/len(baz_dist_km)
                    st_beam.append(st_bu)   # stream of beamformed streams
                    
                    phi_app.append(phi/len(baz_dist_km)) # # normalise phi (makes no difference to current set up)
                    
                    # Saving best fit beam (phase weighted ^4)
                    if i == self.slowness_index:
                        st_plt_beam.data = st_bu.data
                        st_plt_beampws.data = st_bu.data * (np.abs(phi/len(baz_dist_km)))**4 # phase weighting ^4
                    
                    # reset phase weight factor
                    phi = 0
           
        #Phase-weighting beam
        aphi = (np.abs(phi_app))**4  # normalisation already on the stack
              
        st_beam_phasew = st_beam.copy()
        for i in range (len(st_beam_phasew)):
            st_beam_phasew[i].data = st_beam_phasew[i].data * aphi[i]
        
        # Normalise Beams
        beam_max = []
        pw_beam_max = []
        for i in range (len(st_beam)):
            beam_max.append(np.max(abs(st_beam[i].data)))
            pw_beam_max.append(np.max(abs(st_beam_phasew[i].data)))
        
        beam_max = np.max(beam_max)
        pw_beam_max = np.max(pw_beam_max)

        for i in range (len(st_beam)):        
            st_beam[i].data = st_beam[i].data/beam_max
            st_beam_phasew[i].data = st_beam_phasew[i].data/pw_beam_max
            st_plt_beam.data = st_plt_beam.data/np.max(abs(st_plt_beam.data))
            st_plt_beampws.data = st_plt_beampws.data/np.max(abs(st_plt_beampws.data))
                
        self.timeshifted_stream = st_plt # at beampacked found slowness
        self.beams = st_beam # stream of beamformed traces, not phase weighted
        self.phase_weighted_beams = st_beam_phasew
        self.relative_time = rel_time
        
        self.optimum_beam = st_plt_beam.data
        self.PW_optimum_beam = st_plt_beampws.data     
        self.PW_optimum_beam_envelope = obspy.signal.filter.envelope(st_plt_beampws.data)         
        return

    def do_x_corr_check(self, phase='P'):
        '''Check traces in array are correlating well to the array beam, if not an array is created to remove the traces outside the method'''
        def cross_corr(x, y): # NORMALISED X-CORR
            x = np.array(x)
            y = np.array(y)
            fx = fft(x)
            fy = fft(np.flipud(y))
            cc = np.real(ifft(fx*fy))
            return np.fft.fftshift(cc)/ (sum(x**2)*sum(y**2))**0.5

        def compute_shift(x, y):
            assert len(x) == len(y)
            c = cross_corr(x, y)
            assert len(c) == len(x)
            zero_index = int(len(x) / 2) - 1
            shift = zero_index - np.argmax(c)
            return shift*-1  
        
        # Compare each trace to beam, remove outliers
        if phase == 'P':
            trima_time = self.event.origin_time + self.taup_P_time - 2
            trimb_time = self.event.origin_time + self.taup_P_time + 7
        elif phase == 'S':
            trima_time = self.event.origin_time + self.taup_S_time - 10
            trimb_time = self.event.origin_time + self.taup_S_time + 10
        
        # Trim phase weighted (PW) beam for array
        st_beam = deepcopy(self.phase_weighted_beams)       
        st_beam.trim(trima_time, trimb_time, pad=True, fill_value=0)
        st_beam.normalize()
        #st_beam[self.slowness_index].plot()
 
        # Trim timeshifted traces in array
        stream_trim = deepcopy(self.timeshifted_stream)
        stream_trim.trim(trima_time, trimb_time, pad=True, fill_value=0)
        stream_trim.normalize()
        #stream_trim.plot()
        
        # Checking x correlation works
        #x = st_beam[self.slowness_index].data
        #y = st_beam[self.slowness_index].data
        #cc = cross_corr(x,y)
        
        # Set up empty arrays
        corr = [0] * len(stream_trim)
        max_corr = [0] * len(stream_trim)
        shift = [0] * len(stream_trim)
        
        # x correlate each trace to the beam, store results
        for i in range (len(stream_trim)):
            x = st_beam[self.slowness_index].data # optimum beam
            y = stream_trim[i].data               # trace
            corr[i] = cross_corr(x,y)             # x correlation
            shift[i] = compute_shift(x, y)        # time shift required to match inputs as closely as possible
            print(np.max(corr[i]), shift[i]) 
            '''if phase == 'S':
                fig, ax = plt.subplots(3,1)
                ax[0].plot(st_beam[self.slowness_index].data)
                ax[1].plot(stream_trim[i].data  )
                ax[2].plot(corr[i])
                plt.show()'''
        # Find maximum correlation no. per trace
        for i in range (len(corr)):
            max_corr[i] = np.max(corr[i]) 
        
        lag = len(corr[0])/2
        lag = np.arange(-lag, lag, 1)
                
        repeat_loop = False
        trace_to_keep = []
        
        if phase == 'P':
            shift_limit = 5
        elif phase == 'S':
            shift_limit = 5  #5??
        
        # Create array to remove traces which are not correlating well enough later (trace_to_keep)
        for i in range (len(corr)):            
            if abs(max_corr[i]) >= 0.3 and abs(shift[i])<shift_limit:    # x-correlation condition
                print('x-corr & time-shift per trace:', np.round(abs(max_corr[i]),3), shift[i], '    PASS')
                repeat_loop = True
                trace_to_keep.append(i)
            else:
                print('x-corr & time-shift per trace:', np.round(abs(max_corr[i]),3), shift[i], '    FAIL')
                repeat_loop = True                             
        
        self.x_corr_lag = lag
        self.x_corr = corr
        self.x_corr_shift = shift
        self.x_corr_trimmed_PW_beam = st_beam
        self.x_corr_trimmed_traces = stream_trim
 
        return trace_to_keep, repeat_loop

    def assemble_vespagram_array(self):
        ''' Convert Obspy stream of phase weighted beams per slowness, into a 2d numpy array'''           
        sample_pts = len(self.phase_weighted_beams[0].data)
        vespa_grd = np.random.random((len(self.phase_weighted_beams),sample_pts))
    
        for i in range(0, len(vespa_grd)):
            vespa_grd[i]=self.phase_weighted_beams[i].data
        
        self.vespa_grd = vespa_grd
        return
        
    def array_to_vespagram(self, data_dir, ev_dir, vel_model, trim_start, trim_end, populate_array_metadata=True, beampack=True, phase='P'):
        ''' Uses methods in Array class together to turn an Obspy stream of traces into a vespagram with an optimum beam (using best fitting backazimuth and slowness) '''
        # Format array traces into obspy stream
        self.format_stream()

        # Save out trim_start for pmP workflow
        self.trim_start = trim_start
        
        repeat_loop = False
        trace_to_keep = []
        
        while True:
            if repeat_loop == True and len(trace_to_keep) > 7:
                print('Repeated loop')
                new_stream = Stream()
                for i in range (len(self.stream)):
                    for j in range (len(trace_to_keep)):
                        if i == trace_to_keep[j]:
                            new_stream.append(self.stream[i])
                self.stream = new_stream
            
            elif repeat_loop == True and len(trace_to_keep) <= 7:
                print('Sub-array has less than 8 traces, trace no.: %s' %len(trace_to_keep))
                raise AssertionError('Sub-array has less than 8 traces')	  
                return  
            
            self.populate_station_metadata(data_dir)
            if populate_array_metadata==True:
                # Extract metadata for stations in array
                self.get_array_attributes()

            # Create table of station network, name, longitude and latitude for array
            data= {
                'Station_Network': self.stations.stnet,
                'Station_Name': self.stations.stname,
                'Station_Longitude': self.stations.stlo,
                'Station_Latitude': self.stations.stla}

            df=pd.DataFrame(data)
            print(df)
            
            # Save out ad-hoc array information (pre-QC)
            self.output_array_stations(ev_dir, str(self.array_no), 'original_%s' %phase, beampack=False)

            print('Processing Array %s degrees from the event' %self.ev_array_gcarc)

            # Calculate TauP phase arrival times for array centre
            self.save_P_sS_tauP_arrivals(vel_model)
            print('TauP P time (s):', self.taup_P_time)

            # Beampack 'P' to find array-derived backazimuth and slowness 
            if beampack == True:
                if phase == 'P':
                    self.beampack_P(vel_model)
                elif phase == 'S':
                    self.beampack_S(vel_model) # Beampack S for S,sS
            self.create_vespagram_data(trim_start, trim_end)

            if repeat_loop != True:
                trace_to_keep, repeat_loop = self.do_x_corr_check(phase)
                if len(trace_to_keep) < len(self.timeshifted_stream):
                    #raise StopIteration('Noisy Traces, loop will be repeated')
                    pass
            else:
                break
                
        self.repeated_loop = repeat_loop
        self.assemble_vespagram_array()
        return
        
    def find_picking_threshold(self, figure=False):
        ''' Dynamically determine a picking threshold for identifying significant phases on seismic data, using an amplitude distribution curve. used in self.get_picks.'''
        
        # Trim data, to only consider peaks in the area of interest (P-sP arrivals)
        starttime = self.event.origin_time + (self.taup_P_time * 0.98)
        endtime = self.event.origin_time + (self.taup_sP_time * 1.02)
        
        beam_trimmed = deepcopy(self.phase_weighted_beams[self.slowness_index])
        beam_trimmed.trim(starttime, endtime, pad=True, fill_value=0)      
        
        # Calculate envelope of trimmed data 
        envelope_trimmed = obspy.signal.filter.envelope(beam_trimmed.data)       
       
        # slope functions
        def slope(P1, P2):
            ''' dy/dx to find gradient of slope'''
            return(P2[1] - P1[1]) / (P2[0] - P1[0])
        
        def y_intercept(P1, slope):
            ''' finds y axis intercept of a straight line (y = mx + b)'''
            return P1[1] - slope * P1[0]
        
        def line_intersect(m1, b1, m2, b2):
            '''finds coordinates where two lines intersect'''
            if m1 == m2:
                print ("These lines are parallel!!!")
                return None
            # y = mx + b
            # Set both lines equal to find the intersection point in the x direction
            # m1 * x + b1 = m2 * x + b2
            # m1 * x - m2 * x = b2 - b1
            # x * (m1 - m2) = b2 - b1
            # x = (b2 - b1) / (m1 - m2)
            x = (b2 - b1) / (m1 - m2)
            # Now solve for y -- use either line, because they are equal here
            # y = mx + b
            y = m1 * x + b1
            return x,y
        
        # Approximate pre- and onset slopes on amplitude distribution
        A1 = [0, np.percentile(envelope_trimmed, 0)]
        A2 = [25, np.percentile(envelope_trimmed, 25)]
        B1 = [80, np.percentile(envelope_trimmed, 80)]
        B2 = [100, np.percentile(envelope_trimmed, 100)]
        #print(A1,A2, B1, B2)
        
        slope_A = slope(A1, A2)
        slope_B = slope(B1, B2)
        
        # Find intersect between the slope approximations
        y_int_A = y_intercept(A1, slope_A)
        y_int_B = y_intercept(B1, slope_B)
        x,y = line_intersect(slope_A, y_int_A, slope_B, y_int_B)
        x = int(np.round(x, 0))
        
        # line A (pre-onset)
        xa = []
        ya = []
        # line B (onset)
        xb = []
        yb = []
        
        # Plot percentiles of amplitude data
        percentiles = []
        for i in range (0,101):
            percentiles.append(np.percentile(envelope_trimmed, i))
            xa.append(i)
            xb.append(i)
            ya.append(slope_A*i + y_int_A)
            yb.append(slope_B*i + y_int_B)
        
        # Find percentile for slope intersect
        threshold = np.percentile(percentiles, x)
        
        # Plot figure (?)
        if figure == True:
            x_axis = np.arange(0, 101, 1)
            p = plt.figure()
            plt.plot(x_axis, percentiles, color='k', zorder = 1)
            plt.plot(xa, ya)
            plt.plot(xb, yb)
            plt.scatter(A1[0], A1[1])
            plt.scatter(A2[0], A2[1])
            plt.scatter(B1[0], B1[1])
            plt.scatter(B2[0], B2[1])
            plt.scatter(x, y, marker='x', color='r', label='intersection', zorder = 3)
            plt.axhline(threshold, label=threshold, color = 'k', linestyle = ':', zorder = 2)
            plt.axvline(x, label=x, color = 'k', linestyle = '--', zorder = 2)
            plt.ylim(0,1.1)
            plt.legend()
            plt.close()
           
        print('THRESHOLD', threshold)
        self.picking_threshold = threshold
        return
           
    def get_picks(self, trim_start, trim_end, phases):
        ''' Use a dynamic amplitude threshold to pick phases from optimum beam/vespagram data''' 
        
        self.picks = [0]
        
        if phases == ['P','pP','sP']:
        
            # Trim data, to only consider peaks in the area of interest (P-sP arrivals)
            starttime = self.event.origin_time + (self.taup_P_time * 0.98)
            endtime = self.event.origin_time + (self.taup_sP_time * 1.02)
            
            # Calculate time needed to convert trimmed beams back to relative to origin time
            x_axis_time_addition = int(((self.taup_P_time * 0.98) - trim_start)*self.resample)
            x_axis_end = int(((self.taup_sP_time * 1.02) - trim_start)*self.resample)
            
        if phases == ['S','sS']:
            
            # Trim data, to only consider peaks in the area of interest (P-sP arrivals)
            starttime = self.event.origin_time + (self.taup_S_time * 0.98)
            endtime = self.event.origin_time + (self.taup_sS_time * 1.02)
            
            # Calculate time needed to convert trimmed beams back to relative to origin time
            x_axis_time_addition = int(((self.taup_S_time * 0.98) - trim_start)*self.resample)
            x_axis_end = int(((self.taup_sS_time * 1.02) - trim_start)*self.resample)
            
        # Trim
        beam_trimmed = deepcopy(self.phase_weighted_beams[self.slowness_index])
        beam_trimmed.normalize()
        beam_trimmed.trim(starttime, endtime, pad=True, fill_value=0)
        
        # Calculate envelope
        envelope_trimmed = obspy.signal.filter.envelope(beam_trimmed.data)
        
        # Find peaks
        max_peak = np.max(envelope_trimmed)
        peaks_tmp, properties = find_peaks(envelope_trimmed, prominence=0.15*max_peak)  # can add width as a condition in sample numbers   
        
        # if no picks, end method here
        if len(peaks_tmp) == 0:  
            return
    
        # Convert trimmed x-axis peak locations to the untrimmed data x-axis
        peaks = [0] * len(peaks_tmp)
        for i in range (len(peaks)):
            peaks[i] = peaks_tmp[i] + x_axis_time_addition
        
        # Check if peaks are above dynamic picking threshold              
        # Find threshold
        try:
            self.find_picking_threshold(figure=False)
        except:
            print('No picking threshold determined')
            return
        
        envelope = obspy.signal.filter.envelope(self.phase_weighted_beams[self.slowness_index].data)        
        threshold_x = []        
        for i in range (len(peaks)):
            if envelope[peaks[i]]>self.picking_threshold:
                threshold_x.append(peaks[i])
        picks = np.sort(threshold_x)
    
        self.picks = picks   # x axis locations of picks       
        return

    def QC_vespagram(self, trim_start, trim_end, phases):
        ''' Throws out poor quality vespagram arrays using peaks on the vespagram -- how far away they are from the expected phase arrival slowness (weighted mean) and how spread the cluster centres are (standard deviation).'''
        
        if phases == ['P','pP','sP']:
            # Trim data, to only consider peaks in the area of interest (P-sP arrivals)
            starttime = self.event.origin_time + (self.taup_P_time * 0.98)
            endtime = self.event.origin_time + (self.taup_sP_time * 1.02)
            
            # Calculate time needed to convert trimmed beams back to relative to origin time
            x_axis_time_addition = int(((self.taup_P_time * 0.98) - trim_start)*self.resample)
            x_axis_end = int(((self.taup_sP_time * 1.02) - trim_start)*self.resample)
            
        if phases == ['S','sS']:
             # Trim data, to only consider peaks in the area of interest (P-sP arrivals)
            starttime = self.event.origin_time + (self.taup_S_time * 0.98)
            endtime = self.event.origin_time + (self.taup_sS_time * 1.02)
            
            # Calculate time needed to convert trimmed beams back to relative to origin time
            x_axis_time_addition = int(((self.taup_S_time * 0.98) - trim_start)*self.resample)
            x_axis_end = int(((self.taup_sS_time * 1.02) - trim_start)*self.resample)
            
        # Extract peaks etc. from absolute vespagram
        abs_vespa_grd = np.abs(self.vespa_grd)
        noise_peaks = []
        slow_peaks = []
        x_peaks = []
        for i in range (len(self.vespa_grd)):
            peaks, props = find_peaks(abs_vespa_grd[i])  # find peaks in vespagram
            slow_peaks.extend(np.ones(len(peaks))*i)     # find slownesses of peaks (y axis) in slowness index form (1-81)
            noise_peaks.extend(abs_vespa_grd[i][peaks])  # find coherency/amplitude of peaks (colour on vespagram)
            x_peaks.extend(peaks)                        # find time (x axis)
        
        # Vespagram quality check
        noise_peaks = np.asarray(noise_peaks)  # colour
        slow_peaks = np.asarray(slow_peaks)    # y axis
        x_peaks = np.asarray(x_peaks)          # x axis
        
        #print(x_peaks, noise_peaks, slow_peaks)

        # Keep peaks with >60% of the maximum peak's coherency
        max_coherency = np.max(abs_vespa_grd)
        threshold = float(0.6*max_coherency) 
        coherency_peaks = []
        slowness_peaks = []

        for i in range (len(x_peaks)):
            if x_peaks[i] >= x_axis_time_addition and x_peaks[i]<= x_axis_end:
                if noise_peaks[i] >= threshold:
                    coherency_peaks.append(x_peaks[i]/5*self.resample) # x axis, scale down to help with clustering algorithm (5 seconds division)
                    slowness_peaks.append(slow_peaks[i])  # y axis

        # Find statistics
        if slowness_peaks == []:
            print('Vespagram quality deemed too poor to use')
            raise RuntimeError('Vespagram too poor to pick')
        
        if len(slowness_peaks) > 0:
            
            # Look for clusters of peaks, coherent peaks will cluster around real arrivals
            # A large spread of points away from expected slowness = not coherent vespagram
            X = np.array(list(zip(coherency_peaks,slowness_peaks)))
            db = DBSCAN(eps=2, min_samples=2).fit(X)     # needs 2 samples within 10 seconds / 2 slowness intervals (0.001 km/s) to cluster
            labels = db.labels_

            # Number of clusters in labels, ignoring noise if present.
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise_ = list(labels).count(-1)

            #print("Estimated number of clusters: %d" % n_clusters_)
            #print("Estimated number of noise points: %d" % n_noise_)            
            
            # Find clusters
            unique_labels = set(labels)                          
            cores = []
            for k in unique_labels:                      
                if k==-1: #noise
                  pass
                else:
                    class_member_mask = (labels == k)
                    cores.append(X[class_member_mask])  # non-noise clusters in vespagram data      
            
            if n_clusters_ < 1:
                print('No clusters found on vespagram QC check')      
                mean = np.mean(slowness_peaks) # stats on individual peaks
                std = np.std(slowness_peaks)              
                return std, mean
      
            # Find centres of non-noise clusters, and no. of points in each cluster
            core_centre_x = []
            core_centre_y = []
            no_points = []
            
            for i in range (len(cores)):
                core = cores[i]
                core_x = []
                core_y = []
                for j in range(len(core)):
                    core_x.append(core[j][0])
                    core_y.append(core[j][1])
                no_points.append(len(core_x))
                core_centre_x.append(np.mean(core_x))
                core_centre_y.append(np.mean(core_y))
            
            # Calculate weighted mean of clusters in y axis
            core_centre_mean = np.average(core_centre_y, weights=no_points)
            # Find standard deviation of cluster means in y axis
            core_centre_std = np.std(core_centre_y)
            
            self.vespa_QC_ccx = core_centre_x
            self.vespa_QC_ccy = core_centre_y
            self.vespa_QC_cc_std = core_centre_std
            self.vespa_QC_cc_mean = core_centre_mean
            self.vespa_QC_cores = cores
            self.vespa_QC_npts = no_points
            
            print('Core mean', core_centre_mean)
            print('Acceptable range', self.slowness_index-6, self.slowness_index+6)
            print('Core STD', core_centre_std)
            
            # QC conditions which need to be met to continue with the array data
            if core_centre_mean > self.slowness_index+6 or core_centre_mean < self.slowness_index-6:
                print('Vespagram quality deemed too poor to use')
                raise RuntimeError('Vespagram too poor to pick')
                
            if core_centre_std > 10.5:
                print('Vespagram quality deemed too poor to use')
                raise RuntimeError('Vespagram too poor to pick')       
            return            

    def id_phases(self, trim_start, trim_end, phases):
        """Returns 3 picks, P, pP and sP by looking at observed versus modelled differential times."""   
        
        np.sort(self.picks)

        if phases == ['P','pP','sP']:
            # Set up empty pick arrays  
            peaks_id = [0,0,0]
            pick_amplitudes = [0,0,0]
            # Find noise threshold - trim region in front of P arrival
            endtime = self.event.origin_time + (self.taup_P_time * 0.98)
            starttime = endtime - 40
            
        if phases == ['S','sS']:
            peaks_id = [0,0]
            pick_amplitudes = [0,0]
            # Find noise threshold - trim region in front of P arrival
            endtime = self.event.origin_time + (self.taup_S_time * 0.98)
            starttime = endtime - 40
 
        # SNR Check
        envelope = obspy.signal.filter.envelope(self.phase_weighted_beams[self.slowness_index].data)
        pre_arrival_noise = deepcopy(self.phase_weighted_beams[self.slowness_index])
        pre_arrival_noise.trim(starttime, endtime, pad=True, fill_value=0)
        mean_noise = np.mean(abs(pre_arrival_noise.data))
        
        snr = [0] * len(self.picks)
        snr_filtered_peaks = []
        for i in range (len(self.picks)):
            snr[i] = envelope[self.picks[i]]/mean_noise
            if snr[i] >= 5:
                snr_filtered_peaks.append(self.picks[i])
       
        peaks = snr_filtered_peaks
        print('SNR per peak found: ', snr)
        print('Remaining peaks post snr check: ', peaks)
        
        if len(self.picks) < 2: 
            self.phase_id_picks = peaks_id
            self.phase_id_pick_amplitudes = pick_amplitudes
            return
        
        if phases == ['P','pP','sP']:
            # Convert Relative Arrival Times to Sample Number
            P_time = (self.taup_P_time-trim_start)*self.resample
            pP_time = (self.taup_pP_time-trim_start)*self.resample
            sP_time = (self.taup_sP_time-trim_start)*self.resample
            
            model_pP_P = pP_time - P_time
            model_sP_P = sP_time - P_time
            model_sP_pP = sP_time - pP_time
            
            #print('Peaks', self.picks, envelope[np.asarray(self.picks)], P_time)
            
            pick_dt = []
            for i in range (0, len(self.picks)-1):
                pick_dt.append([])
                
                for j in range (i+1, len(self.picks)):
                    dt = self.picks[j]-self.picks[i]
                    pick_dt[i].append(dt)
            
            dt_margin = 25/100
            dt_margin_plus = 1 + dt_margin
            dt_margin_minus = 1 - dt_margin
            
            #print('Peak dt: ', pick_dt)
            #print('Model dt: ', model_pP_P, model_sP_P, model_sP_pP)
            #print('Model dt +/-25%: ', model_pP_P*dt_margin_minus, model_pP_P*dt_margin_plus)
            
            pP_P_pairs = []
            sP_P_pairs = []
            pP_amp = []
            sP_amp = []
            
            flag = False # replaced with True if constants used for dt margins, e.g. if earthquake is too shallow to produce a decent 25% differential time margin
                         # constants selected based upon max. dt variation expected if earthquake is 40 km wrong
            
            for i in range(len(self.picks)-1):
                for j in range (len(pick_dt[i])):
                    if model_pP_P*dt_margin < 8.5:  #provide a +/-8.5 second search window when 25% of dt is less than 8.5 seconds
                        #print('Using 8.5 second dt window')
                        flag = True
                        if pick_dt[i][j] > model_pP_P-85 and pick_dt[i][j] < model_pP_P+85 and self.picks[i+j+1] > P_time+10:
                            #print(pick_dt[i][j], model_pP_P-85, model_pP_P+85, [i,j])
                            pP_P_pairs.append([i,j])
                            pP_amp.append(envelope[self.picks[i]]+envelope[self.picks[i+j+1]])
                    else: 
                        #print('Using %s of dt window' %dt_margin)
                        #print(self.picks[i+j+1], P_time)
                        if pick_dt[i][j] > model_pP_P*dt_margin_minus and pick_dt[i][j] < model_pP_P*dt_margin_plus and self.picks[i+j+1] > P_time+10:
                            #print(pick_dt[i][j], model_pP_P*dt_margin_minus, model_pP_P*dt_margin_plus, [i,j])
                            pP_P_pairs.append([i,j])
                            pP_amp.append(envelope[self.picks[i]]+envelope[self.picks[i+j+1]])
                    
                    if model_sP_P*dt_margin < 12.5:  #provide a +/-12.5 second search window when 25% of dt is less than 8.5 seconds
                        #print('Using 12.5 second dt window')
                        flag = True
                        if pick_dt[i][j] > model_sP_P-125 and pick_dt[i][j] < model_sP_P+125 and self.picks[i+j+1] > P_time+10:
                            #print(pick_dt[i][j], model_sP_P-125, model_sP_P+125, [i,j])
                            sP_P_pairs.append([i,j])
                            sP_amp.append(envelope[self.picks[i]]+envelope[self.picks[i+j+1]])
                    else: 
                        #print('Using %s of dt window' %dt_margin)
                        if pick_dt[i][j] > model_sP_P*dt_margin_minus and pick_dt[i][j] < model_sP_P*dt_margin_plus and self.picks[i+j+1] > P_time+10:
                            #print(pick_dt[i][j], model_sP_P*dt_margin_minus, model_sP_P*dt_margin_plus, [i,j])
                            sP_P_pairs.append([i,j])
                            sP_amp.append(envelope[self.picks[i]]+envelope[self.picks[i+j+1]])
            
            if len(pP_P_pairs) == 0 and len(sP_P_pairs) == 0:
                print('No eligible P-pP or P-sP pairs')                
                self.phase_id_picks = peaks_id
                self.phase_id_pick_amplitudes = pick_amplitudes
                return
            
            if  len(pP_P_pairs) == 0:
                pP_P_pairs = []
                
            if  len(sP_P_pairs) == 0:
                sP_P_pairs = []

            print('Eligible pP-P pairs: ',pP_P_pairs)
            #print('pP summed amplitudes: ', pP_amp)
            print('Eligible sP-P pairs: ',sP_P_pairs)
            #print('sP summed amplitudes: ', sP_amp)
            
            def find_nearest(array, value):
                """ Returns index in an array closest to another given value.
                
                Parameters:
                array: list of numbers
                value: number
                
                Return:
                array[idx]: closest value in the array to the given value """
                
                array = np.asarray(array)
                idx = (np.abs(array - value)).argmin()
               
                return idx
          
            if len(pP_P_pairs) > 0 and len(sP_P_pairs) > 0:
                
                pP_sP_dt = []
                candidates = []
                # Find pP_P and sP_P pairs with same P pick
                for i in range (len(pP_P_pairs)):
                    #print('pP_P pairs: ', pP_P_pairs[i])
                    for j in range (len(sP_P_pairs)):
                        #print('sP_P pair: ', sP_P_pairs[j])
                        if pP_P_pairs[i][0] == sP_P_pairs[j][0]:               
                            #print('SAME P')
                            pP_sP_dt = self.picks[sP_P_pairs[j][0] + 1 + sP_P_pairs[j][1]]-self.picks[pP_P_pairs[i][0] + 1 + pP_P_pairs[i][1]]
                            #print('sP pick ', sP_P_pairs[j][1], 'pP pick ', pP_P_pairs[i][1], 'sP envelope ', self.picks[sP_P_pairs[j][0] + 1 + sP_P_pairs[j][1]], 'pP envelope ', self.picks[pP_P_pairs[i][0] + 1 + pP_P_pairs[i][1]], 'sP-pP dt ', pP_sP_dt, 'P envelope ', self.picks[pP_P_pairs[i][0]])
                            if flag == False:
                                if pP_sP_dt > model_sP_pP*dt_margin_minus and pP_sP_dt < model_sP_pP*dt_margin_plus:
                                    candidates.append([pP_P_pairs[i], sP_P_pairs[j]])  
                            else:
                                if pP_sP_dt > model_sP_pP-50 and pP_sP_dt < model_sP_pP+50: # 5 second margin 
                                    candidates.append([pP_P_pairs[i], sP_P_pairs[j]])  
                
                #print('candidates: ', candidates)
                
                if len(candidates) == 1:
                    P_pick = candidates[0][0][0]
                    pP_pick = candidates[0][0][1]
                    sP_pick = candidates[0][1][1]
                    
                    P = self.picks[P_pick]
                    pP = self.picks[P_pick+1+pP_pick]
                    sP = self.picks[P_pick+1+sP_pick]
                    
                    peaks_id = [P, pP, sP]
                    print('FINAL PEAKS', peaks_id)
                    pick_amplitudes = envelope[np.asarray(peaks_id)]
                    pick_amplitudes[np.asarray(peaks_id) == 0] = 0
                    print('FINAL PEAK AMPLITUDES', pick_amplitudes)
                    
                    self.phase_id_picks = peaks_id
                    self.phase_id_pick_amplitudes = pick_amplitudes
                    return
                
                combined_amp = []
                if len(candidates) > 1:
                    for i in range (len(candidates)):
                        P_pick = candidates[i][0][0]
                        pP_pick = candidates[i][0][1]
                        sP_pick = candidates[i][1][1]
                        combined_amp.append(envelope[self.picks[P_pick]]+envelope[self.picks[P_pick+1+pP_pick]]+envelope[self.picks[P_pick+1+sP_pick]])
                   
                    idx = np.argmax(np.asarray(combined_amp))
                    P_pick = candidates[idx][0][0]
                    pP_pick = candidates[idx][0][1]
                    sP_pick = candidates[idx][1][1]
                    
                    P = self.picks[P_pick]
                    pP = self.picks[P_pick+1+pP_pick]
                    sP = self.picks[P_pick+1+sP_pick]
                    
                    peaks_id = [P, pP, sP]
                    print('FINAL PEAKS', peaks_id)
                    pick_amplitudes = envelope[np.asarray(peaks_id)]
                    pick_amplitudes[np.asarray(peaks_id) == 0] = 0
                    print('FINAL PEAK AMPLITUDES', pick_amplitudes)
                    
                    self.phase_id_picks = peaks_id
                    self.phase_id_pick_amplitudes = pick_amplitudes
                    return
                 
                else:
                    pass
           
            if len(pP_P_pairs) > 0:
                pP_P_idx = np.argmax(np.asarray(pP_amp))
                #print('pP pair', pP_P_pairs[pP_P_idx])
                
                P_pick_1 = pP_P_pairs[pP_P_idx][0]
                pP_pick = pP_P_pairs[pP_P_idx][1]
                
            if len(sP_P_pairs) > 0:
                sP_P_idx = np.argmax(np.asarray(sP_amp))
                #print('sP pair', sP_P_pairs[sP_P_idx])
                
                P_pick_2 = sP_P_pairs[sP_P_idx][0]
                sP_pick = sP_P_pairs[sP_P_idx][1]
            
            if len(pP_P_pairs) > 0 and len(sP_P_pairs) == 0:
                P_pick = P_pick_1
                
                P = self.picks[P_pick]
                pP = self.picks[P_pick+1+pP_pick]
                
                peaks_id = [P, pP,0]
                print('FINAL PEAKS', peaks_id)
                pick_amplitudes = envelope[np.asarray(peaks_id)]
                pick_amplitudes[np.asarray(peaks_id) == 0] = 0
                print('FINAL PEAK AMPLITUDES', pick_amplitudes)
                self.phase_id_picks = peaks_id
                self.phase_id_pick_amplitudes = pick_amplitudes
                return
                
            if len(pP_P_pairs) == 0 and len(sP_P_pairs) > 0:

                P_pick = P_pick_2
                
                P = self.picks[P_pick]
                sP = self.picks[P_pick+1+sP_pick]
                
                peaks_id = [P, 0, sP]
                print('FINAL PEAKS', peaks_id)
                pick_amplitudes = envelope[np.asarray(peaks_id)]
                pick_amplitudes[np.asarray(peaks_id) == 0] = 0
                print('FINAL PEAK AMPLITUDES', pick_amplitudes)

                self.phase_id_picks = peaks_id
                self.phase_id_pick_amplitudes = pick_amplitudes
                return
            
            if len(pP_P_pairs) > 0 and len(sP_P_pairs) > 0:

                if P_pick_1 != P_pick_2:
                    if envelope[P_pick_1] >= envelope[P_pick_2]:
                        P_pick = P_pick_1
                        P = self.picks[P_pick]
                        pP = self.picks[P_pick+1+pP_pick]
                        sP = 0                   
                    else:
                        P_pick = P_pick_2
                        P = self.picks[P_pick]
                        pP = 0
                        sP = self.picks[P_pick+1+sP_pick]                    
                            
                if P_pick_1 == P_pick_2:
                    P_pick = P_pick_1
            
                    P = self.picks[P_pick]
                    pP = self.picks[P_pick+1+pP_pick]
                    sP = self.picks[P_pick+1+sP_pick]
                
                peaks_id = [P, pP, sP]
                print('FINAL PEAKS', peaks_id)
                pick_amplitudes = envelope[np.asarray(peaks_id)]
                pick_amplitudes[np.asarray(peaks_id) == 0] = 0
                print('FINAL PEAK AMPLITUDES', pick_amplitudes)
                
                self.phase_id_picks = peaks_id
                self.phase_id_pick_amplitudes = pick_amplitudes     
                return
        
        if phases == ['S','sS']:
            peaks_id = [0,0]
            pick_amplitudes = [0,0]
            
            pick_dt = []
            for i in range (0, len(peaks)-1):
                pick_dt.append([])     
                for j in range (i+1, len(peaks)):
                    dt = peaks[j]-peaks[i]
                    pick_dt[i].append(dt)
            
            dt_margin = 25/100
            dt_margin_plus = 1 + dt_margin
            dt_margin_minus = 1 - dt_margin
            
            S_time = (self.taup_S_time-trim_start)*self.resample
            sS_time = (self.taup_sS_time-trim_start)*self.resample
        
            model_sS_S = sS_time - S_time
            
            #print('Peak dt: ', pick_dt)
            #print('Model dt: ', model_sS_S)
            #print('Model dt +/-25%: ', model_sS_S*dt_margin_minus, model_sS_S*dt_margin_plus)
            
            sS_S_pairs = []
            sS_amp = []
            
            flag = False # replaced with True if constants used for dt margins, e.g. if earthquake is too shallow to produce a decent 25% differential time margin
                         # constants selected based upon max. dt variation expected if earthquake is 40 km wrong
            
            for i in range(len(peaks)-1):
                for j in range (len(pick_dt[i])):
                    if model_sS_S*dt_margin < 15:  #provide a +/-15 second search window when 25% of dt is less than 8.5 seconds
                        #print('Using 15 second dt window')
                        flag = True
                        if pick_dt[i][j] > model_sS_S-15*self.resample and pick_dt[i][j] < model_sS_S+15*self.resample and peaks[i+j+1] > S_time+1*15*self.resample:
                            #print(pick_dt[i][j], model_pP_P-15*self.resample, model_pP_P+15*self.resample, [i,j])
                            sS_S_pairs.append([i,j])
                            sS_amp.append(envelope[peaks[i]]+envelope[peaks[i+j+1]])
                    else: 
                        #print('Using %s of dt window' %dt_margin)
                        #print(peaks[i+j+1], P_time)
                        if pick_dt[i][j] > model_sS_S*dt_margin_minus and pick_dt[i][j] < model_sS_S*dt_margin_plus and peaks[i+j+1] > S_time+10:
                            #print(pick_dt[i][j], model_pP_P*dt_margin_minus, model_pP_P*dt_margin_plus, [i,j])
                            sS_S_pairs.append([i,j])
                            sS_amp.append(envelope[peaks[i]]+envelope[peaks[i+j+1]])
                
            
            if len(sS_S_pairs) == 0 :
                print('No eligible sS_S pairs')
                self.phase_id_picks = peaks_id
                self.phase_id_pick_amplitudes = pick_amplitudes
                return
                        
            #print('Eligible sS_S pairs: ',sS_S_pairs)
            #print('sS summed amplitudes: ', sS_amp)
           
            if len(sS_S_pairs) > 0:
                sS_S_idx = np.argmax(np.asarray(sS_amp))
                #print('sS pair', sS_S_pairs[sS_S_idx])
                
                S_pick_1 = sS_S_pairs[sS_S_idx][0]
                sS_pick = sS_S_pairs[sS_S_idx][1]
                
            
            if len(sS_S_pairs) > 0:
                S_pick = S_pick_1
                
                S = peaks[S_pick]
                sS = peaks[S_pick+1+sS_pick]
                
                peaks_id = [S, sS]
                print('FINAL PEAKS', peaks_id)
                pick_amplitudes = envelope[np.asarray(peaks_id)]
                pick_amplitudes[np.asarray(peaks_id) == 0] = 0
                print('FINAL PEAK AMPLITUDES', pick_amplitudes)
                self.phase_id_picks = peaks_id
                self.phase_id_pick_amplitudes = pick_amplitudes
                return
                
        
    @staticmethod
    def dt_peaks(late_peak, early_peak, resample):
        ''''Return differential time between 2 picked arrivals (in samples).'''
        
        # find sample point difference between pP-P peaks
        dt_peaks = late_peak - early_peak
        
        if late_peak==0 or early_peak==0:
            dt_peaks = 0
    
        # Convert sample points to time (seconds)
        dt_peaks = dt_peaks/resample
        
        return dt_peaks
   

    def finalise_peaks(self):
        ''' Returns differential times between phases, and their corresponding epicentral distances. '''
        
        if len(self.phase_id_picks)==3: # for P,pP,sP
            dt_picks_1 = self.dt_peaks(self.phase_id_picks[1], self.phase_id_picks[0], self.resample)
            dt_picks_2 = self.dt_peaks(self.phase_id_picks[2], self.phase_id_picks[0], self.resample)      
            
            self.dt_pP_P = dt_picks_1
            self.dt_sP_P = dt_picks_2
            self.epicentral_dist_pP = self.ev_array_gcarc
            self.epicentral_dist_sP = self.ev_array_gcarc     
            return 
            
            
        if len(self.phase_id_picks)==2: # for S,sS
            dt_picks_1 = self.dt_peaks(self.phase_id_picks[1], self.phase_id_picks[0], self.resample)
                    
            self.dt_sS_S = dt_picks_1
            self.epicentral_dist_sS = self.ev_array_gcarc     
            return
            
        if len(self.phase_id_picks)<2: # This should never occur?                  
            print('Less than 2 phases identified.') 
            return

    '''def relative_to_absolute_conversion(self, trim_start, phases):
        
        # Extract variables
        picks = self.phase_id_picks
        trace = self.phase_weighted_beams[self.slowness_index]
        
        if phases == ['P','pP','sP']:
            # Find noise threshold - trim region in front of P arrival
            endtime = self.event.origin_time + (self.taup_P_time * 0.98)
            starttime = endtime - 40
            
            self.P_onset_rel_time = 0
            self.pP_onset_rel_time = 0
            self.sP_onset_rel_time = 0
            self.P_onset_abs_time = 0
            self.pP_onset_abs_time = 0
            self.sP_onset_abs_time = 0
            principal_onset = 0
            
        if phases == ['S','sS']:
            # Find noise threshold - trim region in front of P arrival
            endtime = self.event.origin_time + (self.taup_S_time * 0.98)
            starttime = endtime - 20
            
            self.S_onset_rel_time = 0
            self.sS_onset_rel_time = 0          
            self.S_onset_abs_time = 0
            self.sS_onset_abs_time = 0       

        # FIND FIRST ENVELOPE SAMPLE POINT BELOW NOISE THRESHOLD WHILST BACKSTEPPING DOWN ONSET FRONT, not used.
        if picks[0] != 0:
            
            principal = picks[0]
            
            # Find noise threshold
            pre_arrival_noise = deepcopy(trace)
            pre_arrival_noise.trim(starttime, endtime, pad=True, fill_value=0)         
            abs_pre_arrival_noise = np.abs(pre_arrival_noise.data)
            noise_std = np.std(abs_pre_arrival_noise)
            noise_threshold = np.mean(abs_pre_arrival_noise) + noise_std
            #noise_threshold = np.mean(abs_pre_arrival_noise) + 0.005
            envelope = obspy.signal.filter.envelope(trace.data)

            for i in range (100*self.resample):  #backstep 100 seconds from pick
                if envelope[principal-i] <= noise_threshold:
                    principal_onset = principal-i
                    principal_onset = int(principal_onset)
                    break
            
            if principal_onset == 0:
                print('No relative to absolute conversion')
                return
            
            else:
                noise_picks = [principal_onset]
            
        
        # APPROXIMATE WAVE ONSET WITH LINES< FIND INTERSECTION
        # AND FIND WIDTH OF PEAK AT 0.98 AND FIND INTERSECTION
        if picks[0] != 0:
            
            principal = picks[0]
            
            # Find noise threshold          
            pre_arrival_noise = deepcopy(trace)
            pre_arrival_noise.trim(starttime, endtime, pad=True, fill_value=0)
            abs_pre_arrival_noise = np.abs(pre_arrival_noise.data)
            noise_std = np.std(abs_pre_arrival_noise)
            noise_threshold = np.mean(abs_pre_arrival_noise) + noise_std
            
            # Find  y diff
            prominences = peak_prominences(envelope, picks)[0]
            if prominences[0] == 0: # sometimes peak sample point is a little off from peak due to trim, refind closest peak sample point.
                max_peak = np.max(envelope)
                peaks, properties = find_peaks(envelope, prominence=0.2*max_peak)
                idx = (np.abs(peaks - principal)).argmin()
                prominences = peak_prominences(envelope, peaks)[0]
                contour_heights = envelope[peaks] - prominences
                # Find x diff
                width = peak_widths(envelope, peaks, rel_height=0.98)
                x_int = width[2][0]
                
                width = peak_widths(envelope, peaks, rel_height=0.9)
                #print(width)
                x1 = width[2][idx]
                x2 = principal
                x_diff = x2 - x1
                principal_wave_grad = prominences[idx]*0.7/x_diff # 0.7 of peak height divided peak width 
                # Find equation of the slope/line
                c = envelope[principal] - principal_wave_grad*principal
                
                # Solve equations (y = mx +c)
                x_intersection = (noise_threshold - c)/principal_wave_grad
                x_intersection = int(np.round(x_intersection, decimals = 0))
                
                
            else:
                contour_heights = envelope[picks] - prominences
                
                # Find x diff
                width = peak_widths(envelope, picks, rel_height=0.98)
                x_int = width[2][0]
                
                width = peak_widths(envelope, picks, rel_height=0.9)
                x1 = width[2][0]
                x2 = principal
                x_diff = x2 - x1
                
                principal_wave_grad = prominences[0]*0.7/x_diff  # 0.7 of peak height divided peak width 
            
                # Find equation of the slope/line
                c = envelope[principal] - principal_wave_grad*principal
                
                # Solve equations (y = mx +c)
                x_intersection = (noise_threshold - c)/principal_wave_grad
                x_intersection = int(np.round(x_intersection, decimals = 0))
                
                
            #print()
            #print('Does the line intersection do anything?')
            #print('Lines intersection', x_intersection)
            #print('Width intersection', x_int)
            #print('Noise intersection', noise_picks[0])
            #print()
            
            # If width pick is within 5 seconds of the original P arrival, use it. Else use the line intersection pick, unless it is too high in amplitude. In that case, use the width pick again.
            
            if x_int < picks[0] and x_int > picks[0]-5*self.resample:
                new_pick = int(np.round(x_int,decimals=0))
            
            else:
                if envelope[x_intersection] < 0.1:
                    new_pick = x_intersection
                else:
                    new_pick = int(np.round(x_int,decimals=0))
        
            principal_rel = (new_pick/self.resample) + trim_start
            principal_onset = self.event.origin_time + principal_rel # sample no to seconds, added to origin time
            sample_no_diff = principal - new_pick

            if  picks[1] != 0:
                # Convert pP and sP to onset times
                phase2 = picks[1]-sample_no_diff
                phase2_rel = (phase2/self.resample) + trim_start
                phase2_onset = self.event.origin_time + phase2_rel # sample no to seconds, added to origin time
            
            if phases == ['P','pP','sP']:       
                if picks[2] != 0:
                    phase3_rel = picks[2]-sample_no_diff
                    phase3_rel = (phase3_rel/self.resample) + trim_start
                    phase3_onset = self.event.origin_time + phase3_rel # sample no to seconds, added to origin time
        
                self.P_onset_rel_time = principal_rel
                self.pP_onset_rel_time = phase2_rel
                self.sP_onset_rel_time = phase3_rel
                self.P_onset_abs_time = principal_onset
                self.pP_onset_abs_time = phase2_onset
                self.sP_onset_abs_time = phase3_onset    
                
            if phases == ['S','sS']:       
                self.S_onset_rel_time = principal_rel
                self.sS_onset_rel_time = phase2_rel
                self.S_onset_abs_time = principal_onset
                self.sS_onset_abs_time = phase2_onset     

        return'''

    def create_PhaseNet_beams(self):
        ''' Assemble P wave beampacks in x and y slowness space.'''
        
        # Trim QC'd traces around P wave
        self.stream.detrend(type='demean')
        self.stream.normalize()
        stream_vespa = self.stream.copy()
        
        trima = 7 
        trimb = 10
        
        # Define trace trims
        starttime = self.event.origin_time + self.taup_P_time - trima
        endtime = self.event.origin_time + self.taup_P_time + trimb
        
        # Copy stream for trimming
        stream_vespa.trim(starttime, endtime, pad=True, fill_value=0)     
        stream_vespa.normalize()
        
        #stream_vespa.plot()
        #plt.show()
        
        # Set up Sx and Sy arrays
        sxmin = 0
        sxmax = 0.03
        symin = -0.09
        symax = -0.06
        s_space = 0.0001
        
        # get number of points.
        nsx = int(np.round(((sxmax - sxmin) / s_space), 0) + 1)
        nsy = int(np.round(((symax - symin) / s_space), 0) + 1)

        # make empty array for output.
        results_arr = np.zeros((nsy * nsx, 5))
        pws_grd = np.zeros((nsx, nsy))

        slow_sx = np.linspace(sxmin, sxmax + s_space, nsx)
        slow_sy = np.linspace(symin, symax + s_space, nsy)
        #print('slow', slow_sx, slow_sy)
        
        # Get slowness and backazimuth
        #[slw = (sx^2 + sy^2)^1/2]
        #[baz = arctan(sx/sy)]
        for k in range (len(slow_sx)):
            for j in range (len(slow_sy)):
                abs_slw = (slow_sx[k]**2 + slow_sy[j]**2)**0.5
                baz = np.degrees(np.arctan2(slow_sx[k], slow_sy[j]))                      
                az = baz % -360 + 180  
                
                if baz < 0:
                    baz += 360
                if az < 0:
                    az += 360                                                                                                                       
                print('bazaz',slow_sx[k], slow_sy[j], abs_slw, baz, az)
                
                #if np.round(abs_slw, 3) == 0.068 and np.round(baz,0) == 160:
                
                # Find distance along backazimuth to the array centre, per station in array
                #baz = 160.3
                #abs_slw = 0.068
                baz_rad = math.radians(baz)
                
                # trace by trace
                shifted_traces = np.zeros((len(stream_vespa), len(stream_vespa[0].data)))
                
                shifted_stream = Stream()
                phi = 0
                for i in range (len(self.stations.stla)):
                    baz_dist = (self.array_longitude - self.stations.stlo[i])*math.sin(baz_rad) + (self.array_latitude - self.stations.stla[i])*math.cos(baz_rad)
                    baz_dist_km = degrees2kilometers(baz_dist)
                
                    # Find time difference between station trace and array centre    
                    dt = (abs_slw*baz_dist_km)*-1
                    
                    # shift trace 
                    tr = stream_vespa[i]
                    #tr.plot()
                    data = np.roll(tr.data,int(np.round(dt*self.resample)))
                    tr.data = data
                    #tr.plot()
                    shifted_stream.append(tr)
                    #shifted_stream.plot()
                    shifted_traces[i] = data
                    
                    # for phase weighting beams later
                    phi = phi+np.exp(1j*np.angle(hilbert(data))) 
            
                #print(shifted_traces)
                #print(phi)
                #shifted_stream.plot()
                linear_stack = np.sum(shifted_traces, axis=0)/len(stream_vespa)
                #plt.plot(linear_stack)
                #plt.show()
                pws_stack = linear_stack*(np.abs(phi))**4
                print(np.shape(pws_stack))
                #plt.plot(pws_stack)
                #plt.show()
                pws_pt = np.sum(pws_stack**2)
                print(pws_pt)
                print(slow_sx[k],slow_sy[j])
                #plt.plot(pws_stack)
                #plt.show()
                pws_grd[k,j] = pws_pt
                print(pws_grd[k,j])
                #plt.scatter(1,1)
                #plt.show()
                x=k
                y=j
                                            
        # Normalise power grid
        print(pws_grd)
        print(pws_grd[x,y])
        max_pt = np.max(abs(pws_grd))
        c = plt.imshow(pws_grd)
        plt.colorbar(c)
        plt.show()
        print('max', max_pt)
        norm_grd = np.divide(pws_grd, max_pt)
        c= plt.imshow(norm_grd)
        plt.colorbar(c)
        #plt.show()
        return 
                    
    def prep_PhaseNet_beams(self, ev_dir):
        """
        Function to search over a range of slowness vectors, described in cartesian coordinates, and measure
        the coherent power. Stacks the traces using linear stack.
        Parameters
        ----------
        traces : 2D numpy array of floats
            2D array containing the traces the user wants to conduct analysis with. Shape of [n,p] where n
            is the number of traces and p is the points in each trace.
        sampling_rate : float
            Sampling rate of the data points in s^-1.
        geometry : 2D array of floats
            2D array describing the lon lat and elevation of the stations [lon,lat,depth]
        distance : float
            Epicentral distance from the event to the centre of the array.
        sxmax : float
            Maximum magnitude of slowness on x axis, used for creating the slowness grid.
        sxmin : float
            Minimun magnitude of the slowness on x axis, used for creating the slowness grid.
        symax : float
            Maximum magnitude of slowness on y axis, used for creating the slowness grid.
        symin : float
            Minimun magnitude of the slowness on y axis, used for creating the slowness grid.
        s_space : float
            The slowness interval for each step e.g. 0.1.
        type : string
            Will calculate either using a curved (circ) or plane (plane) wavefront. default
            is 'circ'.
        elevation : bool
            If True, elevation corrections will be added. If False, no elevation
            corrections will be accounted for. Default is False.
        incidence : float
            Not used unless elevation is True. Give incidence angle from vertical
            at the centre of the array to calculate elevation corrections. Default is 90.

        Returns
        -------
        lin_tp : 2D numpy array of floats.
            Linear stack power grid.
        results_arr : 2D numpy array of floats.
            Contains power values for:
            [slow_x, slow_y, power_lin, baz, abs_slow]
        peaks : 2D array of floats.
            Array contains 3 rows describing the X,Y points
            of the maximum power value for phase weighted,
            linear and F-statistic respectively.
        """
        # Trim QC'd traces around P wave
        self.stream.detrend(type='demean')
        self.stream.normalize()
        stream_vespa = self.stream.copy()
        
        trima = 7 
        trimb = 10
        
        # Define trace trims
        starttime = self.event.origin_time + self.taup_P_time - trima
        endtime = self.event.origin_time + self.taup_P_time + trimb
        
        # Copy stream for trimming
        stream_vespa.trim(starttime, endtime, pad=True, fill_value=0)     
        stream_vespa.normalize()
        
        stream = stream_vespa  
        traces = []
        for st in stream:
            #print(st.data)
            traces.append(st.data)
        traces = np.asarray(traces)
        
        print('traces')
        
        sampling_rate = 10
        geometry = [self.stations.stlo, self.stations.stla, self.stations.stel]
        centre_x = self.array_longitude
        centre_y = self.array_latitude
        centre_z = self.array_elevation
        distance = self.ev_array_gcarc
        sxmin = -6
        sxmax = 5.9
        symin = -10
        symax = 5.9
        s_space = 0.1
        type='circ'
        elevation=False
        incidence=90
        
        def roll_1D(x, p):
            """
            Function to shift traces stored in a 1D array (x)
            by the number of points stored in 1D array (p).
            optimised in Numba.

            Parameters
            ----------
            x : 1D array of floats
                array to be shifted

            p : int
                points to shift array by.

            Returns
            -------
            x : 1D array of floats
                shifted array by points p.

            """

            p = p*-1
            x = np.append(x[p:], x[:p])
            return x

        def roll_2D(array, shifts):
            """
            Function to shift traces stored in a 2D array (x)
            by the number of points stored in 1D array (p).
            optimised in Numba.

            Parameters
            ----------
            array : 2D array of floats
                Traces/time series to be shifted.

            shifts : 1D array of floats
                points to shift the respective time series by.

            Returns
            -------
            array_new : 2D array of floats
                2D array of shifted time series.

            """

            n = array.shape[0]
            array_new = np.copy(array)
            for i in range(n):
                array_new[int(i)] = roll_1D(array_new[int(i)],int(shifts[int(i)]))
                # array_new[i] = np.roll(array[i],int(shifts[i]))


            return array_new
        
        def haversine_deg(lat1, lon1, lat2, lon2):
            """
            Function to calculate the distance in degrees between two points on a sphere.

            Parameters
            ----------
            lat1 : float
                Latitiude of point 1.

            lat1 : float
                Longitiude of point 1.

            lat2 : float
                Latitiude of point 2.

            lon2 : float
                Longitude of point 2.

            Returns
            -------
                d : float
                    Distance between the two points in degrees.
            """

            dlat = np.radians(lat2 - lat1)
            dlon = np.radians(lon2 - lon1)
            a = (np.sin(dlat / 2)) ** 2 + np.cos(np.radians(lat1)) * np.cos(
                np.radians(lat2)
            ) * (np.sin(dlon / 2)) ** 2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
            d = np.degrees(c)
            return d


        def coords_lonlat_rad_bearing(lat1, lon1, dist_deg, brng):
            """
            Returns the latitude and longitude of a new cordinate that is the defined distance away and
            at the correct bearing from the starting point.

            Parameters
            ----------
            lat1 : float
                Starting point latitiude.

            lon1 : float
                Starting point longitude.

            dist_deg : float
                Distance from starting point in degrees.

            brng : float
                Angle from north describing the direction where the new coordinate is located.

            Returns
            -------
            lat2 : float
                Longitude of the new cordinate.
            lon2 : float
                Longitude of the new cordinate.
            """

            brng = np.radians(brng)  # convert bearing to radians
            d = np.radians(dist_deg)  # convert degrees to radians
            lat1 = np.radians(lat1)  # Current lat point converted to radians
            lon1 = np.radians(lon1)  # Current long point converted to radians

            lat2 = np.arcsin(
                (np.sin(lat1) * np.cos(d)) + (np.cos(lat1) * np.sin(d) * np.cos(brng))
            )
            lon2 = lon1 + np.arctan2(
                np.sin(brng) * np.sin(d) * np.cos(lat1), np.cos(d) - np.sin(lat1) * np.sin(lat2)
            )

            lat2 = np.degrees(lat2)
            lon2 = np.degrees(lon2)

            # lon2 = np.where(lon2 > 180, lon2 - 360, lon2)
            # lon2 = np.where(lon2 < -180, lon2 + 360, lon2)

            if lon2 > 180:
                lon2 -= 360
            elif lon2 < -180:
                lon2 += 360
            else:
                pass

            return lat2, lon2
        
        def calculate_time_shifts(
            geometry, abs_slow, baz, distance, centre_x, centre_y, type="circ"
        ):
            """
            Calculates the time delay for each station relative to the time the phase
            should arrive at the centre of the array. Will use either a plane or curved
            wavefront approximation.

            Parameters
            ----------
            geometry : 2D array of floats
                2D array describing the lon lat and elevation of the stations [lon,lat,depth]

            distance : float
                Epicentral distance from the event to the centre of the array.

            abs_slow : float
                Horizontal slowness you want to align traces over.

            baz : float
                Backazimuth you want to align traces over.

            centre_x : float
                Mean longitude.

            centre_y : float
                Mean latitude.

            type : string
                Will calculate either using a curved (circ) or plane (plane) wavefront.

            Returns
            -------
            times : 1D numpy array of floats
                The arrival time for the phase at
                each station relative to the centre.

            shifts : 1D numpy array of floats
                The time shift to align on a phase at
                each station relative to the centre.
            """
            slow_x = abs_slow * np.sin(np.radians(baz))
            slow_y = abs_slow * np.cos(np.radians(baz))

            lat_new, lon_new = coords_lonlat_rad_bearing(
                lat1=centre_y, lon1=centre_x, dist_deg=distance, brng=baz
            )

            if type == "circ":
                print(geometry[0])
                dists = haversine_deg(lat1=lat_new, lon1=lon_new, lat2=geometry[1], lon2=geometry[0])

                # get the relative distance
                dists_rel = dists - distance

                # get the travel time for this distance
                times = dists_rel * abs_slow

                # the correction will be dt *-1
                shifts = times * -1


            elif type == "plane":

                shifts = ((geometry[0] - lon_new) * slow_x) + ((geometry[1] - lat_new) * abs_slow)

                times = shifts * -1

            else:
                print("not plane or circ")

            return shifts, times
        
        def get_slow_baz(slow_x, slow_y, dir_type):
            """
            Returns the backazimuth and slowness magnitude of a slowness vector given its x and y components.

            Parameters
            ----------
            slow_x : float
                X component of slowness vector.

            slow_y : float
                Y component of slowness vector.

            dir_type : string
                How do you want the direction to be measured, backazimuth (baz) or azimuth (az).

            Returns
            -------
            slow_mag: float
                Magnitude of slowness vector.
            baz : float
                Backazimuth of slowness vector
            azimuth : float
                Azimuth of slowness vector
            """

            slow_mag = np.sqrt(slow_x ** 2 + slow_y ** 2)
            azimuth = np.degrees(np.arctan2(slow_x, slow_y))  # * (180. / math.pi)

            # % = mod, returns the remainder from a division e.g. 5 mod 2 = 1
            baz = azimuth % -360 + 180

            # make baz positive if it's negative:
            # baz = np.where(baz < 0, baz + 360, baz)
            # azimuth = np.where(azimuth < 0, azimuth + 360, azimuth)
            # baz = np.where(baz > 360, baz - 360, baz)
            # azimuth = np.where(azimuth > 360, azimuth - 360, azimuth)

            if baz < 0:
                baz += 360
            if azimuth < 0:
                azimuth += 360



            if dir_type == "baz":
                return slow_mag, baz
            elif dir_type == "az":
                return slow_mag, azimuth
            else:
                pass

        def get_max_power_loc(tp, sxmin, symin, s_space):
            """
            Finds the location of the maximum power value within a given
            slowness space.

            Parameters
            ----------
            tp : 2D array of floats
                2D array of values to find the maxima in.

            sxmin : float
                Minimum value on the x axis.

            symin : float
                Minimum value on the y axis.

            s_space : float
                Step interval. Assumes x and y axis spacing is the same.

            Returns
            -------
            peaks : 2D numpy array of floats
                2D array of: [[loc_x,loc_y]]
            """

            peaks = np.empty((1, 2))

            iy, ix = np.where(tp == np.amax(tp))

            slow_x_max = sxmin + (ix[0] * s_space)
            slow_y_max = symin + (iy[0] * s_space)

            peaks[int(0)] = np.array([slow_x_max, slow_y_max])

            return peaks, ix, iy

        def shift_traces(
            traces,
            geometry,
            abs_slow,
            baz,
            distance,
            centre_x,
            centre_y,
            sampling_rate,
            elevation=False,
            incidence=90,
            type="circ",
        ):
            """
            Shifts the traces using the predicted arrival times for a given backazimuth and slowness.

            Parameters
            ----------
            traces : 2D numpy array of floats
                A 2D numpy array containing the traces that the user wants to stack.

            geometry : 2D array of floats
                2D array describing the lon lat and elevation of the stations [lon,lat,depth]

            distance : float
                Epicentral distance from the event to the centre of the array.

            abs_slow : float
                Horizontal slowness you want to align traces over.

            baz : float
                Backazimuth you want to align traces over.

            centre_x : float
                Mean longitude.

            centre_y : float
                Mean latitude.

            sampling_rate : float
                Sampling rate of the data points in s^-1.

            elevation : bool
                If True, elevation corrections will be added. If False, no elevation
                corrections will be accounted for. Default is False.

            incidence : float
                Not used unless elevation is True. Give incidence angle from vertical
                at the centre of the array to calculate elevation corrections.
                Default is 90.

            type : string
                Will calculate either using a curved (circ) or plane (plane) wavefront.

            Returns
            -------
                shifted_traces : 2D numpy array of floats
                    The input traces shifted by the predicted arrival time
                    of a curved wavefront arriving from a backazimuth and
                    slowness.
            """

            if elevation == False:
                shifts, times = calculate_time_shifts(
                                                      geometry,
                                                      abs_slow,
                                                      baz,
                                                      distance,
                                                      centre_x,
                                                      centre_y,
                                                      type=type,
                                                      )
            elif elevation == True:
                shifts, times = calculate_time_shifts_elevation(
                                                      incidence,
                                                      geometry,
                                                      abs_slow,
                                                      baz,
                                                      distance,
                                                      centre_x,
                                                      centre_y,
                                                      type=type
                                                      )
            print('here4')
            pts_shifts = shifts * sampling_rate

            shifted_traces = roll_2D(traces, pts_shifts)


            return shifted_traces



        ntrace = len(traces)
        print('ntrace', ntrace)
        
        # get number of plen(buff)oints.
        nsx = int(np.round(((sxmax - sxmin) / s_space), 0) + 1)
        nsy = int(np.round(((symax - symin) / s_space), 0) + 1)
        print('nsx', nsx, nsy)
        
        # make empty array for output.
        results_arr = np.zeros((nsy * nsx, 5))
        lin_tp = np.zeros((nsy, nsx))
        backazimuth = np.zeros((np.shape(lin_tp)))
        slowness = np.zeros((np.shape(lin_tp)))
        slw_x = np.zeros((np.shape(lin_tp)))
        slw_y = np.zeros((np.shape(lin_tp)))
        lin_stack_beam_grd = np.zeros([nsy, nsx, len(stream[0])])

        slow_xs = np.linspace(sxmin, sxmax + s_space, nsx)
        slow_ys = np.linspace(symin, symax + s_space, nsy)
        print('slow_sx')
        
        # loop over slowness grid
        for i in range(slow_ys.shape[0]):
            for j in range(slow_xs.shape[0]):
                sx = float(slow_xs[int(j)])
                sy = float(slow_ys[int(i)])
                
                # get the slowness and backazimuth of the vector
                abs_slow, baz = get_slow_baz(sx, sy, "az")
                slowness[i,j] = abs_slow
                backazimuth[i,j] = baz
                slw_x[i,j] = sx
                slw_y[i,j] = sy
                
                point = int(int(i) + int(slow_xs.shape[0] * j))
                
                # Call function to shift traces
                shifted_traces_lin = shift_traces(
                    traces=traces,
                    geometry=geometry,
                    abs_slow=float(abs_slow),
                    baz=float(baz),
                    distance=float(distance),
                    centre_x=float(centre_x),
                    centre_y=float(centre_y),
                    sampling_rate=sampling_rate,
                    type=type,
                    elevation=elevation,
                    incidence=incidence
                )

                lin_stack = np.sum(shifted_traces_lin, axis=0) / ntrace
                lin_stack_beam_grd[i,j] = lin_stack
                
                # linear stack
                power_lin = np.sum(lin_stack**2)

                lin_tp[i, j] = power_lin

                results_arr[point] = np.array([sx, sy, power_lin, baz, abs_slow])

        # now find the peak in this:
        peaks = np.empty((1, 2))
        peaks[int(0)], idx_x, idx_y = get_max_power_loc(
            tp=lin_tp, sxmin=sxmin, symin=symin, s_space=s_space
        )

        results_arr[:, 2] /= results_arr[:, 2].max() # normalised power grd
        
        # AB edits!
        
        # normalise lin_tp
        norm_lin_tp = np.zeros((np.shape(lin_tp)))
        points = []
        max_power = np.max(lin_tp)
        print(max_power)
        
        for i in range(slow_ys.shape[0]):
            for j in range(slow_xs.shape[0]):           
                norm_lin_tp[i,j] = lin_tp[i,j]/max_power
                points.append([j,i]) 
                        
        fig = plt.figure(figsize=(8,6))
        c = plt.pcolormesh(norm_lin_tp) #, vmin=0, vmax=1)
        plt.scatter(idx_x[0], idx_y[0], c='r', marker='o', s=2)
        plt.axhline(np.linspace(0, slow_xs.shape[0],7)[5] ,linestyle=':', c='white')
        plt.axvline(np.linspace(0, slow_ys.shape[0],9)[3] ,linestyle=':', c='white')
        cbar = plt.colorbar(c)
        plt.xticks(ticks=np.linspace(0, slow_xs.shape[0],7), labels=np.linspace(slow_xs[0], slow_xs[-1],7))
        plt.yticks(ticks=np.linspace(0, slow_ys.shape[0],9), labels=np.linspace(slow_ys[0], slow_ys[-1],9))
        plt.xlabel('Sx (s/degree)')
        plt.ylabel('Sy (s/degree)')
        cbar.set_label('Normalised Power')
        plt.axis('scaled')
        coords = plt.contour(norm_lin_tp, [0.8], colors='white')
        #plt.show()
               
        # Define contour and coords within it
        v = coords.collections[0].get_paths()[0].vertices
        coords_x = v[:,0]
        coords_y = v[:,1]
        np.save('contour_coords', v, allow_pickle=True)
        
        #plt.scatter(coords_x, coords_y)
        #plt.show()
        
        temp_list = []
        for a,b in zip(coords_x, coords_y):
            temp_list.append([a,b])
            
        polygon = np.array(temp_list)
        path = matplotlib.path.Path(polygon)
        mask = path.contains_points(points)
        xv,yv = np.meshgrid(slow_xs, slow_ys)
        mask.shape = xv.shape
        #print('mask', mask)
        
        #plt.pcolormesh(norm_lin_tp)
        #plt.pcolormesh(mask, alpha=0.5)
        #plt.show()
        
        # mask out beam points outside contour line
        contour_power = norm_lin_tp[mask]
        backazimuth = backazimuth[mask]
        slowness = slowness[mask]
        slw_x = slw_x[mask]
        slw_y = slw_y[mask]
        lin_stack_beam_grd = lin_stack_beam_grd[mask]
        
        #print(backazimuth)
        #print(slw_y)
        #print(lin_stack_beam_grd)
        
        # Save out as new directory system for phasenet
        # Create results directory (if it doesn't exist)
        try:
            phasenet_dir = os.path.join(ev_dir, 'PhaseNet_beams')
            directory = '%s' %phasenet_dir
            os.mkdir(directory)
            print('Directory %s created' %directory )

        except FileExistsError:
            pass
            
        # Save out MSEEDS
        
        for b in range (len(lin_stack_beam_grd)):
            #tr = obspy.read('/localhome/not-backed-up/ee18ab/00_ANDES_DATA/ObspyDMT_data/20100523_224651.a/raw/3E.CUKY..*')
            #tr=tr[0]
            tr = deepcopy(stream[0])
            #print(tr)
            tr.data = lin_stack_beam_grd[b]
            tr.network = 'ZZ'
            tr.station = str(self.array_no)
            tr.id = str(tr.network) + '.' + str(tr.station) + '..BHZ'
            #print(tr)
            tr.resample(100)
            #tr.plot()
            
            data = np.array(tr.data)
            #data = []
            #for trace in tr:
            #    data.append(trace.data)
            #data = np.array(data).T
            data_id = tr.get_id()[:-1]
            timestamp = tr.stats.starttime.datetime.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]
            
            if contour_power[b] == 1:
                tr.write(phasenet_dir + '/' + str(self.array_no) + '_BEST_' + str(slw_x[b]) + '_' + str(slw_y[b])+'.MSEED', format='MSEED')
            else:
                tr.write(phasenet_dir + '/' + str(self.array_no) + '_' + str(slw_x[b]) + '_' + str(slw_y[b])+'.MSEED', format='MSEED')
        
        return fig        
        
    
    def plot_1_component(self, trim_start, trim_end, beam=True, vespagram=True, start_phase = 'P', end_phase = 'sP', xlim_secs_pre_start=40, xlim_secs_post_end=40, x_tick_interval=100, picks=False):
        ''' Plot the optimum beams for a component. '''
        
        # -------- Set up plot basics --------
        # Set x ticks and labels with relative time from origin 
        x_ticks = np.arange(0, len(self.PW_optimum_beam.data), x_tick_interval*self.resample)
        relative_time = np.arange(self.relative_time[0], self.relative_time[-1]+(x_tick_interval*2), x_tick_interval) 
        relative_time = np.round(relative_time,0)
        relative_time = relative_time[:len(x_ticks)]
        relative_time = [int(t) for t in relative_time]
        
        # x axis limits in sample number (not time)
        if start_phase == 'P':
            xlim_start = ((self.taup_P_time-trim_start)*self.resample) - (xlim_secs_pre_start*self.resample)
        elif start_phase == 'S':
            xlim_start = ((self.taup_S_time-trim_start)*self.resample) - (xlim_secs_pre_start*self.resample)
        
        if end_phase == 'sP':
            xlim_end = ((self.taup_sP_time-trim_start)*self.resample) + (xlim_secs_post_end*self.resample)
        elif end_phase == 'sS':
            xlim_end = ((self.taup_sS_time-trim_start)*self.resample) + (xlim_secs_post_end*self.resample)
        #print('xlim', xlim_start, xlim_end, self.taup_P_time, trim_start, xlim_secs_pre_start, self.taup_sS_time, trim_end, xlim_secs_post_end)
        
        # Convert TauP model arrivals to sample no.
        def relative_time_to_sample_no(time, trim_start, resample):
            return (time-trim_start)*resample
        
        taup_P_sn = relative_time_to_sample_no(self.taup_P_time, trim_start, self.resample)
        taup_pP_sn = relative_time_to_sample_no(self.taup_pP_time, trim_start, self.resample)
        taup_sP_sn = relative_time_to_sample_no(self.taup_sP_time, trim_start, self.resample)
        taup_S_sn = relative_time_to_sample_no(self.taup_S_time, trim_start, self.resample)
        taup_sS_sn = relative_time_to_sample_no(self.taup_sS_time, trim_start, self.resample)
        
        if vespagram == True:
            vmax = np.max(self.vespa_grd)
            vmin = np.min(self.vespa_grd)
            if abs(vmin) >= abs(vmax):
                vmax= abs(vmin)
                
            y_ticks = np.linspace(0, len(self.slowness_range), 5)
            y_tick_labels = np.round(np.linspace(self.slowness_range[0], self.slowness_range[-1], 5),3)
        
        # ------ Specific plots --------
        if beam == True and vespagram == False:
            
            pw_beam, ax = plt.subplots(1,1, figsize=(10,5))
            ax.plot(self.PW_optimum_beam, color = 'k', linewidth = '1', linestyle = '--', zorder=0.5, label='Beam')
            ax.plot(self.PW_optimum_beam_envelope, linewidth = '2', zorder = 1, color='k', label = 'Envelope')
            ax.set_xlabel('Time (s)', fontsize = 20)                   
            ax.set_ylabel('Velocity (m/s)', fontsize = 20)
            ax.set_ylim(-1.1, 1.1)
            
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(relative_time, rotation=-45, ha='left', fontsize=16) 
            ax.tick_params(axis='both', labelsize=16)
            ax.scatter(taup_P_sn, -1.1, marker = '^', s = 100, color = 'k', label='Modelled Arrivals')
            ax.scatter(taup_pP_sn, -1.1, marker = '^', s = 100, color = 'k')
            ax.scatter(taup_sP_sn, -1.1, marker = '^', s = 100, color = 'k')
            ax.scatter(taup_S_sn, -1.1, marker = '^', s = 100, color = 'k', label='Modelled Arrivals')
            if picks != False:
                ax.scatter(picks, self.PW_optimum_beam_envelope[picks], marker='o', c='yellow', s=100)
            
            ax.scatter(taup_sS_sn, -1.1, marker = '^', s = 100, color = 'k')
            ax.set_xlim(xlim_start, xlim_end)
            return pw_beam
            
        if beam == False and vespagram == True:
            pw_vespagram, ax = plt.subplots(1,1, figsize=(10,5))
            ax.pcolormesh(self.vespa_grd, cmap = 'seismic', vmin = (-vmax*1.05), vmax = (vmax*1.05), rasterized=True)            
            ax.set_xlabel('Time (s)', fontsize = 20)           
            ax.set_ylabel('Slowness (s/km)', fontsize = 20)
            ax.set_ylim(0, len(self.vespa_grd))
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(relative_time, rotation=-45, ha='left', fontsize=16)
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_tick_labels, fontsize=16)  
            ax.tick_params(axis='both', labelsize=16)
            ax.scatter(taup_P_sn, 1, marker = '^', s = 100, color = 'k', label='Modelled Arrivals')
            ax.scatter(taup_pP_sn, 1, marker = '^', s = 100, color = 'k')
            ax.scatter(taup_sP_sn, 1, marker = '^', s = 100, color = 'k')
            ax.scatter(taup_S_sn, 1, marker = '^', s = 100, color = 'k', label='Modelled Arrivals')
            
            if picks != False:
                ax.scatter(picks, np.ones(len(picks))*self.slowness_index, marker='o', c='yellow', s=100)
            
            ax.scatter(taup_sS_sn, 1, marker = '^', s = 100, color = 'k')
            ax.set_xlim(xlim_start, xlim_end)
            return pw_vespagram    
            
        if beam == True and vespagram == True:
            pw_beam_vespagram, axis = plt.subplots(2,1, sharex=True, figsize=(10,10))
            
            axis[0].pcolormesh(self.vespa_grd, cmap = 'seismic', vmin = (-vmax*1.05), vmax = (vmax*1.05), rasterized=True)
            axis[0].scatter(taup_P_sn, 1, marker = '^', s = 100, color = 'k', label='Modelled Arrivals')
            axis[0].scatter(taup_pP_sn, 1, marker = '^', s = 100, color = 'k')
            axis[0].scatter(taup_sP_sn, 1, marker = '^', s = 100, color = 'k')
            axis[0].scatter(taup_S_sn, 1, marker = '^', s = 100, color = 'k', label='Modelled Arrivals')
            if picks != False:
                axis[0].scatter(picks, np.ones(len(picks))*self.slowness_index, marker='o', c='yellow', s=100)
            axis[0].scatter(taup_sS_sn, 1, marker = '^', s = 100, color = 'k')
            axis[0].set_yticks(y_ticks)
            axis[0].set_yticklabels(y_tick_labels, fontsize=16) 
            
            axis[1].plot(self.PW_optimum_beam, color = 'k', linewidth = '1', linestyle = '--', zorder=0.5, label='Beam')
            axis[1].plot(self.PW_optimum_beam_envelope, linewidth = '2', zorder = 1, color='k', label = 'Envelope')
            axis[1].set_ylim(-1.1, 1.1)
            axis[1].set_xlabel('Time (s)', fontsize = 20) 
            axis[1].scatter(taup_P_sn, -1.1, marker = '^', s = 100, color = 'k', label='Modelled Arrivals')
            axis[1].scatter(taup_pP_sn, -1.1, marker = '^', s = 100, color = 'k')
            axis[1].scatter(taup_sP_sn, -1.1, marker = '^', s = 100, color = 'k')
            axis[1].scatter(taup_S_sn, -1.1, marker = '^', s = 100, color = 'k', label='Modelled Arrivals')
            axis[1].scatter(taup_sS_sn, -1.1, marker = '^', s = 100, color = 'k')
            
            if picks != False:
                axis[1].scatter(picks, self.PW_optimum_beam_envelope[picks], marker='o', c='yellow', s=100)
            
            for ax in axis:                  
                ax.set_ylabel('Velocity (m/s)', fontsize = 20) 
                ax.set_xticks(x_ticks)
                ax.set_xticklabels(relative_time, rotation=-45, ha='left', fontsize=16) 
                ax.tick_params(axis='both', labelsize=16)
                ax.set_xlim(xlim_start, xlim_end)
            return pw_beam_vespagram
    
    @staticmethod
    def plot_3_components(array_Z, array_N, array_E, trim_start, trim_end, resample, beams=True, vespagrams=False, xlim_secs_preP=40, xlim_secs_postsS=40, x_tick_interval=100):
        ''' Plots either ZNE beams on one figures, or ZNE vespagrams on one figure. '''
        # -------- Set up plot basics --------
        # Set x ticks and labels with relative time from origin 
        x_ticks = np.arange(0, len(array_Z.PW_optimum_beam.data), x_tick_interval*resample)
        relative_time = np.arange(array_Z.relative_time[0], array_Z.relative_time[-1]+(x_tick_interval*2), x_tick_interval) 
        relative_time = relative_time[:len(x_ticks)]
        relative_time = [int(np.round(t,0)) for t in relative_time]
        
        # x axis limits in sample number (not time)
        xlim_start = ((array_Z.taup_P_time-trim_start)*resample) - (xlim_secs_preP*resample)
        xlim_end = ((array_Z.taup_sS_time-trim_start)*resample) + (xlim_secs_postsS*resample)
        print('xlim', xlim_start, xlim_end)
        
        # Convert TauP model arrivals to sample no.
        def relative_time_to_sample_no(time, trim_start, resample):
            return (time-trim_start)*resample
        
        taup_P_sn = relative_time_to_sample_no(array_Z.taup_P_time, trim_start, resample)
        taup_pP_sn = relative_time_to_sample_no(array_Z.taup_pP_time, trim_start, resample)
        taup_sP_sn = relative_time_to_sample_no(array_Z.taup_sP_time, trim_start, resample)
        taup_S_sn = relative_time_to_sample_no(array_Z.taup_S_time, trim_start, resample)
        taup_sS_sn = relative_time_to_sample_no(array_Z.taup_sS_time, trim_start, resample)           
        
        if vespagrams == True:
            vmax = np.max(array_Z.vespa_grd)
            vmin = np.min(array_Z.vespa_grd)
            if abs(vmin) >= abs(vmax):
                vmax= abs(vmin)
                              
            y_ticks = np.linspace(0, len(array_Z.slowness_range), 5)
            y_tick_labels = np.round(np.linspace(array_Z.slowness_range[0], array_Z.slowness_range[-1], 5),3)
              
        if beams == True and vespagrams == False:        
            # Beams
            pw_beams_ZNE, axis = plt.subplots(3,1, sharex=True, figsize=(10,15))
            
            axis[0].plot(array_Z.PW_optimum_beam, color = 'k', linewidth = '1', linestyle = '--', zorder=0.5, label='Z Beam')
            axis[0].plot(array_Z.PW_optimum_beam_envelope, linewidth = '2', zorder = 1, color='k', label = 'Z Envelope')
            
            axis[1].plot(array_N.PW_optimum_beam, color = 'k', linewidth = '1', linestyle = '--', zorder=0.5, label='N Beam')
            axis[1].plot(array_N.PW_optimum_beam_envelope, linewidth = '2', zorder = 1, color='k', label = 'N Envelope')
            
            axis[2].plot(array_E.PW_optimum_beam, color = 'k', linewidth = '1', linestyle = '--', zorder=0.5, label='E Beam')
            axis[2].plot(array_E.PW_optimum_beam_envelope, linewidth = '2', zorder = 1, color='k', label = 'E Envelope')
            axis[2].set_xlabel('Time (s)', fontsize = 20)    
               
            for ax in axis:
                ax.set_ylabel('Velocity (m/s)', fontsize = 20)
                ax.set_ylim(-1.1, 1.1)
                ax.set_xticks(x_ticks)
                ax.set_xticklabels(relative_time, rotation=-45, ha='left', fontsize=16) 
                ax.tick_params(axis='both', labelsize=16)
                ax.scatter(taup_P_sn, -1.1, marker = '^', s = 100, color = 'k', label='Modelled Arrivals')
                ax.scatter(taup_pP_sn, -1.1, marker = '^', s = 100, color = 'k')
                ax.scatter(taup_sP_sn, -1.1, marker = '^', s = 100, color = 'k')
                ax.scatter(taup_S_sn, -1.1, marker = '^', s = 100, color = 'k', label='Modelled Arrivals')
                ax.scatter(taup_sS_sn, -1.1, marker = '^', s = 100, color = 'k')
                ax.set_xlim(xlim_start, xlim_end)
            return pw_beams_ZNE
            
        if beams == False and vespagrams == True:
            # Vespagrams
            pw_vespagrams_ZNE, axis = plt.subplots(3,1, sharex=True, figsize=(10,15))
            
            axis[0].pcolormesh(array_Z.vespa_grd, cmap = 'seismic', vmin = (-vmax*1.05), vmax = (vmax*1.05), rasterized=True)
            
            axis[1].pcolormesh(array_N.vespa_grd, cmap = 'seismic', vmin = (-vmax*1.05), vmax = (vmax*1.05), rasterized=True)
            
            axis[2].pcolormesh(array_E.vespa_grd, cmap = 'seismic', vmin = (-vmax*1.05), vmax = (vmax*1.05), rasterized=True)
            axis[2].set_xlabel('Time (s)', fontsize = 20)
                 
            for ax in axis:
                ax.set_ylabel('Slowness (s/km)', fontsize = 20)
                ax.set_ylim(0, len(array_Z.vespa_grd))
                ax.set_xticks(x_ticks)
                ax.set_xticklabels(relative_time, rotation=-45, ha='left', fontsize=16) 
                ax.set_yticks(y_ticks)
                ax.set_yticklabels(y_tick_labels, fontsize=16) 
                ax.tick_params(axis='both', labelsize=16)
                ax.scatter(taup_P_sn, 1, marker = '^', s = 100, color = 'k', label='Modelled Arrivals')
                ax.scatter(taup_pP_sn, 1, marker = '^', s = 100, color = 'k')
                ax.scatter(taup_sP_sn, 1, marker = '^', s = 100, color = 'k')
                ax.scatter(taup_S_sn, 1, marker = '^', s = 100, color = 'k', label='Modelled Arrivals')
                ax.scatter(taup_sS_sn, 1, marker = '^', s = 100, color = 'k')
                ax.set_xlim(xlim_start, xlim_end)
            return pw_vespagrams_ZNE


class Global:
    def __init__(self, event, array_class, array_class2=None):             
        '''Store attributes from Array class that are needed for depth modelling.''' 
        
        self.event = event
        
        def store_attribute(array_class, wanted_attribute):
            store_list = []           
            for array in range(len(array_class)):
                try:
                    attr = getattr(array_class[array], wanted_attribute)
                except:
                    print('No %s attribute' %wanted_attribute)
                    store_list.append(0)
                    continue                                      
                if type(attr) != list:
                    store_list.append(attr)
                else:
                    store_list.extend(np.array([attr]))
                #print(store_list)
            return store_list
            
        wanted_attributes = ['array_no', 'ev_array_gcarc','array_latitude','array_longitude','array_baz','array_elevation',
        'beampack_backazimuth','beampack_slowness','phase_id_picks','phase_id_pick_amplitudes',
        'P_onset_rel_time',       'pP_onset_rel_time','sP_onset_rel_time','P_onset_abs_time','pP_onset_abs_time',
        'sP_onset_abs_time','dt_pP_P','dt_sP_P','epicentral_dist_pP','epicentral_dist_sP', 'array_statistics']
        
        #for array in range(len(array_class)):
        #    attributes = array_class[array].get_class_variables()   
        
        self.Z_array_no = store_attribute(array_class, wanted_attributes[0])
        self.Z_ev_array_gcarc = store_attribute(array_class, wanted_attributes[1])
        self.Z_array_latitude = store_attribute(array_class, wanted_attributes[2])
        self.Z_array_longitude = store_attribute(array_class, wanted_attributes[3])
        self.Z_array_baz = store_attribute(array_class, wanted_attributes[4])
        self.Z_array_elevation = store_attribute(array_class, wanted_attributes[5])
        self.Z_beampack_backazimuth = store_attribute(array_class, wanted_attributes[6])
        self.Z_beampack_slowness = store_attribute(array_class, wanted_attributes[7])
        self.Z_phase_id_picks = store_attribute(array_class, wanted_attributes[8])
        self.Z_phase_id_pick_amplitudes = store_attribute(array_class, wanted_attributes[9])
        self.dt_pP_P = store_attribute(array_class, wanted_attributes[16])
        self.dt_sP_P  = store_attribute(array_class, wanted_attributes[17])
        self.epicentral_dist_pP = store_attribute(array_class, wanted_attributes[18])
        self.epicentral_dist_sP = store_attribute(array_class, wanted_attributes[19])
        try:
            self.array_statistics_Z = array_class[0].array_statistics
        except:
            self.array_statistics_Z = [0,0,0]
        streams = store_attribute(array_class, 'phase_weighted_beams')
        slw_indexes = store_attribute(array_class, 'slowness_index')
        
        # convert relative peak times into absolute
        utc_time = []
        for i in range (len(array_class)):
            utc_time.append(streams[i][slw_indexes[i]].times('utcdatetime')) 
        
        peaks = []
        for i in range (len(self.Z_phase_id_picks)):
            array_peaks = self.Z_phase_id_picks[i]
            
            if array_peaks[0] == 0:
                P = 0
            else:
                P = utc_time[i][array_peaks[0]]
            
            if array_peaks[1] == 0:
                pP = 0
            else:
                pP = utc_time[i][array_peaks[1]]
            
            if array_peaks[2] == 0:
                sP = 0
            else:
                sP = utc_time[i][array_peaks[2]]   
            peaks.append([P,pP,sP])
        self.Z_phase_id_picks_abs = peaks
        
        if isinstance(array_class2,np.ndarray): 

            wanted_attributes = ['array_no', 'ev_array_gcarc','array_latitude','array_longitude','array_baz','array_elevation',
        'beampack_backazimuth','beampack_slowness','phase_id_picks','phase_id_pick_amplitudes',
        'S_onset_rel_time','sS_onset_rel_time','S_onset_abs_time','sS_onset_abs_time',
        'dt_sS_S','epicentral_dist_sS','array_statistics']
            self.T_array_no = store_attribute(array_class2, wanted_attributes[0])
            self.T_ev_array_gcarc = store_attribute(array_class2, wanted_attributes[1])
            self.T_array_latitude = store_attribute(array_class2, wanted_attributes[2])
            self.T_array_longitude = store_attribute(array_class2, wanted_attributes[3])
            self.T_array_baz = store_attribute(array_class2, wanted_attributes[4])
            self.T_array_elevation = store_attribute(array_class2, wanted_attributes[5])
            self.T_beampack_backazimuth = store_attribute(array_class2, wanted_attributes[6])
            self.T_beampack_slowness = store_attribute(array_class2, wanted_attributes[7])
            self.T_phase_id_picks = store_attribute(array_class2, wanted_attributes[8])
            self.T_phase_id_pick_amplitudes = store_attribute(array_class2, wanted_attributes[9])
            self.dt_sS_S = store_attribute(array_class2, wanted_attributes[14])
            self.epicentral_dist_sS = store_attribute(array_class2, wanted_attributes[15]) 
            try:
                self.array_statistics_T = array_class2[0].array_statistics
            except:
                self.array_statistics_T = [0,0,0]        

            # convert relative peak picks to absolute time
            streams = store_attribute(array_class2, 'phase_weighted_beams')
            slw_indexes = store_attribute(array_class2, 'slowness_index')

            utc_time = []
            for i in range (len(array_class2)):
                utc_time.append(streams[i][slw_indexes[i]].times('utcdatetime')) 
            
            peaks = []
            for i in range (len(self.T_phase_id_picks)):
                array_peaks = self.T_phase_id_picks[i]
                if array_peaks[0] == 0:
                    S = 0
                else:
                    S = utc_time[i][array_peaks[0]]
                
                if array_peaks[1] == 0:
                    sS = 0
                else:
                    sS = utc_time[i][array_peaks[1]]
                peaks.append([S,sS])
            self.T_phase_id_picks_abs = peaks
        return

    def remove_empty_arrays(self, component):
        '''Remove arrays without any picks, that made it through QC during array processing loop.'''
        
        # Define filter for Z component arrays
        Z_filter = np.ones(len(self.Z_array_no))
        for i in range (len(self.Z_array_no)):
            if np.sum(self.Z_phase_id_picks[i]) == 0:
                Z_filter[i] = 0
                
        # Apply filter
        self.Z_array_no = list(np.asarray(self.Z_array_no)[Z_filter==1])
        self.Z_ev_array_gcarc = list(np.asarray(self.Z_ev_array_gcarc)[Z_filter==1])
        self.Z_array_latitude = list(np.asarray(self.Z_array_latitude)[Z_filter==1])
        self.Z_array_longitude = list(np.asarray(self.Z_array_longitude)[Z_filter==1])
        self.Z_array_baz = list(np.asarray(self.Z_array_baz)[Z_filter==1])
        self.Z_array_elevation = list(np.asarray(self.Z_array_elevation)[Z_filter==1])
        self.Z_beampack_backazimuth = list(np.asarray(self.Z_beampack_backazimuth)[Z_filter==1])
        self.Z_beampack_slowness = list(np.asarray(self.Z_beampack_slowness)[Z_filter==1])
        self.Z_phase_id_picks = list(np.asarray(self.Z_phase_id_picks)[Z_filter==1])
        self.Z_phase_id_pick_amplitudes = list(np.asarray(self.Z_phase_id_pick_amplitudes)[Z_filter==1])
        self.Z_phase_id_picks_abs = list(np.asarray(self.Z_phase_id_picks_abs)[Z_filter==1])
        self.dt_pP_P = list(np.asarray(self.dt_pP_P)[Z_filter==1])
        self.dt_sP_P = list(np.asarray(self.dt_sP_P)[Z_filter==1])
        self.epicentral_dist_pP = list(np.asarray(self.epicentral_dist_pP)[Z_filter==1])
        self.epicentral_dist_sP = list(np.asarray(self.epicentral_dist_sP)[Z_filter==1])
                    
        if component == 'ZNE':
            T_filter = np.ones(len(self.T_array_no))
            for i in range (len(self.T_array_no)):
                if np.sum(self.T_phase_id_picks[i]) == 0:
                    T_filter[i] = 0
                    
            # Apply filter
            self.T_array_no = list(np.asarray(self.T_array_no)[T_filter==1])
            self.T_ev_array_gcarc = list(np.asarray(self.T_ev_array_gcarc)[T_filter==1])
            self.T_array_latitude = list(np.asarray(self.T_array_latitude)[T_filter==1])
            self.T_array_longitude = list(np.asarray(self.T_array_longitude)[T_filter==1])
            self.T_array_baz = list(np.asarray(self.T_array_baz)[T_filter==1])
            self.T_array_elevation = list(np.asarray(self.T_array_elevation)[T_filter==1])
            self.T_beampack_backazimuth = list(np.asarray(self.T_beampack_backazimuth)[T_filter==1])
            self.T_beampack_slowness = list(np.asarray(self.T_beampack_slowness)[T_filter==1])
            self.T_phase_id_picks = list(np.asarray(self.T_phase_id_picks)[T_filter==1])
            self.T_phase_id_pick_amplitudes = list(np.asarray(self.T_phase_id_pick_amplitudes)[T_filter==1])
            self.T_phase_id_picks_abs = list(np.asarray(self.T_phase_id_picks_abs)[T_filter==1])
            self.dt_sS_S = list(np.asarray(self.dt_sS_S)[T_filter==1])
            self.epicentral_dist_sS = list(np.asarray(self.epicentral_dist_sS)[T_filter==1])
            return        
        
    
    def create_metadata_dataframe(self, component):
        '''Create a pandas dataframe to compare Z and T component data.'''
        
        data_Z = { 
            'Z_arr_no': self.Z_array_no, 
            'Z_gcarc': self.Z_ev_array_gcarc,
            'Z_lat': self.Z_array_latitude,
            'Z_lon': self.Z_array_longitude,
            'Z_baz': self.Z_array_baz,
            'Z_elev': self.Z_array_elevation,
            'Z_bp_baz': self.Z_beampack_backazimuth,
            'Z_bp_slw': self.Z_beampack_slowness,
            'Z_picks': self.Z_phase_id_picks,
            'Z_picks_abs': self.Z_phase_id_picks_abs,
            'Z_pick_amps': self.Z_phase_id_pick_amplitudes,
            'Z_epi_dist_pP': self.epicentral_dist_pP,
            'Z_epi_dist_sP': self.epicentral_dist_sP}
        
        df_Z=pd.DataFrame(data_Z)   
        with pd.option_context('display.max_rows',None,'display.max_columns',None): 
            print(df_Z)
            
        if component == 'ZNE':   
        
            #print(self.T_array_no, self.T_ev_array_gcarc,self.T_array_latitude,self.T_array_longitude,self.T_array_baz,self.T_array_elevation,self.T_beampack_backazimuth,self.T_beampack_slowness,self.T_phase_id_picks,self.T_phase_id_pick_amplitudes,self.epicentral_dist_sS)
        
            data_T = { 
                'T_arr_no': self.T_array_no,  
                'T_gcarc': self.T_ev_array_gcarc,
                'T_lat': self.T_array_latitude,
                'T_lon': self.T_array_longitude,
                'T_baz': self.T_array_baz,
                'T_elev': self.T_array_elevation,
                'T_bp_baz': self.T_beampack_backazimuth,
                'T_bp_slw': self.T_beampack_slowness,
                'T_picks': self.T_phase_id_picks,
                'T_picks_abs': self.T_phase_id_picks_abs,
                'T_pick_amps': self.T_phase_id_pick_amplitudes,
                'T_epi_dist_sS': self.epicentral_dist_sS}
        
            df_T=pd.DataFrame(data_T)
            with pd.option_context('display.max_rows',None,'display.max_columns',None):
                print(df_T)
        return

    def forward_model_depth(self, vel_model, phases):
        '''Returns best fit depth for the provided differential time between pP-P, using forward modelling (residual plot optional).'''
        
        # Set up empty residual arrays
        residuals = []
        depths = []
        
        # Define inputs
        no_arrays = len(self.Z_array_latitude)
        
        if phases == ['P','pP']:
            gcarc = self.Z_ev_array_gcarc
            dt_peaks = self.dt_pP_P 
            
        if phases == ['P','sP']:
            gcarc = self.Z_ev_array_gcarc
            dt_peaks = self.dt_sP_P 
            
        if phases == ['S','sS']:
            gcarc = self.T_ev_array_gcarc
            dt_peaks = self.dt_sS_S        
        
        # Calculate test depths
        test_depths = np.arange((self.event.evdp-40), (self.event.evdp+40), 0.1) # ~0.1km intervals
        test_depths[test_depths < 0] = np.nan
        test_depths = test_depths[~np.isnan(test_depths)]
            
        # Determine depth for each array
        for i in range (no_arrays):
            depth = []
            residual = []
            
            # Loop through test depths straddling initial catalogue depth by +/- 40 km
            for j in range(len(test_depths)):
                try:
                   arrivals = vel_model.get_travel_times(source_depth_in_km=test_depths[j], distance_in_degree=gcarc[i],phase_list=phases)

                   if len(arrivals) < 2:  
                       residual.append(np.nan)
                       
                   if len(arrivals) == 2:
                       peak1 = arrivals[0].time
                       peak2 = arrivals[1].time
                       dt = peak2 - peak1
                       residual.append(abs(dt - dt_peaks[i]))
                       
                   if len(arrivals) > 2:
                        for k in range (len(arrivals)):
                            if arrivals[k].name == phases[0]:
                                peak1 = arrivals[k].time
                                break
                                
                        for l in range (len(arrivals)):
                            if arrivals[l].name == phases[1]:
                                peak2 = arrivals[l].time
                                break
                        dt = peak2 - peak1
                        residual.append(abs(dt - dt_peaks[i]))
                               
                except:
                    residual.append(np.nan)

            residual = np.array(residual)
            
            # Find test depth with smallest residual between observed and modelled dt
            try:
                index = np.nanargmin(residual)
                depth = test_depths[index]
            except ValueError:
                continue
            residuals.append(residual)
            depths.append(depth)
            
        if phases == ['P','pP']:
            self.depths_pP = depths
            self.depth_residuals_pP = residuals
            
        if phases == ['P','sP']:
            self.depths_sP = depths
            self.depth_residuals_sP = residuals
            
        if phases == ['S','sS']:
            self.depths_sS = depths
            self.depth_residuals_sS = residuals            
        return

    def find_cleaning_filter(self, phases):
        '''Remove anomalous depths using standard deviation of 1.3.'''
        
        if phases == ['P','pP','sP']:
            # Use pP and sP depths together
            depths = np.append(self.depths_pP, self.depths_sP)
            epicentral_dists = np.append(self.epicentral_dist_pP, self.epicentral_dist_sP)
            
            # Find median depth
            median = np.median(depths)
            
            # Find 1.3 standard deviation filter
            pP_diff = abs(self.depths_pP - median)
            sP_diff = abs(self.depths_sP - median)
            self.pP_std_filter = pP_diff < np.std(depths) * 1.3
            self.sP_std_filter = sP_diff < np.std(depths) * 1.3
            
            for i in range (len(self.dt_pP_P)):
                if self.dt_pP_P[i] == 0:
                    self.pP_std_filter[i] = 0
                    
            for i in range (len(self.dt_sP_P)):
                if self.dt_sP_P[i] == 0:
                    self.sP_std_filter[i] = 0
            
            print('FILTER', self.pP_std_filter, self.sP_std_filter)

            '''self.Z_ev_array_gcarc = np.array(self.Z_ev_array_gcarc)[std_filter]
            self.Z_array_latitude = np.array(self.Z_array_latitude)[std_filter]
            self.Z_array_longitude = np.array(self.Z_array_longitude)[std_filter]
            self.Z_array_baz = np.array(self.Z_array_baz)[std_filter]
            self.Z_array_elevation = np.array(self.Z_array_elevation)[std_filter]
            self.Z_beampack_backazimuth = np.array(self.Z_beampack_backazimuth)[std_filter]
            self.Z_beampack_slowness = np.array(self.Z_beampack_slowness)[std_filter]
            self.Z_phase_id_picks = np.array(self.Z_phase_id_picks)[std_filter]
            self.Z_phase_id_pick_amplitudes = np.array(self.Z_phase_id_pick_amplitudes)[std_filter]
            self.P_onset_rel_time = np.array(self.P_onset_rel_time)[std_filter]
            self.pP_onset_rel_time = np.array(self.pP_onset_rel_time)[std_filter]
            self.sP_onset_rel_time = np.array(self.sP_onset_rel_time)[std_filter]
            self.P_onset_abs_time = np.array(self.P_onset_abs_time)[std_filter]
            self.pP_onset_abs_time = np.array(self.pP_onset_abs_time)[std_filter]
            self.sP_onset_abs_time = np.array(self.sP_onset_abs_time)[std_filter]
            self.dt_pP_P = np.array(self.dt_pP_P)[std_filter]
            self.dt_sP_P = np.array(self.dt_sP_P)[std_filter]
            self.epicentral_dist_pP = np.array(self.epicentral_dist_pP)[std_filter]
            self.epicentral_dist_sP = np.array(self.epicentral_dist_sP)[std_filter]'''
               
        if phases == ['S','sS']:
        
            # Use pP and sP depths together
            depths = self.depths_sS
            epicentral_dists = self.epicentral_dist_sS
            
            # Find median depth
            median = np.median(depths)
            
            # Find 1.3 standard deviation filter
            median_diff = abs(depths - median)
            self.sS_std_filter = median_diff < np.std(depths) * 1.3
            print('sS FILTER:', self.sS_std_filter)
            
            for i in range (len(self.dt_sS_S)):
                if self.dt_sS_S[i] == 0:
                    self.sS_std_filter[i] = 0
                
            '''self.T_ev_array_gcarc = np.array(self.T_ev_array_gcarc)[sS_std_filter]
            self.T_array_latitude = np.array(self.T_array_latitude)[sS_std_filter]
            self.T_array_longitude = np.array(self.T_array_longitude)[sS_std_filter]
            self.T_array_baz = np.array(self.T_array_baz)[sS_std_filter]
            self.T_array_elevation = np.array(self.T_array_elevation)[sS_std_filter]
            self.T_beampack_backazimuth = np.array(self.T_beampack_backazimuth)[sS_std_filter]
            self.T_beampack_slowness = np.array(self.T_beampack_slowness)[sS_std_filter]
            self.T_phase_id_picks = np.array(self.T_phase_id_picks)[sS_std_filter]
            self.T_phase_id_pick_amplitudes = np.array(self.T_phase_id_pick_amplitudes)[sS_std_filter]
            self.S_onset_rel_time = np.array(self.S_onset_rel_time)[sS_std_filter]
            self.sS_onset_rel_time = np.array(self.sS_onset_rel_time)[sS_std_filter]
            self.S_onset_abs_time = np.array(self.S_onset_abs_time)[sS_std_filter]
            self.sS_onset_abs_time = np.array(self.sS_onset_abs_time)[sS_std_filter]
            self.dt_sS_S = np.array(self.dt_sS_S)[sS_std_filter]
            self.epicentral_dist_sS = np.array(self.epicentral_dist_sS)[sS_std_filter]'''
        return


    def forward_depth_modelling_P_coda(self):
        '''Returns best fit depth for the provided differential time between pP-P and sP-P, using forward modelling.'''
            
        self.P_coda_depths = []
        
        test_depths = np.arange((self.event.evdp-40), (self.event.evdp+40), 0.1) # ~0.1km intervals
        test_depths[test_depths < 0] = np.nan
        test_depths = test_depths[~np.isnan(test_depths)]
            
        dist_pP = np.array(self.epicentral_dist_pP)[self.pP_std_filter]
        dist_sP = np.array(self.epicentral_dist_sP)[self.sP_std_filter]
        
        residual_pP = np.array(self.depth_residuals_pP)[self.pP_std_filter]
        residual_sP = np.array(self.depth_residuals_sP)[self.sP_std_filter]
        
        if len(residual_pP) == 0 & len(residual_sP) == 0:
            print('No pP or sP values for combined depth')
            return
        
        if len(residual_pP) == 0:
            print('No pP values for combined depth')
            dist = dist_sP
            res = residual_sP
        
        if len(residual_sP) == 0:
            print('No sP values for combined depth')
            dist = dist_pP
            res= residual_pP
        
        else:  
            print('Using both pP and sP values for combined depth')
            dist = np.concatenate((dist_pP, dist_sP), axis=0)
            res = np.concatenate((residual_pP, residual_sP), axis=0)                
 
        if len(dist) > 1: 
            sort = np.argsort(dist)
            dist = dist[sort]
            res = res[sort]

            distance = []
            residual = []
            unique_dist = np.unique(dist)
            
            for d in unique_dist:
                indices = np.where(dist == d)
                res_selected = np.array(res)[indices]
                total = 0
                for r in res_selected:
                    total += res_selected[0]/len(indices) # dived by number of phases used to find residual to prevent bias towards pP and sP residual arrays
                distance.append(d)
                residual.append(total)
                     
            for i in range (len(residual)):
                index = np.nanargmin(residual[i])
                depth = test_depths[index]
                print('depth', depth)
                self.P_coda_depths.append(depth)
        else:
            index = np.nanargmin(res)
            depth = test_depths[index]
            print('depth', depth)
            self.P_coda_depths.append(depth)
                    
        print('P_coda_depths: ', self.P_coda_depths)
        return


    def forward_depth_modelling_P_S_coda(self):
        '''Returns best fit depth for the provided differential time between pP-P and sP-P, using forward modelling.'''
            
        self.P_S_coda_depths = []
        
        test_depths = np.arange((self.event.evdp-40), (self.event.evdp+40), 0.1) # ~0.1km intervals
        test_depths[test_depths < 0] = np.nan
        test_depths = test_depths[~np.isnan(test_depths)]
        
        # Clean data point of anomalies
        dist_pP = np.array(self.epicentral_dist_pP)[self.pP_std_filter]
        dist_sP = np.array(self.epicentral_dist_sP)[self.sP_std_filter]
        dist_sS = np.array(self.epicentral_dist_sS)[self.sS_std_filter]
        
        residual_pP = np.array(self.depth_residuals_pP)[self.pP_std_filter]
        residual_sP = np.array(self.depth_residuals_sP)[self.sP_std_filter]
        residual_sS = np.array(self.depth_residuals_sS)[self.sS_std_filter]
        
        print(len(dist_pP), len(residual_pP), len(residual_sS), len(test_depths))
        
        if len(residual_pP) == 0 & len(residual_sP) == 0 & len(residual_sS) == 0:
            print('No pP or sP or sS values for combined depth')
            return
        
        if len(residual_pP) != 0 or len(residual_sP) != 0 or len(residual_sS) != 0:  
            print('Using both pP and sP values for combined depth')
            dist = np.concatenate((np.concatenate((dist_pP, dist_sP), axis=0), dist_sS), axis=0)
            res = np.concatenate((np.concatenate((residual_pP, residual_sP), axis=0), residual_sS), axis=0)                
           
        if len(dist) > 1: 
            sort = np.argsort(dist)
            dist = dist[sort]
            res = res[sort]

            distance = []
            residual = []
            unique_dist = np.unique(dist)
            
            for d in unique_dist:
                indices = np.where(dist == d)
                res_selected = np.array(res)[indices]
                total = 0
                for r in res_selected:
                    total += res_selected[0]/len(indices)
                distance.append(d)
                residual.append(total)
                     
            for i in range (len(residual)):
                index = np.nanargmin(residual[i])
                depth = test_depths[index]
                print('depth', depth)
                self.P_S_coda_depths.append(depth)
        else:
            index = np.nanargmin(res)
            depth = test_depths[index]
            print('depth', depth)
            self.P_S_coda_depths.append(depth)
                    
        print('P_&_S_coda_depths: ', self.P_S_coda_depths)
        return

    def statistics(self, f):
            '''Returns key statistics for the modelled depths (median, mean, standard deviation) for pP-P and sP-P times, returns histogram.'''
            
            depths_pP = self.depths_pP
            depths_sP = self.depths_sP
            all_depths = self.P_coda_depths
            self.median_depth_pP = 0
            self.mean_pP = 0
            self.mad_pP = 0
            self.sd_pP = 0
            self.median_depth_sP = 0
            self.mean_sP = 0
            self.mad_sP = 0
            self.sd_sP = 0
            self.all_median_depth = 0
            self.all_mean = 0
            self.all_mad = 0
            self.all_sd = 0      

            if len(depths_pP) > 1:
                    self.median_depth_pP = np.median(depths_pP)
                    self.mean_pP = np.mean(depths_pP)
                    self.mad_pP = median_abs_deviation(depths_pP)
                    self.sd_pP = statistics.stdev(depths_pP)
                    print()
                    print('Median Depth from pP-P (km):', self.median_depth_pP)
                    print('Mean Depth from pP-P (km):', self.mean_pP)
                    print('MAD from pP-P (km):', self.mad_pP)
                    print('Standard Deviation from pP-P (km):', self.sd_pP)
                    print()
                    f.write('Median Depth from pP-P (km):' + str(self.median_depth_pP) + '\n')
                    f.write('Mean Depth from pP-P (km):' + str(self.mean_pP) + '\n')
                    f.write('MAD from pP-P (km):' + str(self.mad_pP) + '\n')
                    f.write('Standard Deviation from pP-P (km):' + str(self.sd_pP) + '\n')
                    f.write('\n')
            
            if len(depths_pP) == 1:
                    self.median_depth_pP = depths_pP[0]
                    self.mean_pP = depths_pP[0]
                    self.mad_pP = 0
                    self.sd_pP = 0
                    print()
                    print('Median Depth from pP-P (km):', self.median_depth_pP)
                    print('Mean Depth from pP-P (km):', self.mean_pP)
                    print('MAD from pP-P (km):', self.mad_pP)
                    print('Standard Deviation from pP-P (km):', self.sd_pP)
                    print()
                    f.write('Median Depth from pP-P (km):' + str(self.median_depth_pP) + '\n')
                    f.write('Mean Depth from pP-P (km):' + str(self.mean_pP) + '\n')
                    f.write('MAD from pP-P (km):' + str(self.mad_pP) + '\n')
                    f.write('Standard Deviation from pP-P (km):' + str(self.sd_pP) + '\n')
                    f.write('\n')            
             
            if len(depths_sP) > 1:
                    self.median_depth_sP = np.median(depths_sP)
                    self.mean_sP = np.mean(depths_sP)
                    self.mad_pP = median_abs_deviation(depths_sP)
                    self.sd_sP = statistics.stdev(depths_sP)
                    print('Median Depth from sP-P (km):', self.median_depth_sP)
                    print('Mean Depth from sP-P (km):', self.mean_sP)
                    print('MAD from sP-P (km):', self.mad_sP)
                    print('Standard Deviation from sP-P (km):', self.sd_sP)
                    print()
                    f.write('Median Depth from sP-P (km):' + str(self.median_depth_sP) + '\n')
                    f.write('Mean Depth from sP-P (km):' + str(self.mean_sP) + '\n')
                    f.write('MAD from sP-P (km):' + str(self.mad_sP) + '\n')
                    f.write('Standard Deviation from sP-P (km):' + str(self.sd_sP) + '\n')
                    f.write('\n')
                    
            if len(depths_sP) == 1:
                    self.median_depth_sP = depths_sP[0]
                    self.mean_sP = depths_sP[0]
                    self.mad_pP = 0
                    self.sd_sP = 0
                    print()
                    print('Median Depth from sP-P (km):', self.median_depth_sP)
                    print('Mean Depth from sP-P (km):', self.mean_sP)
                    print('MAD from sP-P (km):', self.mad_sP)
                    print('Standard Deviation from sP-P (km):', self.sd_sP)
                    print()
                    f.write('Median Depth from sP-P (km):' + str(self.median_depth_sP) + '\n')
                    f.write('MAD from sP-P (km):' + str(self.mad_sP) + '\n')
                    f.write('Mean Depth from sP-P (km):' + str(self.mean_sP) + '\n')
                    f.write('Standard Deviation from sP-P (km):' + str(self.sd_sP) + '\n')
                    f.write('\n')    
            
            '''   
            all_depths = np.append(depths_pP, depths_sP)  # combined depths from pP and sP
            if len(all_depths) > 1:
                    self.all_median_depth = np.median(all_depths)
                    self.all_mean = np.mean(all_depths)
                    self.all_sd = statistics.stdev(all_depths)
                    print('Median Depth from pP-P and sP-P (km):', self.all_median_depth)
                    print('Mean Depth from pP-P and sP-P(km):', self.all_mean)
                    print('Standard Deviation from pP-P and sP-P (km):', self.all_sd)
                    print()
                    f.write('Median Depth from pP-P and sP-P (km):' + str(self.all_median_depth) + '\n')
                    f.write('Mean Depth from pP-P and sP-P (km):' + str(self.all_mean) + '\n')
                    f.write('Standard Deviation from pP-P and sP-P (km):' + str(self.all_sd) + '\n')
                    f.write('\n')
                    
                    # Sort histogram bins
                    sorted_depths = np.sort(all_depths)
                    print('Sorted_depths', sorted_depths)
                    all_bin = np.arange(np.floor(sorted_depths[0]), np.ceil(sorted_depths[-1])+1, 0.3)
                    
                    depth_hist = plt.figure(figsize=(15,10))
                    x = plt.hist(sorted_depths, alpha = 1, bins = all_bin, color = 'red', label = 'all depths', zorder = 0)
                    plt.hist(depths_pP, alpha = 1, bins = all_bin, color = 'blue', label = 'pP depths', zorder= 1)
                    plt.hist(depths_sP, alpha = 1, bins = all_bin, color = 'green', label = 'sP depths', zorder = 2)
                    plt.xlabel('Depth (km)')
                    plt.ylabel('Frequency')
                    plt.title('Depth Histogram')
                    if len(x) < 1:
                        high_freq = x[1][np.argmax(x[0])]
                        plt.text(0.01, 0.97, 'Depth (km): %s' %high_freq, transform=plt.gca().transAxes)
                    plt.legend()
                    plt.close()'''
            
            if len(all_depths) > 1:
                    self.all_median_depth = np.median(all_depths)  # smallest residual depths found using pP and sP
                    self.all_mad = median_abs_deviation(all_depths)
                    self.all_mean = np.mean(all_depths)
                    self.all_sd = statistics.stdev(all_depths)
                    print('Median Depth from pP-P and sP-P (km):', self.all_median_depth)
                    print('Mean Depth from pP-P and sP-P(km):', self.all_mean)
                    print('MAD from pP-P and sP-P (km):', self.all_mad)
                    print('Standard Deviation from pP-P and sP-P (km):', self.all_sd)
                    print()
                    f.write('Median Depth from pP-P and sP-P (km):' + str(self.all_median_depth) + '\n')
                    f.write('Mean Depth from pP-P and sP-P (km):' + str(self.all_mean) + '\n')
                    f.write('MAD from pP-P and sP-P (km):' + str(self.all_mad) + '\n')
                    f.write('Standard Deviation from pP-P and sP-P (km):' + str(self.all_sd) + '\n')
                    f.write('\n')
                    
                    # Sort histogram bins
                    sorted_depths = np.sort(all_depths)
                    print('Sorted_depths', sorted_depths)
                    all_bin = np.arange(np.floor(sorted_depths[0]), np.ceil(sorted_depths[-1])+1, 0.3)
                    
                    depth_hist = plt.figure(figsize=(15,10))
                    x = plt.hist(sorted_depths, alpha = 1, bins = all_bin, color = 'red', label = 'all depths', zorder = 0)
                    plt.hist(depths_pP, alpha = 1, bins = all_bin, color = 'blue', label = 'pP depths', zorder= 1)
                    plt.hist(depths_sP, alpha = 1, bins = all_bin, color = 'green', label = 'sP depths', zorder = 2)
                    plt.xlabel('Depth (km)')
                    plt.ylabel('Frequency')
                    plt.title('Depth Histogram')
                    if len(x) < 1:
                        high_freq = x[1][np.argmax(x[0])]
                        plt.text(0.01, 0.97, 'Depth (km): %s' %high_freq, transform=plt.gca().transAxes)
                    plt.legend()
                    plt.close()     
                    
                    return depth_hist
                    
            if len(all_depths) == 1:
                    self.all_median_depth = all_depths[0]  # smallest residual depths found using pP and sP
                    self.all_mean = all_depths[0]
                    self.all_mad = 0
                    self.all_sd = 0
                    print('Median Depth from pP-P and sP-P (km):', self.all_median_depth)
                    print('Mean Depth from pP-P and sP-P(km):', self.all_mean)
                    print('MAD from pP-P and sP-P (km):', self.all_mad)
                    print('Standard Deviation from pP-P and sP-P (km):', self.all_sd)
                    print()
                    f.write('Median Depth from pP-P and sP-P (km):' + str(self.all_median_depth) + '\n')
                    f.write('Mean Depth from pP-P and sP-P (km):' + str(self.all_mean) + '\n')
                    f.write('MAD from pP-P and sP-P (km):' + str(self.all_mad) + '\n')
                    f.write('Standard Deviation from pP-P and sP-P (km):' + str(self.all_sd) + '\n')
                    f.write('\n')                
                    
            return

    def write_out_final_outputs(self, ev_dir, component):
        '''Return text files containing key outputs from the Global class.'''
        
        outputs_cleaned = os.path.join(ev_dir, 'outputs_cleaned_pP.txt')

        f = open(outputs_cleaned, 'w')
        f.write('array_no' + '\t' + 'array_lat    ' + '\t' + 'array_lon    ' + '\t'+'epicentral_distance' + '\t' + 'backazimuth' + '\t' + 'dt_pP'+ '\t' + 'depth' + '\t' + 'P_abs_onset       ' + '\t' + 'pP_abs_onset      ' + '\t' + 'amplitude_P' + '\t' + 'amplitude_pP' + '\n')
        if sum(self.pP_std_filter) == 1:
            f.write(str(np.array(self.Z_array_no)[self.pP_std_filter]) + '\t')
            f.write(str(np.array(self.Z_array_latitude)[self.pP_std_filter]) + '\t')
            f.write(str(np.array(self.Z_array_longitude)[self.pP_std_filter]) + '\t')
            f.write(str(np.array(self.epicentral_dist_pP)[self.pP_std_filter])+'\t')
            f.write(str(np.array(self.Z_beampack_backazimuth)[self.pP_std_filter])+ '\t')
            f.write(str(np.array(self.dt_pP_P)[self.pP_std_filter])+'\t')
            f.write(str(np.array(self.depths_pP)[self.pP_std_filter]) + '\t')
            f.write(str(np.array(self.Z_phase_id_picks_abs)[self.pP_std_filter][0][0]) + '\t')
            f.write(str(np.array(self.Z_phase_id_picks_abs)[self.pP_std_filter][0][1]) +'\t')
            f.write(str(np.array(self.Z_phase_id_pick_amplitudes)[self.pP_std_filter][0][0]) +'\t')
            f.write(str(np.array(self.Z_phase_id_pick_amplitudes)[self.pP_std_filter][0][1]) +'\n')
        
        elif sum(self.pP_std_filter) > 1:
            for i in range (len(np.array(self.dt_pP_P)[self.pP_std_filter])):
                f.write(str(np.array(self.Z_array_no)[self.pP_std_filter][i]) + '\t')
                f.write(str(np.array(self.Z_array_latitude)[self.pP_std_filter][i]) + '\t')
                f.write(str(np.array(self.Z_array_longitude)[self.pP_std_filter][i]) + '\t')
                f.write(str(np.array(self.epicentral_dist_pP)[self.pP_std_filter][i])+'\t')
                f.write(str(np.array(self.Z_beampack_backazimuth)[self.pP_std_filter][i])+ '\t')
                f.write(str(np.array(self.dt_pP_P)[self.pP_std_filter][i])+'\t')
                f.write(str(np.array(self.depths_pP)[self.pP_std_filter][i]) + '\t')
                f.write(str(np.array(self.Z_phase_id_picks_abs)[self.pP_std_filter][i][0]) + '\t')
                f.write(str(np.array(self.Z_phase_id_picks_abs)[self.pP_std_filter][i][1]) +'\t')
                f.write(str(np.array(self.Z_phase_id_pick_amplitudes)[self.pP_std_filter][i][0]) +'\t')
                f.write(str(np.array(self.Z_phase_id_pick_amplitudes)[self.pP_std_filter][i][1]) +'\n')
        
        else:
            print('No pP results to export.')
            pass
        f.close()
        
        outputs_cleaned = os.path.join(ev_dir, 'outputs_cleaned_sP.txt')

        f = open(outputs_cleaned, 'w')
        f.write('array_no' + '\t' + 'array_lat    ' + '\t' + 'array_lon    ' + '\t'+'epicentral_distance' + '\t' + 'backazimuth' + '\t' + 'dt_sP'+ '\t' + 'depth' + '\t' + 'P_abs_onset       ' + '\t' + 'sP_abs_onset'  + '\t' + 'amplitude_sP' + '\n')
        if sum(self.sP_std_filter) == 1:
            f.write(str(np.array(self.Z_array_no)[self.sP_std_filter]) + '\t')
            f.write(str(np.array(self.Z_array_latitude)[self.sP_std_filter]) + '\t')
            f.write(str(np.array(self.Z_array_longitude)[self.sP_std_filter]) + '\t')
            f.write(str(np.array(self.epicentral_dist_sP)[self.sP_std_filter])+'\t')
            f.write(str(np.array(self.Z_beampack_backazimuth)[self.sP_std_filter])+ '\t')
            f.write(str(np.array(self.dt_sP_P)[self.sP_std_filter])+'\t')
            f.write(str(np.array(self.depths_sP)[self.sP_std_filter]) + '\t')
            f.write(str(np.array(self.Z_phase_id_picks_abs)[self.sP_std_filter][0][0]) + '\t')
            f.write(str(np.array(self.Z_phase_id_picks_abs)[self.sP_std_filter][0][2]) +'\t')
            f.write(str(np.array(self.Z_phase_id_pick_amplitudes)[self.sP_std_filter][0][0]) +'\t')
            f.write(str(np.array(self.Z_phase_id_pick_amplitudes)[self.sP_std_filter][0][2]) +'\n')
        
        elif sum(self.sP_std_filter) > 1:
            for i in range (len(np.array(self.dt_sP_P)[self.sP_std_filter])):
                f.write(str(np.array(self.Z_array_no)[self.sP_std_filter][i]) + '\t')
                f.write(str(np.array(self.Z_array_latitude)[self.sP_std_filter][i]) + '\t')
                f.write(str(np.array(self.Z_array_longitude)[self.sP_std_filter][i]) + '\t')
                f.write(str(np.array(self.epicentral_dist_sP)[self.sP_std_filter][i])+'\t')
                f.write(str(np.array(self.Z_beampack_backazimuth)[self.sP_std_filter][i])+ '\t')
                f.write(str(np.array(self.dt_sP_P)[self.sP_std_filter][i])+'\t')
                f.write(str(np.array(self.depths_sP)[self.sP_std_filter][i]) + '\t')
                f.write(str(np.array(self.Z_phase_id_picks_abs)[self.sP_std_filter][i][0]) + '\t')
                f.write(str(np.array(self.Z_phase_id_picks_abs)[self.sP_std_filter][i][2]) +'\t')
                f.write(str(np.array(self.Z_phase_id_pick_amplitudes)[self.sP_std_filter][i][0]) +'\t')
                f.write(str(np.array(self.Z_phase_id_pick_amplitudes)[self.sP_std_filter][i][2]) +'\n')  
         
        else:
            print('No sP results to export.')
            pass
        f.close()

        outputs_cleaned = os.path.join(ev_dir, 'outputs_cleaned_sS.txt')
 
        if component == 'ZNE':
            f = open(outputs_cleaned, 'w')
            f.write('array_no' + '\t' + 'array_lat    ' + '\t' + 'array_lon    ' + '\t'+'epicentral_distance' + '\t' + 'backazimuth' + '\t' + 'dt_sS'+ '\t' + 'depth' + '\t' + 'S_abs_onset       ' + '\t' + 'sS_abs_onset      ' + '\t' + 'amplitude_S' + '\t' + 'amplitude_sS' + '\n')
            if sum(self.sS_std_filter) == 1:
                f.write(str(np.array(self.T_array_no)[self.sS_std_filter]) + '\t')
                f.write(str(np.array(self.T_array_latitude)[self.sS_std_filter]) + '\t')
                f.write(str(np.array(self.T_array_longitude)[self.sS_std_filter]) + '\t')
                f.write(str(np.array(self.epicentral_dist_sS)[self.sS_std_filter])+'\t')
                f.write(str(np.array(self.T_beampack_backazimuth)[self.sS_std_filter])+ '\t')
                f.write(str(np.array(self.dt_sS_S)[self.sS_std_filter])+'\t')
                f.write(str(np.array(self.depths_sS)[self.sS_std_filter]) + '\t')
                f.write(str(np.array(self.T_phase_id_picks_abs)[self.sS_std_filter][0]) + '\t')
                f.write(str(np.array(self.T_phase_id_picks_abs)[self.sS_std_filter][1]) +'\t')
                f.write(str(np.array(self.T_phase_id_pick_amplitudes)[self.sS_std_filter][0]) +'\t')
                f.write(str(np.array(self.T_phase_id_pick_amplitudes)[self.sS_std_filter][1]) +'\n')
                
            elif sum(self.sS_std_filter) > 1:
                for i in range (len(np.array(self.dt_sS_S)[self.sS_std_filter])):
                    f.write(str(np.array(self.T_array_no)[self.sS_std_filter][i]) + '\t')
                    f.write(str(np.array(self.T_array_latitude)[self.sS_std_filter][i]) + '\t')
                    f.write(str(np.array(self.T_array_longitude)[self.sS_std_filter][i]) + '\t')
                    f.write(str(np.array(self.epicentral_dist_sS)[self.sS_std_filter][i])+'\t')
                    f.write(str(np.array(self.T_beampack_backazimuth)[self.sS_std_filter][i])+ '\t')
                    f.write(str(np.array(self.dt_sS_S)[self.sS_std_filter][i])+'\t')
                    f.write(str(np.array(self.depths_sS)[self.sS_std_filter][i]) + '\t')
                    f.write(str(np.array(self.T_phase_id_picks_abs)[self.sS_std_filter][i][0]) + '\t')
                    f.write(str(np.array(self.T_phase_id_picks_abs)[self.sS_std_filter][i][1]) +'\t')
                    f.write(str(np.array(self.T_phase_id_pick_amplitudes)[self.sS_std_filter][i][0]) +'\t')
                    f.write(str(np.array(self.T_phase_id_pick_amplitudes)[self.sS_std_filter][i][1]) +'\n')
                
            else:
                print('No sS results to export.')
                pass
            f.close()
   
        # Write out a phases only list
        Phase_list = os.path.join(ev_dir, 'Phase_list.txt')    
        f = open(Phase_list, 'w')
        f.write('Array' + '\t' + 'Lat' + '\t' +'Lon' + '\t' + 'Elev' + '\t' + 'Dist' + '\t' + 'Baz' + '\t' + 'Phase' + '\t' + 'Arrival_Time' '\t' + 'Env_Amplitude' + '\t' + '\n')
        
        if sum(self.pP_std_filter) == 1:
            #if np.array(self.P_onset_abs_time)[self.pP_std_filter] != 0:                
            f.write(str(np.array(self.Z_array_no)[self.pP_std_filter]) + '\t')
            f.write(str(np.array(self.Z_array_latitude)[self.pP_std_filter]) + '\t')
            f.write(str(np.array(self.Z_array_longitude)[self.pP_std_filter]) + '\t')
            f.write(str(np.array(self.Z_array_elevation)[self.pP_std_filter]) + '\t')
            f.write(str(np.array(self.epicentral_dist_pP)[self.pP_std_filter])+'\t')
            f.write(str(np.array(self.Z_beampack_backazimuth)[self.pP_std_filter])+ '\t')
            f.write('P' + '\t')
            f.write(str(np.array(self.Z_phase_id_picks_abs)[self.pP_std_filter][0][0]) + '\t')
            f.write(str(np.array(self.Z_phase_id_pick_amplitudes)[self.pP_std_filter][0][0]) +'\n')        
            
            #if np.array(self.pP_onset_abs_time)[self.pP_std_filter] != 0:
            f.write(str(np.array(self.Z_array_no)[self.pP_std_filter]) + '\t')
            f.write(str(np.array(self.Z_array_latitude)[self.pP_std_filter]) + '\t')
            f.write(str(np.array(self.Z_array_longitude)[self.pP_std_filter]) + '\t')
            f.write(str(np.array(self.Z_array_elevation)[self.pP_std_filter]) + '\t')
            f.write(str(np.array(self.epicentral_dist_pP)[self.pP_std_filter])+'\t')
            f.write(str(np.array(self.Z_beampack_backazimuth)[self.pP_std_filter])+ '\t')
            f.write('pP' + '\t')
            f.write(str(np.array(self.Z_phase_id_picks_abs)[self.pP_std_filter][0][1]) + '\t')
            f.write(str(np.array(self.Z_phase_id_pick_amplitudes)[self.pP_std_filter][0][1]) +'\n')
        if sum(self.pP_std_filter) > 1:
            for i in range (len(np.array(self.Z_array_no)[self.pP_std_filter])):
                #if np.array(self.P_onset_abs_time)[self.pP_std_filter][i] != 0:                
                f.write(str(np.array(self.Z_array_no)[self.pP_std_filter][i]) + '\t')
                f.write(str(np.array(self.Z_array_latitude)[self.pP_std_filter][i]) + '\t')
                f.write(str(np.array(self.Z_array_longitude)[self.pP_std_filter][i]) + '\t')
                f.write(str(np.array(self.Z_array_elevation)[self.pP_std_filter][i]) + '\t')
                f.write(str(np.array(self.epicentral_dist_pP)[self.pP_std_filter][i])+'\t')
                f.write(str(np.array(self.Z_beampack_backazimuth)[self.pP_std_filter][i])+ '\t')
                f.write('P' + '\t')
                f.write(str(np.array(self.Z_phase_id_picks_abs)[self.pP_std_filter][i][0]) + '\t')
                f.write(str(np.array(self.Z_phase_id_pick_amplitudes)[self.pP_std_filter][i][0]) +'\n')        
                    
                #if np.array(self.pP_onset_abs_time)[self.pP_std_filter][i] != 0:
                f.write(str(np.array(self.Z_array_no)[self.pP_std_filter][i]) + '\t')
                f.write(str(np.array(self.Z_array_latitude)[self.pP_std_filter][i]) + '\t')
                f.write(str(np.array(self.Z_array_longitude)[self.pP_std_filter][i]) + '\t')
                f.write(str(np.array(self.Z_array_elevation)[self.pP_std_filter][i]) + '\t')
                f.write(str(np.array(self.epicentral_dist_pP)[self.pP_std_filter][i])+'\t')
                f.write(str(np.array(self.Z_beampack_backazimuth)[self.pP_std_filter][i])+ '\t')
                f.write('pP' + '\t')
                f.write(str(np.array(self.Z_phase_id_picks_abs)[self.pP_std_filter][i][1]) + '\t')
                f.write(str(np.array(self.Z_phase_id_pick_amplitudes)[self.pP_std_filter][i][1]) +'\n')

        if sum(self.sP_std_filter) == 1:
            #if np.array(self.P_onset_abs_time)[self.sP_std_filter] != 0 and (np.array(self.P_onset_abs_time)[self.sP_std_filter] not in np.array(self.P_onset_abs_time)[self.pP_std_filter]):                
            f.write(str(np.array(self.Z_array_no)[self.sP_std_filter]) + '\t')
            f.write(str(np.array(self.Z_array_latitude)[self.sP_std_filter]) + '\t')
            f.write(str(np.array(self.Z_array_longitude)[self.sP_std_filter]) + '\t')
            f.write(str(np.array(self.Z_array_elevation)[self.sP_std_filter]) + '\t')
            f.write(str(np.array(self.epicentral_dist_pP)[self.sP_std_filter])+'\t')
            f.write(str(np.array(self.Z_beampack_backazimuth)[self.sP_std_filter])+ '\t')
            f.write('P' + '\t')
            f.write(str(np.array(self.Z_phase_id_picks_abs)[self.sP_std_filter][0][0]) + '\t')
            f.write(str(np.array(self.Z_phase_id_pick_amplitudes)[self.sP_std_filter][0][0]) + '\n')       
            #if np.array(self.sP_onset_abs_time)[self.sP_std_filter] != 0:
            f.write(str(np.array(self.Z_array_no)[self.sP_std_filter]) + '\t')
            f.write(str(np.array(self.Z_array_latitude)[self.sP_std_filter]) + '\t')
            f.write(str(np.array(self.Z_array_longitude)[self.sP_std_filter]) + '\t')
            f.write(str(np.array(self.Z_array_elevation)[self.sP_std_filter]) + '\t')
            f.write(str(np.array(self.epicentral_dist_pP)[self.sP_std_filter])+'\t')
            f.write(str(np.array(self.Z_beampack_backazimuth)[self.sP_std_filter])+ '\t')
            f.write('sP' + '\t')
            f.write(str(np.array(self.Z_phase_id_picks_abs)[self.sP_std_filter][0][2]) + '\t')
            f.write(str(np.array(self.Z_phase_id_pick_amplitudes)[self.sP_std_filter][0][2]) +'\n')

        if sum(self.sP_std_filter) > 1:   
            for i in range (len(np.array(self.Z_array_no)[self.sP_std_filter])):
                #if np.array(self.P_onset_abs_time)[self.sP_std_filter][i] != 0 and (np.array(self.P_onset_abs_time)[self.sP_std_filter][i] not in np.array(self.P_onset_abs_time)[self.pP_std_filter]):    
                f.write(str(np.array(self.Z_array_no)[self.sP_std_filter][i]) + '\t')
                f.write(str(np.array(self.Z_array_latitude)[self.sP_std_filter][i]) + '\t')
                f.write(str(np.array(self.Z_array_longitude)[self.sP_std_filter][i]) + '\t')
                f.write(str(np.array(self.Z_array_elevation)[self.sP_std_filter][i]) + '\t')
                f.write(str(np.array(self.epicentral_dist_pP)[self.sP_std_filter][i])+'\t')
                f.write(str(np.array(self.Z_beampack_backazimuth)[self.sP_std_filter][i])+ '\t')
                f.write('P' + '\t')
                f.write(str(np.array(self.Z_phase_id_picks_abs)[self.sP_std_filter][i][0]) + '\t')
                f.write(str(np.array(self.Z_phase_id_pick_amplitudes)[self.sP_std_filter][i][0]) + '\n')       
                #if np.array(self.sP_onset_abs_time)[self.sP_std_filter][i] != 0:
                f.write(str(np.array(self.Z_array_no)[self.sP_std_filter][i]) + '\t')
                f.write(str(np.array(self.Z_array_latitude)[self.sP_std_filter][i]) + '\t')
                f.write(str(np.array(self.Z_array_longitude)[self.sP_std_filter][i]) + '\t')
                f.write(str(np.array(self.Z_array_elevation)[self.sP_std_filter][i]) + '\t')
                f.write(str(np.array(self.epicentral_dist_pP)[self.sP_std_filter][i])+'\t')
                f.write(str(np.array(self.Z_beampack_backazimuth)[self.sP_std_filter][i])+ '\t')
                f.write('sP' + '\t')
                f.write(str(np.array(self.Z_phase_id_picks_abs)[self.sP_std_filter][i][2]) + '\t')
                f.write(str(np.array(self.Z_phase_id_pick_amplitudes)[self.sP_std_filter][i][2]) +'\n')

        if component == 'ZNE':
            if sum(self.sS_std_filter) == 1: 
                #if np.array(self.S_onset_abs_time)[self.sS_std_filter] != 0:                
                f.write(str(np.array(self.T_array_no)[self.sS_std_filter]) + '\t')
                f.write(str(np.array(self.T_array_latitude)[self.sS_std_filter]) + '\t')
                f.write(str(np.array(self.T_array_longitude)[self.sS_std_filter]) + '\t')
                f.write(str(np.array(self.T_array_elevation)[self.sS_std_filter]) + '\t')
                f.write(str(np.array(self.epicentral_dist_sS)[self.sS_std_filter])+'\t')
                f.write(str(np.array(self.T_beampack_backazimuth)[self.sS_std_filter])+ '\t')
                f.write('S' + '\t')
                f.write(str(np.array(self.T_phase_id_picks_abs)[self.sS_std_filter][0]) + '\t')
                f.write(str(np.array(self.T_phase_id_pick_amplitudes)[self.sS_std_filter][0]) +'\n')        
                    
                #if np.array(self.sS_onset_abs_time)[self.sS_std_filter] != 0:
                f.write(str(np.array(self.T_array_no)[self.sS_std_filter]) + '\t')
                f.write(str(np.array(self.T_array_latitude)[self.sS_std_filter]) + '\t')
                f.write(str(np.array(self.T_array_longitude)[self.sS_std_filter]) + '\t')
                f.write(str(np.array(self.T_array_elevation)[self.sS_std_filter]) + '\t')
                f.write(str(np.array(self.epicentral_dist_sS)[self.sS_std_filter])+'\t')
                f.write(str(np.array(self.T_beampack_backazimuth)[self.sS_std_filter])+ '\t')
                f.write('sS' + '\t')
                f.write(str(np.array(self.T_phase_id_picks_abs)[self.sS_std_filter][1]) + '\t')
                f.write(str(np.array(self.T_phase_id_pick_amplitudes)[self.sS_std_filter][1]) +'\n')
            if sum(self.sS_std_filter) > 1:        
                for i in range (len(np.array(self.T_array_no)[self.sS_std_filter])):
                    #if np.array(self.S_onset_abs_time)[self.sS_std_filter][i] != 0:                
                    f.write(str(np.array(self.T_array_no)[self.sS_std_filter][i]) + '\t')
                    f.write(str(np.array(self.T_array_latitude)[self.sS_std_filter][i]) + '\t')
                    f.write(str(np.array(self.T_array_longitude)[self.sS_std_filter][i]) + '\t')
                    f.write(str(np.array(self.T_array_elevation)[self.sS_std_filter][i]) + '\t')
                    f.write(str(np.array(self.epicentral_dist_sS)[self.sS_std_filter][i])+'\t')
                    f.write(str(np.array(self.T_beampack_backazimuth)[self.sS_std_filter][i])+ '\t')
                    f.write('S' + '\t')
                    f.write(str(np.array(self.T_phase_id_picks_abs)[self.sS_std_filter][i][0]) + '\t')
                    f.write(str(np.array(self.T_phase_id_pick_amplitudes)[self.sS_std_filter][i][0]) +'\n')        
                        
                    #if np.array(self.sS_onset_abs_time)[self.sS_std_filter][i] != 0:
                    f.write(str(np.array(self.T_array_no)[self.sS_std_filter][i]) + '\t')
                    f.write(str(np.array(self.T_array_latitude)[self.sS_std_filter][i]) + '\t')
                    f.write(str(np.array(self.T_array_longitude)[self.sS_std_filter][i]) + '\t')
                    f.write(str(np.array(self.T_array_elevation)[self.sS_std_filter][i]) + '\t')
                    f.write(str(np.array(self.epicentral_dist_sS)[self.sS_std_filter][i])+'\t')
                    f.write(str(np.array(self.T_beampack_backazimuth)[self.sS_std_filter][i])+ '\t')
                    f.write('sS' + '\t')
                    f.write(str(np.array(self.T_phase_id_picks_abs)[self.sS_std_filter][i][1]) + '\t')
                    f.write(str(np.array(self.T_phase_id_pick_amplitudes)[self.sS_std_filter][i][1]) +'\n')
                        
        f.close()       
        return
    
    def write_out_catalogue(self, results_dir, catalogue_name, component):   
        
        # Set up Final Catalogue txt file
        output_file = os.path.join(results_dir, catalogue_name)      
        
        f = open(output_file, 'a+')
        if os.path.getsize(output_file) == 0:
            f.write('Event'.ljust(19) + '\t'+ 'Event_id'.ljust(10) + '\t' + 'Lat'.ljust(8) + '\t' + 'Lon'.ljust(8) + '\t' + 'mb'.ljust(4) + '\t' + 'ISC_Depth'.ljust(9) + '\t' + 'R_Depth'.ljust(6) + '\t' + 'Stdev'.ljust(5) + '\t' + 'Mad'.ljust(5) + '\n')
        if component=='Z' and len(self.P_coda_depths) > 1:       
            f.write(str(self.event.origin_time)[:19].ljust(19) + '\t' + str(self.event.event_id).ljust(10) + '\t' + str(self.event.evla).ljust(8) + '\t' + str(self.event.evlo).ljust(8) + '\t' + str(self.event.evm).ljust(4) + '\t' + str(np.round(self.event.evdp,2)).ljust(9) + '\t' + str(np.round(np.median(self.P_coda_depths),2)).ljust(7) + '\t' + str(np.round(statistics.stdev(self.P_coda_depths),2)).ljust(5) + '\t' + str(np.round(median_abs_deviation(self.P_coda_depths),2)).ljust(5)+ '\n')
        elif component=='Z' and len(self.P_coda_depths) == 1:       
            f.write(str(self.event.origin_time)[:19].ljust(19) + '\t' + str(self.event.event_id).ljust(10) + '\t' + str(self.event.evla).ljust(8) + '\t' + str(self.event.evlo).ljust(8) + '\t' + str(self.event.evm).ljust(4) + '\t' + str(np.round(self.event.evdp,2)).ljust(9) + '\t' + str(np.round(np.median(self.P_coda_depths),2)).ljust(7) + '\t' + str(0).ljust(5) + '\t' + str(0).ljust(5)+ '\n')
  
        elif component=='ZNE' and len(self.P_S_coda_depths) > 1:
            f.write(str(self.event.origin_time)[:19].ljust(19) + '\t' + str(self.event.event_id).ljust(10) + '\t' + str(self.event.evla).ljust(8) + '\t' + str(self.event.evlo).ljust(8) + '\t' + str(self.event.evm).ljust(4) + '\t' + str(np.round(self.event.evdp,2)).ljust(9) + '\t'+ str(np.round(np.median(self.P_S_coda_depths),2)).ljust(7) + '\t' + str(np.round(statistics.stdev(self.P_S_coda_depths),2)).ljust(5) + '\t' + str(np.round(median_abs_deviation(self.P_S_coda_depths),2)).ljust(5)+ '\n')
        elif component=='ZNE' and len(self.P_S_coda_depths) == 1:
            f.write(str(self.event.origin_time)[:19].ljust(19) + '\t' + str(self.event.event_id).ljust(10) + '\t' + str(self.event.evla).ljust(8) + '\t' + str(self.event.evlo).ljust(8) + '\t' + str(self.event.evm).ljust(4) + '\t' + str(np.round(self.event.evdp,2)).ljust(9) + '\t'+ str(np.round(np.median(self.P_S_coda_depths),2)).ljust(7) + '\t' + str(0).ljust(5) + '\t' + str(0).ljust(5)+ '\n')
        else:
            pass
        f.close()
        return
        
    def write_out_catalogue_detailed(self, results_dir, catalogue_name, component):   
        
        # Set up Final Catalogue txt file
        output_file = os.path.join(results_dir, catalogue_name)
        
        # Find Numbers of Picks    
        no_P = len(self.Z_array_no)
        
        sum_filter = np.zeros(len(self.pP_std_filter))
        for i in range (len(self.pP_std_filter)):
            sum_filter[i] = self.pP_std_filter[i] + self.sP_std_filter[i]
            if sum_filter[i] == 2:
                sum_filter[i] = 1
        no_P_final = int(sum(sum_filter))
        no_pP = len(np.array(self.dt_pP_P)[np.array(self.dt_pP_P)>0])
        no_pP_final = np.sum(self.pP_std_filter)
        no_sP = len(np.array(self.dt_sP_P)[np.array(self.dt_sP_P)>0])
        no_sP_final = np.sum(self.sP_std_filter)
        
        if component == 'ZNE':
            no_S = len(self.T_array_no)
            no_S_final = np.sum(self.sS_std_filter)
            no_sS = len(np.array(self.dt_sS_S)[np.array(self.dt_sS_S)>0])
            no_sS_final = np.sum(self.sS_std_filter)
        
        f = open(output_file, 'a+')
        if os.path.getsize(output_file) == 0:
            f.write('Event'.ljust(19) + '\t'+ 'Event_id'.ljust(10) + '\t' + 'Lat'.ljust(8) + '\t' + 'Lon'.ljust(8) + '\t' + 'mb'.ljust(4) + '\t' + 'ISC_Depth'.ljust(9) + '\t' + 'R_Depth_P'.ljust(10) + '\t' + 'R_Depth_PS'.ljust(10) + '\t' + 'Stdev'.ljust(5) + '\t' + 'Mad'.ljust(5) + '\t')
            f.write('Total_arrays'.ljust(13) + '\t' + 'Failed_P_arrays'.ljust(16) + '\t' + 'QC_Failed_P_arrays'.ljust(19) + '\t' + 'No.P_picks'.ljust(12)+ '\t' + 'No.pP_picks'.ljust(12)+ '\t' + 'No.sP_picks'.ljust(12)+ '\t' + 'No.P_final_picks'.ljust(18)+ '\t' + 'No.pP_final_picks'.ljust(18)+ '\t' + 'No.sP_final_picks'.ljust(18))
            if component == 'ZNE':
                f.write('\t' + 'Failed_S_arrays'.ljust(16) + '\t' + 'QC_Failed_S_arrays'.ljust(19) + '\t' + 'No.S_picks'.ljust(12) + '\t' + 'No.sS_picks'.ljust(12) + '\t' + 'No.S_final_picks'.ljust(18) + '\t' + 'No.sS_final_picks'.ljust(18) + '\n')
     
        if component=='Z' and len(self.P_coda_depths) > 1:     
            f.write('\n')  
            f.write(str(self.event.origin_time)[:19].ljust(19) + '\t' + str(self.event.event_id).ljust(10) + '\t' + str(self.event.evla).ljust(8) + '\t' + str(self.event.evlo).ljust(8) + '\t' + str(self.event.evm).ljust(4) + '\t' + str(np.round(self.event.evdp,2)).ljust(9) + '\t'+ str(np.round(np.median(self.P_coda_depths),2)).ljust(10) + '\t'+ str(0).ljust(10) + '\t' + str(np.round(statistics.stdev(self.P_coda_depths),2)).ljust(5) + '\t' + str(np.round(median_abs_deviation(self.P_coda_depths),2)).ljust(5)+ '\t')
            f.write(str(self.array_statistics_Z[0]).ljust(13) + '\t' + str(self.array_statistics_Z[1]).ljust(16) + '\t' + str(self.array_statistics_Z[2]).ljust(19) + '\t' + str(no_P).ljust(12)+ '\t' + str(no_pP).ljust(12) + '\t' + str(no_sP).ljust(12) + '\t' + str(no_P_final).ljust(18) + '\t' + str(no_pP_final).ljust(18) + '\t' + str(no_sP_final).ljust(18) + '\n')
        elif component=='Z' and len(self.P_coda_depths) == 1:       
            f.write('\n') 
            f.write(str(self.event.origin_time)[:19].ljust(19) + '\t' + str(self.event.event_id).ljust(10) + '\t' + str(self.event.evla).ljust(8) + '\t' + str(self.event.evlo).ljust(8) + '\t' + str(self.event.evm).ljust(4) + '\t' + str(np.round(self.event.evdp,2)).ljust(9) + '\t' + str(np.round(np.median(self.P_coda_depths),2)).ljust(10) + '\t'+ str(0).ljust(10) + '\t' + str(0).ljust(5) + '\t' + str(0).ljust(5)+ '\n')
            f.write(str(self.array_statistics_Z[0]).ljust(13) + '\t' + str(self.array_statistics_Z[1]).ljust(16) + '\t' + str(self.array_statistics_Z[2]).ljust(19) + '\t' + str(no_P).ljust(12)+ '\t' + str(no_pP).ljust(12) + '\t' + str(no_sP).ljust(12) + '\t' + str(no_P_final).ljust(18) + '\t' + str(no_pP_final).ljust(18) + '\t' + str(no_sP_final).ljust(18) + '\n')
        elif component=='ZNE' and len(self.P_S_coda_depths) > 1:
            f.write(str(self.event.origin_time)[:19].ljust(19) + '\t' + str(self.event.event_id).ljust(10) + '\t' + str(self.event.evla).ljust(8) + '\t' + str(self.event.evlo).ljust(8) + '\t' + str(self.event.evm).ljust(4) + '\t' + str(np.round(self.event.evdp,2)).ljust(9) + '\t' + str(np.round(np.median(self.P_coda_depths),2)).ljust(10) + '\t' + str(np.round(np.median(self.P_S_coda_depths),2)).ljust(10) + '\t' + str(np.round(statistics.stdev(self.P_S_coda_depths),2)).ljust(5) + '\t' + str(np.round(median_abs_deviation(self.P_S_coda_depths),2)).ljust(5)+ '\t')
            f.write(str(self.array_statistics_Z[0]).ljust(13) + '\t' + str(self.array_statistics_Z[1]).ljust(16) + '\t' + str(self.array_statistics_Z[2]).ljust(19) + '\t' + str(no_P).ljust(12)+ '\t' + str(no_pP).ljust(12) + '\t' + str(no_sP).ljust(12) + '\t' + str(no_P_final).ljust(18) + '\t' + str(no_pP_final).ljust(18) + '\t' + str(no_sP_final).ljust(18) + '\t')
            f.write(str(self.array_statistics_T[0]).ljust(13) + '\t' + str(self.array_statistics_T[1]).ljust(16) + '\t' + str(self.array_statistics_T[2]).ljust(19) + '\t' + str(no_S).ljust(12)+ '\t' + str(no_sS).ljust(12) + '\t' + str(no_S_final).ljust(18) + '\t' + str(no_sS_final).ljust(18) + '\n')
        elif component=='ZNE' and len(self.P_S_coda_depths) == 1:
            f.write(str(self.event.origin_time)[:19].ljust(19) + '\t' + str(self.event.event_id).ljust(10) + '\t' + str(self.event.evla).ljust(8) + '\t' + str(self.event.evlo).ljust(8) + '\t' + str(self.event.evm).ljust(4) + '\t' + str(np.round(self.event.evdp,2)).ljust(9) + '\t' + str(np.round(np.median(self.P_coda_depths),2)).ljust(10) + '\t' + str(np.round(np.median(self.P_S_coda_depths),2)).ljust(10) + '\t' + str(0).ljust(5)+ '\n')
            f.write(str(self.array_statistics_Z[0]).ljust(13) + '\t' + str(self.array_statistics_Z[1]).ljust(16) + '\t' + str(self.array_statistics_Z[2]).ljust(19) + '\t' + str(no_P).ljust(12)+ '\t' + str(no_pP).ljust(12) + '\t' + str(no_sP).ljust(12) + '\t' + str(no_P_final).ljust(18) + '\t' + str(no_pP_final).ljust(18) + '\t' + str(no_sP_final).ljust(18) + '\t')
            f.write(str(self.array_statistics_T[0]).ljust(13) + '\t' + str(self.array_statistics_T[1]).ljust(16) + '\t' + str(self.array_statistics_T[2]).ljust(19) + '\t' + str(no_S).ljust(12)+ '\t' + str(no_sS).ljust(12) + '\t' + str(no_S_final).ljust(18) + '\t' + str(no_sS_final).ljust(18) + '\n')
        else:
            pass
        f.close()
        return

class ISCloc:
    def __init__(self):
        return

    def load_event(self, event=None, event_id=None, evname=None, magnitude=None, ev_lat=None, ev_lon=None, ev_depth=None):
        '''Load in key event parameters.'''
        if event != None:
            self.event = event
        elif event_id != None:
            self.event.event_id = event_id
            self.event.evname = evname
            self.event.evm = magnitude
            self.event.evla = ev_lat
            self.event.evlo = ev_lon
            self.event.evdp = ev_depth
        else:
            print('Event cannot be loaded without the relevant inputs.')
        return
        
    @staticmethod    
    def create_input_ISFs(iscloc_inputs_dir, event=None, evname=None, event_id=None, phase_list=None, append=True):     
        
        def get_appending_data(path):
            try:
                sta, phase, time = np.loadtxt(path, skiprows=1, usecols = (0,6,7), unpack=True, dtype=str, delimiter='\t')
                lat, lon, elev, dist, baz = np.loadtxt(path, skiprows=1, usecols = (1, 2, 3, 4, 5), unpack=True, dtype=float, delimiter='\t') 
            except:
                 return 

            back = baz
            
            #Convert backazimuth to azimuth
            evaz = [0]*len(back)
            for i in range (len(back)):
                if back[i] < 180:
                    evaz[i] = back[i] + 180
                if back[i] > 180:
                    evaz[i] = back[i] - 180
                if back[i] == 180:
                    evaz[i] = 0
                if back[i] == 0 or back[i] == 360:
                    evaz[i] = 180
            
            appending_data = []
            for i in range (len(sta)):
                appending_data.append(str(sta[i]).ljust(6) + str(np.round(dist[i],2)).rjust(6) + str(np.round(evaz[i],1)).rjust(6) + ' ' + phase[i].ljust(9) + time[i][11:-5].ljust(98) + 'AEB' + '   ' + 'ZZ'.ljust(30) + str(np.round(lat[i],4)).rjust(12) + str(np.round(lon[i],4)).rjust(10) + str(np.round(elev[i],1)).rjust(8))
                #appending_data.append(str(sta[i]).ljust(6) + str(np.round(dist[i],2)).rjust(6) + str(np.round(calc_az[i],1)).rjust(6) + ' ' + phase[i].ljust(9) + time[i][11:-5].ljust(98) + 'AEB' + '   ' + 'ZZ'.ljust(30) + str(np.round(lat[i],4)).rjust(12) + str(np.round(lon[i],4)).rjust(10) + str(np.round(elev[i],1)).rjust(8))

            return appending_data

        # Open event URLs and scrape text ---------------------------------------------

        def scrape(url):
            response = requests.get(url)
            html_document = response.text
            
            # create soup object
            soup = BeautifulSoup(html_document, 'html.parser')
            txt = soup.get_text()
            
            return txt

        # Open event URLs, scrape text and append Alice's phase data ------------------
        def scrape_and_append(url, evname, event_ids, phase_list):
            response = requests.get(url)
            html_document = response.text
            
            # create soup object
            soup = BeautifulSoup(html_document, 'html.parser')
            out_file = open(iscloc_inputs_dir + str(event_ids) + ".in", "a+")  # ISC phases plus mine
        
            # Find matching evname from Alice's relocate algorithm
            path = phase_list
            #print(path)
            count_phases = 0
            depth_phase_counter = 0
            if os.path.isfile(path):
                try:
                    appending_data = get_appending_data(path)
                    phase = []
                    for string in soup.strings: 
                        if ' pP ' in string or ' sP ' in string or 'sS' in string:
                            depth_phase_counter += 1 
                        if 'STOP' in string:
                            out_file.write(string[:string.index("\n\n\nSTOP")])
                            break
                        else:
                            out_file.write(string)

                    for line in range (len(appending_data)):
                        print(appending_data[line])
                        count_phases += 1
                        out_file.write('\n' + appending_data[line])
                    #out_file.writelines([i + '\n' for i in appending_data])

                except Exception as e:
                    print(e)
                    print('NO ADDITIONAL PHASES ADDED TO EVENT DATA')
               
                out_file.close()
           
            else:
                print('No event match or phase file: %s' %event_ids)
            print('No. of ISC depth phases: ',depth_phase_counter)
            print('No. of Phases Added: ', count_phases)
            return count_phases, depth_phase_counter
    
        # ===========================================================
             
        try:
            if event != None:    
                # Load up ISC url per Event ID
                url = 'http://www.isc.ac.uk/cgi-bin/web-db-run?event_id=' + str(event.event_id) + '&out_format=ISF2&request=COMPREHENSIVE'
                print(url)

                # Set up ISC only output file
                out_file = open(iscloc_inputs_dir+"ISF2_" + str(event.event_id) + ".dat", "a+")  #Just ISC phases printed out
                text = scrape(url)
                out_file.write(text)
		        
                if append == True: 
                    # Connect ISC phases with AB relocation phases
                    no_phases, ISC_depth_phases = scrape_and_append(url, event.evname, event.event_id, phase_list)
                    #ISC_depth_phases = np.asarray(ISC_depth_phases)
                    #np.save('ISC_phases.npy', ISC_depth_phases, allow_pickle=True)
                    #no_phases = np.asarray(no_phases)
                    #np.save('no_phases.npy', no_phases, allow_pickle=True)
            elif evname!= None:
                # Load up ISC url per Event ID
                url = 'http://www.isc.ac.uk/cgi-bin/web-db-run?event_id=' + str(event_id) + '&out_format=ISF2&request=COMPREHENSIVE'
                print(url)

                # Set up ISC only output file
                out_file = open(iscloc_inputs_dir+"ISF2_" + str(event_id) + ".dat", "a+")  #Just ISC phases printed out
                text = scrape(url)
                out_file.write(text)
		        
                if append == True: 
                    # Connect ISC phases with AB relocation phases
                    no_phases, ISC_depth_phases = scrape_and_append(url, evname, event_id, phase_list)
                    #ISC_depth_phases = np.asarray(ISC_depth_phases)
                    #np.save('ISC_phases.npy', ISC_depth_phases, allow_pickle=True)
                    #no_phases = np.asarray(no_phases)
                    #np.save('no_phases.npy', no_phases, allow_pickle=True)
            else:
                print('No event parameters input.')
        except Exception as e:
            print('FAILED:',e)          
        return
        
        
    def create_input_ISFs_ISC(self, iscloc_inputs_dir, event_id, sta=None, dist=None, back=None, phase=None, time=None, lat=None, lon=None, elev=None, append=True):     
        
        # Open event URLs and scrape text ---------------------------------------------

        def scrape(url):
            response = requests.get(url)
            html_document = response.text
            
            # create soup object
            soup = BeautifulSoup(html_document, 'html.parser')
            txt = soup.get_text()            
            return txt
        
        def format_appending_data(sta, dist, back, phase, time, lat, lon, elev):
            #Convert backazimuth to azimuth -----------------------------------
            evaz = [0]*len(back)
            for i in range (len(back)):
                if back[i] < 180:
                    evaz[i] = back[i] + 180
                if back[i] > 180:
                    evaz[i] = back[i] - 180
                if back[i] == 180:
                    evaz[i] = 0
                if back[i] == 0 or back[i] == 360:
                    evaz[i] = 180

            # Make station list 5 characters if longer than 5.
            for i in range (len(sta)):
                if len(sta[i]) > 5:
                    #print(sta[i][-5:])
                    sta[i] = sta[i][-5:]
                    #print(sta)

            appending_data = []
            for i in range (len(sta)):
                appending_data.append(str(sta[i]).ljust(6) + str(np.round(dist[i],2)).rjust(6) + str(np.round(evaz[i],1)).rjust(6) + ' ' + phase[i].ljust(9) + str(time[i])[11:-5].ljust(98) + 'AEB' + '   ' + 'ZZ'.ljust(30) + str(np.round(lat[i],4)).rjust(12) + str(np.round(lon[i],4)).rjust(10) + str(np.round(elev[i],1)).rjust(8) + '\n')
            return appending_data

        
        # Open event URLs, scrape text and append Alice's phase data ------------------
        def append_data_to_ISF(event_id, appending_data):
            
            # Check if ISF file is already downloaded
            if os.path.exists(iscloc_inputs_dir + "/ISF2_" + str(event_id) + ".dat"):
                pass
            else:
                out_file = open(iscloc_inputs_dir + "/ISF2_" + str(event_id) + ".dat", "w")  #Just ISC phases printed out
                text = scrape(url)
                out_file.write(text)
                out_file.close()
            
            # Append phase data
            count_phases = 0
            depth_phase_counter = 0
            blank_space_counter = 0
            
            with open(iscloc_inputs_dir + '/' + str(event_id) + ".in", "w") as new_ISF:  # ISC phases plus mine
                with open(iscloc_inputs_dir + "/ISF2_" + str(event_id) + ".dat") as og_ISF:
                
                    for line_no, line in enumerate(og_ISF):
                        if 'Sta     Dist  EvAz' in line:
                            new_ISF.write(line)
                            break
                            
                        else:
                            new_ISF.write(line)
                            
                    for line_no, line in enumerate(og_ISF):        
                        if re.search('[A-z]+[0-9]*\s+[0-9]+\.[0-9]+\s+[0-9]+\.[0-9]\s[A-z]*\s+[0-9]+\:[0-9]+\:[0-9]+', line):
                            if float(line.split()[1]) >= 120:
                                    continue
                                    
                            else:
                                new_ISF.write(line)
                                if ' pP ' in line or ' sP ' in line or 'sS' in line:
                                    depth_phase_counter += 1 
                        
                    og_ISF.close()

                    try:
                        for line in appending_data:
                            #print(line)
                            count_phases += 1
                            new_ISF.write(line)


                    except Exception as e:
                        print(e)
                        print('NO ADDITIONAL PHASES ADDED TO EVENT DATA')
           
            new_ISF.close()
           
            print('No. of ISC depth phases: ',depth_phase_counter)
            print('No. of Phases Added: ', count_phases)
            return count_phases, depth_phase_counter
    
        # ===========================================================
        # NEW WITH EVENT IDs FROM ORIGINAL CATALOGUE
        
        try:
            # Load up ISC url per Event ID
            url = 'http://www.isc.ac.uk/cgi-bin/web-db-run?event_id=' + str(event_id) + '&out_format=ISF2&request=COMPREHENSIVE'
            print(url)

            # Set up ISC only output file
            # Check if ISF file is already downloaded
            if os.path.exists(iscloc_inputs_dir + "/ISF2_" + str(event_id) + ".dat"):
                pass
            else:
                out_file = open(iscloc_inputs_dir + "/ISF2_" + str(event_id) + ".dat", "w")  #Just ISC phases printed out
                text = scrape(url)
                out_file.write(text)
            
            if append == True: 
                # Connect ISC phases with AB relocation phases
                appending_data = format_appending_data(sta, dist, back, phase, time, lat, lon, elev)
                append_data_to_ISF(event_id, appending_data)
                
                '''ISC_depth_phases = np.asarray(ISC_depth_phases)
                np.save('ISC_phases.npy', ISC_depth_phases, allow_pickle=True)
                no_phases = np.asarray(no_phases)
                np.save('no_phases.npy', no_phases, allow_pickle=True)'''
        except Exception as e:
            print('FAILED:',e)  
        return   
        
    def load_array_metadata(self, results_dir, evname):
                      
        # Load in stations used in arrays for event
        array_no,station_seed = np.loadtxt(results_dir + '/' + str(evname) + '/array_stations_successful.txt', usecols=(0,3), unpack=True, dtype=str, delimiter='\t')
        array_lat, array_lon, array_baz, array_slw, array_elev, array_gcarc = np.loadtxt(results_dir + '/' + str(evname) + '/array_stations_successful.txt', usecols=(1,2,6,7,4,5), unpack=True, dtype=float, delimiter='\t')
        
        #print(array_no, type(station_seed[0]))
        # unique array names, sort other arrays by same unique index
        indexes = np.unique(array_no, return_index=True)[1]
        array_names = [array_no[index] for index in sorted(indexes)]          
        array_lat = [array_lat[index] for index in sorted(indexes)]
        array_lon = [array_lon[index] for index in sorted(indexes)]
        array_baz = [array_baz[index] for index in sorted(indexes)]
        array_slw = [array_slw[index] for index in sorted(indexes)]
        array_elev = [array_elev[index] for index in sorted(indexes)]
        array_gcarc = [array_gcarc[index] for index in sorted(indexes)]
        #print(array_names)
        assert len(array_names) == len(array_lat)
        
        # split out station name from seed
        stations = []
        for sta in station_seed:
            split_sta = sta.split('.')
            #print(split_sta)
            stations.append(split_sta[1])
            
        #print(stations)
        
        # per array name, append stations which are within array -- trace number per array
        arrays = []     
        trace_no = []         
        for j in range (len(array_names)):
            arrays.append([])
            for k in range (len(array_no)):
                if array_names[j] == array_no[k]:
                    arrays[j].append(stations[k])
            trace_no.append(len(arrays[j]))
        
        self.array_names = array_names
        self.array_lat = array_lat
        self.array_lon = array_lon
        self.array_baz = array_baz
        self.array_slw = array_slw
        self.array_elev = array_elev
        self.array_gcarc = array_gcarc
        self.trace_no = trace_no  
        return
        
    def extract_ISC_P_arrivals(self, iscloc_inputs_dir, event_id):     
        # load in station coordinates from ISF with P phase
        # check for stations within 1.25 degrees of array centre
        # Use these.    

        # load in relevant ISF file
        sta = []
        phase = []
        arr_time = []
        lat = []
        lon = []
        with open(iscloc_inputs_dir+"/ISF2_" + str(event_id) + ".dat") as f:
            for line_no, line in enumerate(f):
                if 'Sta' in line:  
                    for line_no, line in enumerate(f):     
                        #print(line[18:21], line[73:76])              
                        if line[18:21] == ' P ' and line[73:76]=='T__':    # only time defining P arrivals extracted                   
                            try:
                                sta.append(line[:5])
                                phase.append(line[18:21])
                                arr_time.append(line[27:39])
                                lat.append(float(line[166:174]))
                                lon.append(float(line[175:185]))
                            except ValueError:
                                if len(sta) != len(phase) != len(arr_time) != len(lat) != len(lon):
                                    smallest = min([sta,phase,arr_time,lat,lon], key=len)
                                    if len(sta) > smallest:
                                        del sta[-1]
                                    if len(phase) > smallest:
                                        del phase[-1]
                                    if len(arr_time) > smallest:
                                        del arr_time[-1]
                                    if len(lat) > smallest:
                                        del lat[-1]
                                    if len(lon) > smallest:
                                        del lon[-1]
                                    assert len(sta) == len(phase) == len(arr_time) == len(lat) == len(lon) # sometimes a column is empty, this deletes the other values from the row
		
        #print(sta, phase, arr_time, lat, lon)
        
        # Check for ISF stations within 1.25 degrees of array centre
        ISF_lats_lons = np.array(list(zip(np.deg2rad(lat), np.deg2rad(lon))))
        array_lats_lons = np.array(list(zip(np.deg2rad(self.array_lat), np.deg2rad(self.array_lon))))

        if ISF_lats_lons.size == 0:
            self.array_ISF_stations =  []
            self.array_ISF_arr_time =   []         
            self.array_ISF_lat = []
            self.array_ISF_lon = [] 
            return

        tree = BallTree(ISF_lats_lons, leaf_size=int(np.round(ISF_lats_lons.shape[0]/2,0)), metric="haversine")
        
        min_radius=np.deg2rad(1.25)
        
        array_ISF_stations = []
        array_ISF_phase = []
        array_ISF_arr_time = []
        array_ISF_lat = []
        array_ISF_lon = []
        
        for i in range(len(array_lats_lons)):
            array_ISF_stations_index = tree.query_radius(X=np.array([array_lats_lons[i]]), r=min_radius, return_distance=False)[0]
            array_ISF_stations.append(np.array(sta)[array_ISF_stations_index])
            array_ISF_phase.append(np.array(phase)[array_ISF_stations_index])
            array_ISF_arr_time.append(np.array(arr_time)[array_ISF_stations_index])
            array_ISF_lat.append(np.array(lat)[array_ISF_stations_index])
            array_ISF_lon.append(np.array(lon)[array_ISF_stations_index])
          
        self.array_ISF_stations = array_ISF_stations
        self.array_ISF_arr_time = array_ISF_arr_time

        self.array_ISF_lat = array_ISF_lat
        self.array_ISF_lon = array_ISF_lon              
        return     

    def extract_ISC_S_arrivals(self, iscloc_inputs_dir, event_id):     
        # load in station coordinates from ISF with P phase
        # check for stations within 1.25 degrees of array centre
        # Use these.    

        # load in relevant ISF file
        sta = []
        phase = []
        arr_time = []
        lat = []
        lon = []
        with open(iscloc_inputs_dir+"/ISF2_" + str(event_id) + ".dat") as f:
            for line_no, line in enumerate(f):
                if 'Sta' in line:  
                    for line_no, line in enumerate(f):     
                        #print(line[18:21], line[73:76])              
                        if line[18:21] == ' S ' and line[73:76]=='T__':    # only time defining P arrivals extracted
                            try:
                                sta.append(line[:5])
                                phase.append(line[18:21])
                                arr_time.append(line[27:39])
                                lat.append(float(line[166:174]))
                                lon.append(float(line[175:185]))
                            except ValueError:
                                if len(sta) != len(phase) != len(arr_time) != len(lat) != len(lon):
                                    smallest = min([sta,phase,arr_time,lat,lon], key=len)
                                    if len(sta) > smallest:
                                        del sta[-1]
                                    if len(phase) > smallest:
                                        del phase[-1]
                                    if len(arr_time) > smallest:
                                        del arr_time[-1]
                                    if len(lat) > smallest:
                                        del lat[-1]
                                    if len(lon) > smallest:
                                        del lon[-1]
                                    assert len(sta) == len(phase) == len(arr_time) == len(lat) == len(lon) # sometimes a column is empty, this deletes the other values from the row
		
        #print(sta, phase, arr_time, lat, lon)
        
        # Check for ISF stations within 1.25 degrees of array centre
        ISF_lats_lons = np.array(list(zip(np.deg2rad(lat), np.deg2rad(lon))))
        array_lats_lons = np.array(list(zip(np.deg2rad(self.array_lat), np.deg2rad(self.array_lon))))

        if ISF_lats_lons.size == 0:
            self.array_ISF_S_stations =  []
            self.array_ISF_S_arr_time =   []         
            self.array_ISF_S_lat = []
            self.array_ISF_S_lon = [] 
            return

        tree = BallTree(ISF_lats_lons, leaf_size=int(np.round(ISF_lats_lons.shape[0]/2,0)), metric="haversine")
        
        min_radius=np.deg2rad(1.25)
        
        array_ISF_stations = []
        array_ISF_phase = []
        array_ISF_arr_time = []
        array_ISF_lat = []
        array_ISF_lon = []
        
        for i in range(len(array_lats_lons)):
            array_ISF_stations_index = tree.query_radius(X=np.array([array_lats_lons[i]]), r=min_radius, return_distance=False)[0]
            array_ISF_stations.append(np.array(sta)[array_ISF_stations_index])
            array_ISF_phase.append(np.array(phase)[array_ISF_stations_index])
            array_ISF_arr_time.append(np.array(arr_time)[array_ISF_stations_index])
            array_ISF_lat.append(np.array(lat)[array_ISF_stations_index])
            array_ISF_lon.append(np.array(lon)[array_ISF_stations_index])
        
        self.array_ISF_S_stations = array_ISF_stations
        self.array_ISF_S_arr_time = array_ISF_arr_time
        self.array_ISF_S_lat = array_ISF_lat
        self.array_ISF_S_lon = array_ISF_lon              
        return     
        
    def determine_array_failure_rate_for_P(self, results_dir):
        '''Find how many ad-hoc arrays do not have a recorded P pick in the ISC catalogue.'''
        try:
            with open(results_dir+'/No_manual_pick_failure_for_P.txt') as f:
                if str(self.event.event_id) in f.read():
                    pass
                else:
                    fail = 0
                    for sta in self.array_ISF_stations:
                        if sta.size == 0:
                            fail += 1
                    if self.array_ISF_stations == []: # force 100% failure when no stations available
                        print('Failure no:', fail, 'out of', len(self.array_names)) ; print('Failure Rate:', 100, '%')
                        f = open(results_dir+'/No_manual_pick_failure_P.txt', 'a+')
                        f.write(str(self.event.evname).ljust(20) + str(self.event.event_id).ljust(20) + str(self.event.evm).ljust(5) + str(len(self.array_names)).ljust(5) + str(fail).ljust(5) + str(100).ljust(5) + '\n')
                        f.close()
                    else:
                        print('Failure no:', fail, 'out of', len(array_names)) ; print('Failure Rate:', np.round((fail/len(self.array_names))*100,2), '%')
                        f = open(results_dir+'/No_manual_pick_failure_P.txt', 'a+')
                        f.write(str(self.event.evname).ljust(20) + str(self.event.event_id).ljust(20) + str(self.event.evm).ljust(5) + str(len(self.array_names)).ljust(5) + str(fail).ljust(5) + str(np.round((fail/len(self.array_names))*100,2)).ljust(5) + '\n')
                        f.close()
        except FileNotFoundError:
            fail = 0
            for sta in self.array_ISF_stations:
                if sta.size == 0:
                    fail += 1
            if self.array_ISF_stations == []: # force 100% failure when no stations available
                print('Failure no:', fail, 'out of', len(self.array_names)) ; print('Failure Rate:', 100, '%')
                f = open(results_dir+'/No_manual_pick_failure_P.txt', 'a+')
                f.write(str(self.event.evname).ljust(20) + str(self.event.event_id).ljust(20) + str(self.event.evm).ljust(5) + str(len(self.array_names)).ljust(5) + str(fail).ljust(5) + str(100).ljust(5) + '\n')
                f.close()
            else:
                print('Failure no:', fail, 'out of', len(self.array_names)) ; print('Failure Rate:', np.round((fail/len(self.array_names))*100,2), '%')
                f = open(results_dir+'/No_manual_pick_failure_P.txt', 'a+')
                f.write(str(self.event.evname).ljust(20) + str(self.event.event_id).ljust(20) + str(self.event.evm).ljust(5) + str(len(self.array_names)).ljust(5) + str(fail).ljust(5) + str(np.round((fail/len(self.array_names))*100,2)).ljust(5) + '\n')
                f.close()
        return   
        
    def determine_array_failure_rate_for_S(self, results_dir):
        '''Find how many ad-hoc arrays do not have a recorded P pick in the ISC catalogue.'''
        try:
            with open(results_dir+'/No_manual_pick_failure_for_S.txt') as f:
                if str(self.event.event_id) in f.read():
                    pass
                else:
                    fail = 0
                    for sta in self.array_ISF_S_stations:
                        if sta.size == 0:
                            fail += 1
                    if self.array_ISF_S_stations == []: # force 100% failure when no stations available
                        print('Failure no:', fail, 'out of', len(self.array_names)) ; print('Failure Rate:', 100, '%')
                        f = open(results_dir+'/No_manual_pick_failure_S.txt', 'a+')
                        f.write(str(self.event.evname).ljust(20) + str(self.event.event_id).ljust(20) + str(self.event.evm).ljust(5) + str(len(self.array_names)).ljust(5) + str(fail).ljust(5) + str(100).ljust(5) + '\n')
                        f.close()
                    else:
                        print('Failure no:', fail, 'out of', len(array_names)) ; print('Failure Rate:', np.round((fail/len(self.array_names))*100,2), '%')
                        f = open(results_dir+'/No_manual_pick_failure_S.txt', 'a+')
                        f.write(str(self.event.evname).ljust(20) + str(self.event.event_id).ljust(20) + str(self.event.evm).ljust(5) + str(len(self.array_names)).ljust(5) + str(fail).ljust(5) + str(np.round((fail/len(self.array_names))*100,2)).ljust(5) + '\n')
                        f.close()
        except FileNotFoundError:
            fail = 0
            for sta in self.array_ISF_S_stations:
                if sta.size == 0:
                    fail += 1
            if self.array_ISF_S_stations == []: # force 100% failure when no stations available
                print('Failure no:', fail, 'out of', len(self.array_names)) ; print('Failure Rate:', 100, '%')
                f = open(results_dir+'/No_manual_pick_failure_S.txt', 'a+')
                f.write(str(self.event.evname).ljust(20) + str(self.event.event_id).ljust(20) + str(self.event.evm).ljust(5) + str(len(self.array_names)).ljust(5) + str(fail).ljust(5) + str(100).ljust(5) + '\n')
                f.close()
            else:
                print('Failure no:', fail, 'out of', len(self.array_names)) ; print('Failure Rate:', np.round((fail/len(self.array_names))*100,2), '%')
                f = open(results_dir+'/No_manual_pick_failure_S.txt', 'a+')
                f.write(str(self.event.evname).ljust(20) + str(self.event.event_id).ljust(20) + str(self.event.evm).ljust(5) + str(len(self.array_names)).ljust(5) + str(fail).ljust(5) + str(np.round((fail/len(self.array_names))*100,2)).ljust(5) + '\n')
                f.close()
        return     
        
    def correct_P_onsets_to_array_centre(self):
        ''' Use a planar slowness correction to move ISC reported P phases to ad-hoc array geometric centre. Find median P wave arrival time, per ad-hoc array.'''

        event = self.event.evname
        stations = self.array_ISF_stations
        arr_time = self.array_ISF_arr_time
        station_lat = self.array_ISF_lat
        station_lon = self.array_ISF_lon
        point_lat = self.array_lat
        point_lon = self.array_lon
        slowness = self.array_slw
        backazimuth = np.radians(self.array_baz) 
        self.P_ISF = [0]*len(stations)
        self.P_ISF_timestamp = [0]*len(stations)
        self.no_used_ISF_P_phases = [0]*len(stations)   
        
        # convert arr_time to datetime format
        yyyy = int(event[:4])
        mn = int(event[4:6])
        dd = int(event[6:8])
        
        for j in range (len(stations)):  # for each ad-hoc array     
            if self.array_ISF_stations[j].size == 0:
                self.P_ISF[j] = 0
                self.P_ISF_timestamp[j] = 0
                self.no_used_ISF_P_phases[j] = 0
                continue
            
            arr_time_UTC = [0]*len(arr_time[j])
            arr_time_UTC_timestamp = [0]*len(arr_time[j])
            for i in range (len(arr_time[j])): # for each ISF P phase in ad-hoc array
                hh = int(arr_time[j][i][1:3])
                mm = int(arr_time[j][i][4:6])
                ss = float(arr_time[j][i][7:-1])  
            
                arr_time_UTC[i] = obspy.UTCDateTime(yyyy,mn,dd,hh,mm,ss)
                arr_time_UTC_timestamp[i] = arr_time_UTC[i].timestamp       
                
            # Calculate and apply timeshift
            corr_arr_time = [0]* len(arr_time_UTC)
            for i in range (len(stations[j])):
                baz_dist = degrees2kilometers((point_lon[j] - float(station_lon[j][i]))*math.sin(backazimuth[j]) + (point_lat[j] - float(station_lat[j][i]))*math.cos(backazimuth[j]))
                timeshift = (slowness[j]*baz_dist)*-1        
                corr_arr_time[i] = arr_time_UTC[i] + timeshift
            
            add = [] # temporary list to store corrected arrival times
            for i in range (len(corr_arr_time)):
                # sanity check that stations are within ad-hoc array parameters
                if float(station_lat[j][i]) > point_lat[j]+2.5 or float(station_lat[j][i]) < point_lat[j]-2.5 or float(station_lon[j][i]) > point_lon[j]+2.5 or float(station_lon[j][i]) < point_lon[j]-2.5:
                    continue
                else:
                    add.append(corr_arr_time[i].timestamp)
            
            if add == []:
                self.P_ISF[j] = 0
                self.P_ISF_timestamp[j] = 0
                self.no_used_ISF_P_phases[j] = 0
                continue
                
            elif len(add)==1:
                self.P_ISF[j] = UTCDateTime(add[0])
                self.P_ISF_timestamp[j] = add[0]
                self.no_used_ISF_P_phases[j] = 1
                continue
            
            else:
                av_P_onset = UTCDateTime(np.median(add))     # use median if there are multiple ISC phases   
                self.P_ISF[j] = av_P_onset
                self.P_ISF_timestamp[j] = av_P_onset.timestamp
                self.no_used_ISF_P_phases[j] = len(add)
                continue
        return 
        
    def correct_S_onsets_to_array_centre(self):
        ''' Use a planar slowness correction to move ISC reported P phases to ad-hoc array geometric centre. Find median P wave arrival time, per ad-hoc array.'''

        event = self.event.evname
        stations = self.array_ISF_S_stations
        arr_time = self.array_ISF_S_arr_time
        station_lat = self.array_ISF_S_lat
        station_lon = self.array_ISF_S_lon
        point_lat = self.array_lat
        point_lon = self.array_lon
        slowness = self.array_slw
        backazimuth = np.radians(self.array_baz)   
        
        self.S_ISF = [0]*len(stations)
        self.S_ISF_timestamp = [0]*len(stations)
        self.no_used_ISF_S_phases = [0]*len(stations)   
        
        # convert arr_time to datetime format
        yyyy = int(event[:4])
        mn = int(event[4:6])
        dd = int(event[6:8])
        
        for j in range (len(stations)):  # for each ad-hoc array     
            if self.array_ISF_S_stations[j].size == 0:
                self.S_ISF[j] = 0
                self.S_ISF_timestamp[j] = 0
                self.no_used_ISF_S_phases[j] = 0
                continue
            
            arr_time_UTC = [0]*len(arr_time[j])
            arr_time_UTC_timestamp = [0]*len(arr_time[j])
            for i in range (len(arr_time[j])): # for each ISF S phase in ad-hoc array
                hh = int(arr_time[j][i][1:3])
                mm = int(arr_time[j][i][4:6])
                ss = float(arr_time[j][i][7:-1])  
            
                arr_time_UTC[i] = obspy.UTCDateTime(yyyy,mn,dd,hh,mm,ss)
                arr_time_UTC_timestamp[i] = arr_time_UTC[i].timestamp       
            
            # Calculate and apply timeshift
            corr_arr_time = [0]* len(arr_time_UTC)
            for i in range (len(stations[j])):
                baz_dist = degrees2kilometers((point_lon[j] - float(station_lon[j][i]))*math.sin(backazimuth[j]) + (point_lat[j] - float(station_lat[j][i]))*math.cos(backazimuth[j]))
                timeshift = (slowness[j]*baz_dist)*-1          
                corr_arr_time[i] = arr_time_UTC[i] + timeshift
            
            add = [] # temporary list to store corrected arrival times
            for i in range (len(corr_arr_time)):
                # sanity check that stations are within ad-hoc array parameters
                if float(station_lat[j][i]) > point_lat[j]+2.5 or float(station_lat[j][i]) < point_lat[j]-2.5 or float(station_lon[j][i]) > point_lon[j]+2.5 or float(station_lon[j][i]) < point_lon[j]-2.5:
                    continue
                else:
                    add.append(corr_arr_time[i].timestamp)
            
            if add == []:
                self.S_ISF[j] = UTCDateTime(0)
                self.S_ISF_timestamp[j] = 0
                self.no_used_ISF_S_phases[j] = 0
                continue
                
            elif len(add)==1:
                self.S_ISF[j] = UTCDateTime(add[0])
                self.S_ISF_timestamp[j] = add[0]
                self.no_used_ISF_S_phases[j] = 1
                continue
            
            else:
                av_S_onset = UTCDateTime(np.median(add))     # use median if there are multiple ISC phases   
                self.S_ISF[j] = av_S_onset
                self.S_ISF_timestamp[j] = av_S_onset.timestamp
                self.no_used_ISF_S_phases[j] = len(add)
                continue
        return 

    def extract_peaks_in_utc(self, array_class):
                 
        def store_attribute(array_class, wanted_attribute):
            store_list = []           
            for array in range(len(array_class)):
                try:
                    attr = getattr(array_class[array], wanted_attribute)
                except:
                    print('No %s attribute' %wanted_attribute)
                    store_list.append(0)
                    continue                                      
                if type(attr) != list:
                    store_list.append(attr)
                else:
                    store_list.extend(np.array([attr]))
            return store_list
            
        picks = store_attribute(array_class, 'phase_id_picks')        
        streams = store_attribute(array_class, 'phase_weighted_beams')
        slw_indexes = store_attribute(array_class, 'slowness_index')
        
        utc_time = []
        for i in range (len(array_class)):
            utc_time.append(streams[i][slw_indexes[i]].times('utcdatetime')) 

        peaks = []
        for i in range (len(picks)):
            array_peaks = picks[i]

            if array_peaks[0] == 0:
                P = 0
            else:
                P = utc_time[i][array_peaks[0]]
            
            if array_peaks[1] == 0:
                pP = 0
            else:
                pP = utc_time[i][array_peaks[1]]
            
            if array_peaks[2] == 0:
                sP = 0
            else:
                sP = utc_time[i][array_peaks[2]]
            
            peaks.append([P,pP,sP])
        self.peaks = peaks
        return 
        
    def extract_S_peaks_in_utc(self, array_class, array_class_Z):
                 
        def store_attribute(array_class, wanted_attribute):
            store_list = []           
            for array in range(len(array_class)):
                try:
                    attr = getattr(array_class[array], wanted_attribute)
                except:
                    print('No %s attribute' %wanted_attribute)
                    store_list.append(0)
                    continue                                      
                if type(attr) != list:
                    store_list.append(attr)
                else:
                    store_list.extend(np.array([attr]))
            return store_list
            
        picks = store_attribute(array_class, 'phase_id_picks')        
        streams = store_attribute(array_class, 'phase_weighted_beams')
        slw_indexes = store_attribute(array_class, 'slowness_index')
        array_no_T = store_attribute(array_class, 'array_no')
        array_no_Z = store_attribute(array_class_Z, 'array_no')
        
        print()
        print(array_no_Z)
        print()
        print(array_no_T)
        print()
        
        utc_time = []
        for i in range (len(array_class)):
            utc_time.append(streams[i][slw_indexes[i]].times('utcdatetime')) 
        
        peaks = []       
        for j in range (len(array_no_Z)):  # Populate array in line with P picks array no.s
            flag = 0
            for i in range (len(picks)):
                if array_no_Z[j] == array_no_T[i]:
                    array_peaks = picks[i]
                    
                    if array_peaks[0] == 0:
                        S = 0
                    else:
                        S = utc_time[i][array_peaks[0]]
                    
                    if array_peaks[1] == 0:
                        sS = 0
                    else:
                        sS = utc_time[i][array_peaks[1]]

                    peaks.append([S,sS])
                    flag = 1
                    break
            
            if flag == 0:    
                peaks.append([0,0])

        self.S_peaks = peaks
        return 
        
  
    def find_timeshift(self):
        ''' Find timeshift between amplitude picks and ISF median P onset.'''
        P_peak_diff = [0]*len(self.peaks)
        print(self.P_ISF, self.peaks)
        for i in range (len(self.peaks)):
            P_peak_diff[i] = UTCDateTime(self.P_ISF[i]) - self.peaks[i][0]
        self.P_peak_diff = P_peak_diff
        return
        
    def find_S_timeshift(self):
        ''' Find timeshift between amplitude picks and ISF median P onset.'''
        S_peak_diff = [0]*len(self.S_peaks)
        for i in range (len(self.S_peaks)):
            #print()
            #print(self.S_ISF, self.S_peaks)
            S_peak_diff[i] = UTCDateTime(self.S_ISF[i]) - self.S_peaks[i][0]
        self.S_peak_diff = S_peak_diff
        return
  
    def apply_timeshift_to_peaks(self):
        # Shift AB picks to ISF P onset time for ISCloc
        # Find difference between P_peak and P_ISF, remove zero values    
        final_P = []
        array_no_P = []
        final_pP = []
        array_no_pP = []
        final_sP = []
        array_no_sP = []
        dist_P = []
        dist_pP = []
        dist_sP = []
        back_P = []
        back_pP = []
        back_sP = []
        lat_P = []
        lat_pP = []
        lat_sP = []
        lon_P = []
        lon_pP = []
        lon_sP = []
        elev_P = []
        elev_pP = []
        elev_sP = []

        for i in range (len(self.P_peak_diff)):
            if self.peaks[i][0] != 0 and self.no_used_ISF_P_phases[i]!=0:
                final_P.append(self.P_ISF[i])
                array_no_P.append(self.array_names[i])
                dist_P.append(self.array_gcarc[i])
                back_P.append(self.array_baz[i])
                lat_P.append(self.array_lat[i])
                lon_P.append(self.array_lon[i])
                elev_P.append(self.array_elev[i])
                
            if self.peaks[i][1] != 0 and self.no_used_ISF_P_phases[i]!=0:
                final_pP.append(self.peaks[i][1] - self.P_peak_diff[i])
                array_no_pP.append(self.array_names[i])
                dist_pP.append(self.array_gcarc[i])
                back_pP.append(self.array_baz[i])
                lat_pP.append(self.array_lat[i])
                lon_pP.append(self.array_lon[i])
                elev_pP.append(self.array_elev[i])
                            
            if self.peaks[i][2] != 0 and self.no_used_ISF_P_phases[i]!=0:
                final_sP.append(self.peaks[i][2] - self.P_peak_diff[i])
                array_no_sP.append(self.array_names[i])
                dist_sP.append(self.array_gcarc[i])
                back_sP.append(self.array_baz[i])
                lat_sP.append(self.array_lat[i])
                lon_sP.append(self.array_lon[i])
                elev_sP.append(self.array_elev[i])
                
        data= {
            'Array Name': array_no_P,
            'Final P': final_P}
        df=pd.DataFrame(data)
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(df)
            
        data= {
            'Array Name': array_no_pP,
            'Final pP': final_pP}
        df=pd.DataFrame(data)
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(df)
            
        data= {
            'Array Name': array_no_sP,
            'Final sP': final_sP}
        df=pd.DataFrame(data)
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(df)
            
        sta = np.concatenate((array_no_P, array_no_pP, array_no_sP))
        dist = np.concatenate((dist_P, dist_pP, dist_sP))
        back = np.concatenate((back_P, back_pP, back_sP))
        phase = np.concatenate((['P']*len(array_no_P), ['pP']*len(array_no_pP), ['sP']*len(array_no_sP)))
        time = np.concatenate((final_P, final_pP, final_sP))
        lat =  np.concatenate((lat_P, lat_pP, lat_sP))
        lon =  np.concatenate((lon_P, lon_pP, lon_sP))
        elev = np.concatenate((elev_P, elev_pP, elev_sP))
       
        return sta, dist, back, phase, time, lat, lon, elev
        
    def apply_timeshift_to_S_peaks(self, sta, dist, back, phase, times, lat, lon, elev, use_P=False):
        # Shift AB picks to ISF S onset time for ISCloc
        # Find difference between S_peak and S_ISF, remove zero values    
        final_S = []
        array_no_S = []
        final_sS = []
        array_no_sS = []
        dist_S = []
        dist_sS = []  
        back_S = []
        back_sS = []   
        lat_S = []
        lat_sS = []  
        lon_S = []
        lon_sS = []   
        elev_S = []
        elev_sS = []
        
        if use_P == False:
            for i in range (len(self.S_peaks)):
                if self.S_peaks[i][0] != 0 and self.no_used_ISF_S_phases[i]!=0:
                    final_S.append(self.S_ISF[i])
                    array_no_S.append(self.array_names[i])
                    dist_S.append(self.array_gcarc[i])
                    back_S.append(self.array_baz[i])
                    lat_S.append(self.array_lat[i])
                    lon_S.append(self.array_lon[i])
                    elev_S.append(self.array_elev[i])
                    
                if self.S_peaks[i][1] != 0 and self.no_used_ISF_S_phases[i]!=0:
                    final_sS.append(self.S_peaks[i][1] - self.S_peak_diff[i])
                    array_no_sS.append(self.array_names[i])
                    dist_sS.append(self.array_gcarc[i])
                    back_sS.append(self.array_baz[i])
                    lat_sS.append(self.array_lat[i])
                    lon_sS.append(self.array_lon[i])
                    elev_sS.append(self.array_elev[i])
        
        if use_P == True: # Use time shift found from ISF P arrivals/P arrivals
            for i in range (len(self.S_peaks)):
                if self.S_peaks[i][0] != 0 and self.no_used_ISF_P_phases[i]!=0:
                    final_S.append(self.S_peaks[i][0] - self.P_peak_diff[i])
                    array_no_S.append(self.array_names[i])
                    dist_S.append(self.array_gcarc[i])
                    back_S.append(self.array_baz[i])
                    lat_S.append(self.array_lat[i])
                    lon_S.append(self.array_lon[i])
                    elev_S.append(self.array_elev[i])
                    
                if self.S_peaks[i][1] != 0 and self.no_used_ISF_P_phases[i]!=0:
                    final_sS.append(self.S_peaks[i][1] - self.P_peak_diff[i])
                    array_no_sS.append(self.array_names[i])
                    dist_sS.append(self.array_gcarc[i])
                    back_sS.append(self.array_baz[i])
                    lat_sS.append(self.array_lat[i])
                    lon_sS.append(self.array_lon[i])
                    elev_sS.append(self.array_elev[i])
                
        data= {
            'Array Name': array_no_S,
            'Final S': final_S}
        df=pd.DataFrame(data)
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(df)
            
        data= {
            'Array Name': array_no_sS,
            'Final sS': final_sS}
        df=pd.DataFrame(data)
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(df)
            
        sta = np.concatenate((sta, array_no_S, array_no_sS))
        dist = np.concatenate((dist, dist_S, dist_sS))
        back = np.concatenate((back, back_S, back_sS))
        phase = np.concatenate((phase, ['S']*len(array_no_S), ['sS']*len(array_no_sS)))
        time = np.concatenate((times, final_S, final_sS))
        lat =  np.concatenate((lat, lat_S, lat_sS))
        lon =  np.concatenate((lon,lon_S, lon_sS))
        elev = np.concatenate((elev, elev_S, elev_sS))
       
        return sta, dist, back, phase, time, lat, lon, elev
        
        
    def make_event_station_list(self, iscloc_inputs_dir, event_id, sta_list_location, new_list, augmented=False):
        # load in input ISF 2.1
        if os.path.exists(iscloc_inputs_dir+"/ISF2_" + str(event_id) + ".dat"):
                pass
        else:
            print('No ISF 2.1 file for event')
            return
        
        # Strip out station names, lat, lon, elevation       
        staname = []
        lat = []
        lon = []
        elev = []
        
        if augmented == False:
            ISF = iscloc_inputs_dir+"/ISF2_" + str(event_id) + ".dat"
        else:
            ISF = iscloc_inputs_dir+"/" + str(event_id) + ".in"
        
        with open(ISF) as file:
            iscloc_lines = [line.rstrip() for line in file]
        file.close()
            
        for line in iscloc_lines:
            #print(line)
            if re.search('^[\w]+\s+[0-9]+\.[0-9]+\s+[0-9]+\.[0-9]\s+',line): #[0-9]{2}\:[0-9]{2}\:[0-9]{2} not all phases have a phase or time...
                if line[73:76] == '___' or line[73:76] == 'T__' or line[132:134] == 'ZZ': 
                    staname.append(line[:6].strip())
                    lat.append(line[166:174].strip())
                    lon.append(line[176:184].strip())
                    elev.append(line[185:192].strip())       
        
        def unique(sequence):
            seen = set()
            return [x for x in sequence if not (x in seen or seen.add(x))]
       
        # Find unique station entries
        new_sta = unique(staname)
        #print(new_sta)
        
        indices_to_copy = [list(staname).index(element) for element in new_sta]

        new_lon = [lon[index] for index in indices_to_copy]
        new_lat = [lat[index] for index in indices_to_copy]
        new_elev = [elev[index] for index in indices_to_copy]
        
        # Write out station list into event named file
        if new_list==True:
            sta_list = open(sta_list_location + '/station_list.%s' %event_id, "w")  #open new station list file
        else:
            sta_list = open(sta_list_location + '/station_list.%s' %event_id, "a+")  #append to station list file
            
        for i in range(len(new_sta)):  
            if str(new_sta[i]) + ', ' + str(new_sta[i]) + ', ' + str(new_lat[i])  + ', ' + str(new_lon[i]) + ', '  + str(new_elev[i]) + '\n' in open(sta_list_location + '/station_list.%s' %event_id).read():
                continue
            else:
                sta_list.write(str(new_sta[i]) + ', ' + str(new_sta[i]) + ', ' + str(new_lat[i])  + ', ' + str(new_lon[i]) + ', '  + str(new_elev[i]) + '\n')          
        return   
        
                
    def correct_P_onset_to_point_singular(self, event, arr_time, station_lat, station_lon, point_lat, point_lon, slowness, backazimuth):
        
        #print(event, arr_time, station_lat, station_lon, point_lat, point_lon, slowness, backazimuth)
        
        backazimuth = np.radians(backazimuth)
        
        if type(arr_time) != UTCDateTime:
            # convert arr_time to datetime format
            yyyy = int(event[:4])
            mn = int(event[4:6])
            dd = int(event[6:8])  
            hh = int(arr_time[1:3])
            mm = int(arr_time[4:6])
            ss = float(arr_time[7:-1])  
            arr_time_UTC = obspy.UTCDateTime(yyyy,mn,dd,hh,mm,ss)
        else:
            arr_time_UTC = arr_time
        arr_time_UTC_timestamp = arr_time_UTC.timestamp
        #print(point_lat, point_lon, slowness, np.degrees(backazimuth))       
        
        # Calculate and apply timeshift
        baz_dist = (float(point_lon) - float(station_lon))*math.sin(backazimuth) + (float(point_lat) - float(station_lat))*math.cos(backazimuth)
        #print(baz_dist)
        baz_dist = degrees2kilometers(baz_dist)
        #print(baz_dist)
        timeshift = (slowness*baz_dist)*-1          
        corr_arr_time = arr_time_UTC + timeshift
        print('Initial time  ', arr_time_UTC)
        print('Corrected time', corr_arr_time)
        
        return corr_arr_time, corr_arr_time.timestamp              
            
    def plot_trace_with_picks(self, trace, picks): 
        # [P_ISF[j], P_lines[j], P_width[j], P_backstep[j], P_peak[j]]
        #print(picks)
        if picks[0] == 0:
            return
        fig, ax = plt.subplots(1,1)
        trace.trim(picks[0]-20,picks[0]+20)
        ax.plot(trace.times('utcdatetime'), trace, color='k')
        if picks[0] != 0:
            ax.scatter(picks[0], 0,color='grey')
        if picks[1] != 0:
            ax.scatter(picks[1], 0,color='green')
        if picks[2] != 0:
            ax.scatter(picks[2], 0,color='red')
        if picks[3] != 0:
            ax.scatter(picks[3], 0,color='purple')
        if picks[4] != 0:
            ax.scatter(picks[4], 0,color='blue')
        plt.close()
        return fig

class Figures:
    def __init__(self, array_class=None, global_class=None, event_class=None):
            ''' initialise class with key inputs'''
            # If no array_class, need event_class input for event details #
            if array_class == None:
                if event_class == None:
                    pass
                else:
                    self.event = event_class
            else:
                self.event = array_class.event
                self.array_class = array_class
                
            if global_class == None:
                pass
            else:
                self.global_class = global_class
                 
    def BP_polar_plot(self, ax=0, fig=0, phase='P'):
        '''Beampack polar plot'''
        
        if phase == 'P':
            back = self.array_class.backazimuth_range
            back_rad = [0] * len(back)
            for i in range (len(back)):
                back_rad[i] = math.radians(back[i])
            slow_range = self.array_class.slowness_range
            max_env_grd = self.array_class.max_P_envelope_grd
            beampack_baz = self.array_class.beampack_backazimuth
            beampack_slow = self.array_class.beampack_slowness
            subarray_baz = self.array_class.array_baz
            subarray_slowness = self.array_class.taup_slowness
            ev_st_gcarc = self.array_class.ev_array_gcarc
        if phase =='S':
            back = self.array_class.backazimuth_range
            back_rad = [0] * len(back)
            for i in range (len(back)):
                back_rad[i] = math.radians(back[i])
            slow_range = self.array_class.slowness_range
            max_env_grd = self.array_class.max_P_envelope_grd
            beampack_baz = self.array_class.beampack_backazimuth
            beampack_slow = self.array_class.beampack_slowness
            subarray_baz = self.array_class.array_baz
            subarray_slowness = self.array_class.taup_S_slowness
            ev_st_gcarc = self.array_class.ev_array_gcarc
        
        # Make polar plot
        cmap = obspy_sequential
        theta, r = np.meshgrid(back_rad, slow_range)
    
        if ax == 0:
            fig, ax = plt.subplots(subplot_kw=dict(projection='polar'), figsize = (8,8))
            ax.contourf(theta, r, max_env_grd)
            ax.set_thetamin(back[0])
            ax.set_thetamax(back[-1])
            cax = fig.add_axes([1, 0.2, 0.04, 0.5])
            ColorbarBase(cax, cmap=cmap, norm=Normalize(vmin=max_env_grd.min(), vmax=max_env_grd.max()))
            ax.set_theta_direction(-1)
            ax.set_theta_zero_location("N")
            ax.scatter(math.radians(beampack_baz), beampack_slow, marker = 'o', color = 'r', s=30, label = 'Best Fit')
            ax.scatter(math.radians(subarray_baz), subarray_slowness, marker = 'o', facecolor = 'none', edgecolor = 'red', s=30, label = 'Expected Fit')
            ax.xaxis.set_tick_params(labelsize=14, pad =10)
            ax.yaxis.set_tick_params(labelsize=14)
            ax.legend(loc='upper right', borderpad=0.5, fontsize=12)
            #ax.title('Beampacking_%s' %ev_st_gcarc)
            #plt.show()
            plt.close()
            
        else:
            ax.contourf(theta, r, max_env_grd)
            ax.set_thetamin(back[0])
            ax.set_thetamax(back[-1])      
            cax = fig.add_axes([0.2, 0.15, 0.22, 0.02])
            ColorbarBase(cax, cmap=cmap, norm=Normalize(vmin=max_env_grd.min(), vmax=max_env_grd.max()), orientation='horizontal')
            cb = fig.colorbar(cm.ScalarMappable(norm=Normalize(vmin=max_env_grd.min(), vmax=max_env_grd.max()),cmap=cmap), cax = cax, orientation='horizontal', pad=0.05) 
            cb.ax.tick_params(labelsize=14)
            ax.set_theta_direction(-1)
            ax.set_theta_zero_location("N")
            ax.scatter(math.radians(beampack_baz), beampack_slow, marker = 'o', color = 'r', s=50, label = 'Best Fit')
            ax.scatter(math.radians(subarray_baz), subarray_slowness, marker = 'o', facecolor = 'none', edgecolor = 'red', s=50, label = 'Expected Fit')
            ax.xaxis.set_tick_params(labelsize=14, pad =10)
            ax.yaxis.set_tick_params(labelsize=14)
            ax.legend(loc='upper right', borderpad=0.5, fontsize=12)
            #ax.title('Beampacking_%s' %ev_st_gcarc)
            #plt.close()
            fig.tight_layout()
        return fig
        
    def Timeshift_Traces(self, xlim_start=None, xlim_end=None, phase='P'):
        '''Plot slowness corrected traces in ad-hoc array, post x-correlation QC check'''
        
        st_plt = self.array_class.timeshifted_stream
        bin_P_time = self.array_class.taup_P_time
        #repeat = self.array_class.repeated_loop
        repeat = False # CHANGE ONCE SAVED IN ARRAY CLASS
        rel_time= self.array_class.relative_time 
        st_plt_beam = self.array_class.optimum_beam
        st_plt_beampws = self.array_class.PW_optimum_beam
        stname= self.array_class.stations.stname
        
        # PLOTTING BEAMPACK VALUE BEAMFORMING
        fig, ax = plt.subplots(len(st_plt)+2, 1, figsize = (10,1.5*len(st_plt)+2), sharex=True)
        #ax[0].set_title(str(self.evname) + ' Traces_%s' %self.array_class.ev_array_gcarc)
        
        if xlim_start == None:
            if phase == 'P':
                xlim_start = self.array_class.taup_P_time - 40
            if phase == 'S':
                xlim_start = self.array_class.taup_S_time - 40
        if xlim_end == None:
            xlim_end = xlim_start + 140

        if repeat == True:
            for i in range (len(st_plt)): 
                ax[i].plot(rel_time, st_plt[i], color='grey', label = stname[i])
                ax[i].set_ylim(-1, 1)            
                ax[i].legend(loc='upper left', handlelength=0, handletextpad=0)
            ax[-2].plot(rel_time, st_plt_beam, color='grey', label='Beam')
            ax[-2].set_xlim(xlim_start, xlim_end)
            ax[-2].set_ylim(max(abs(st_plt_beam))*-1.01, max(abs(st_plt_beam))*1.01)
            ax[-2].legend(loc='upper left', handlelength=0, handletextpad=0)
            ax[-1].plot(rel_time, st_plt_beampws, color='red', label='PW Beam')
            ax[-1].set_xlim(xlim_start, xlim_end)
            ax[-1].set_ylim(max(abs(st_plt_beampws))*-1.01, max(abs(st_plt_beampws))*1.01)
            ax[-1].legend(loc='upper left', handlelength=0, handletextpad=0)
            ax[-1].set_xlabel('Time (s)')
            #plt.show()
            plt.close()
        
        else:
            for i in range (len(st_plt)): 
                ax[i].plot(rel_time, st_plt[i], color='k', label = stname[i])
                ax[i].set_ylim(-1, 1)            
                ax[i].legend(loc='upper left', handlelength=0, handletextpad=0)
            ax[-2].plot(rel_time, st_plt_beam, color='k', label='Beam')
            ax[-2].set_xlim(xlim_start, xlim_end)
            ax[-2].set_ylim(max(abs(st_plt_beam))*-1.01, max(abs(st_plt_beam))*1.01)
            ax[-2].legend(loc='upper left', handlelength=0, handletextpad=0)
            ax[-1].plot(rel_time, st_plt_beampws, color='red', label='PW Beam')
            ax[-1].set_xlim(xlim_start, xlim_end)
            ax[-1].set_ylim(max(abs(st_plt_beampws))*-1.01, max(abs(st_plt_beampws))*1.01)
            ax[-1].legend(loc='upper left', handlelength=0, handletextpad=0)
            ax[-1].set_xlabel('Time (s)')
            #plt.show()
            plt.close()
            
        return fig
    
    def Plain_Vespagram(self, xlim_start=None, xlim_end=None, phase='P'):
        '''Plot normalised vespagram for ad-hoc array'''        
        vespa_grd = self.array_class.vespa_grd
        relative_time = self.array_class.relative_time 
        trim_start = self.array_class.trim_start
        rsample = self.array_class.resample
        bin_P_time = self.array_class.taup_P_time
        bin_pP_time = self.array_class.taup_pP_time
        bin_sP_time = self.array_class.taup_sP_time
        bin_S_time = self.array_class.taup_S_time
        bin_sS_time = self.array_class.taup_sS_time
        slw = self.array_class.slowness_range
            
        if xlim_start == None:
            if phase == 'P':
                xlim_start = (self.array_class.taup_P_time-trim_start)*rsample - 400
            if phase == 'S':
                xlim_start = (self.array_class.taup_S_time-trim_start)*rsample - 400
        if xlim_end == None:
            xlim_end = xlim_start + 1400
        
        vmax = np.max(vespa_grd)
        vmin = np.min(vespa_grd)
        if abs(vmin) >= abs(vmax):
            vmax= abs(vmin)
        
        x_tick_interval = 20   
        x_ticks = np.arange(0, len(vespa_grd[0]), x_tick_interval*rsample)
        relative_time = np.arange(relative_time[0], relative_time[-1]+(x_tick_interval*2), x_tick_interval) 
        relative_time = np.round(relative_time,0)
        relative_time = relative_time[:len(x_ticks)]
        relative_time = [int(t) for t in relative_time]
        
        yaxis_p = np.linspace(0, len(slw), 5)
        yaxis_l = np.round(np.linspace(slw[0], slw[-1], 5),3)
        
        fig, ax1 = plt.subplots(1,1, sharex=True, figsize=(20,5))
        #ax1.set_title(str(outputs.evname) + ' Vespagram_%s' %self.array_class.ev_array_gcarc)
        
        # PLOT
        mesh = ax1.pcolormesh(vespa_grd, cmap = 'seismic', vmin = (-vmax*1.05), vmax = (vmax*1.05), rasterized=True)
        ax1.set_ylabel('Slowness (s/km)', fontsize = 20)
        ax1.set_xlabel('Time (s)', fontsize = 20)
        ax1.set_xticks(x_ticks)
        ax1.set_xticklabels(relative_time, rotation=-45, ha='left', fontsize=16)
        ax1.set_xlim(xlim_start, xlim_end) # turn off for full trace vespagram
        ax1.set_ylim(0, len(vespa_grd))
        ax1.set_yticks(yaxis_p)
        ax1.set_yticklabels(yaxis_l, fontsize=16)
        ax1.scatter(((bin_P_time-trim_start)*rsample), 1, marker = '^', s = 100, color = 'k', label='Modelled Arrivals')
        ax1.scatter(((bin_pP_time-trim_start)*rsample), 1, marker = '^', s = 100, color = 'k')
        ax1.scatter(((bin_sP_time-trim_start)*rsample), 1, marker = '^', s = 100, color = 'k')
        ax1.scatter(((bin_S_time-trim_start)*rsample), 1, marker = '^', s = 100, color = 'k')
        ax1.scatter(((bin_sS_time-trim_start)*rsample), 1, marker = '^', s = 100, color = 'k')
        ax1.tick_params(width=1, length = 8)
        ax1.legend(loc='upper right', fontsize=15)
        fig.colorbar(mesh, ax=ax1)
        plt.close()
        
        return fig 
        
    def Plot_picking_threshold(self, phase='P'):
        '''Plots the system used to define a threshold to pick arrivals (dynamic to each ad-hoc array'''
        
        # Extract variables
        stream = self.array_class.phase_weighted_beams
        slow_index = self.array_class.slowness_index
        rsample = self.array_class.resample
        bin_P_time = self.array_class.taup_P_time
        bin_pP_time = self.array_class.taup_pP_time
        bin_sP_time = self.array_class.taup_sP_time
        bin_S_time = self.array_class.taup_S_time
        bin_sS_time = self.array_class.taup_sS_time
        trim_start = self.array_class.trim_start
        origin_time = self.event.origin_time
        
        stream.normalize()
        
        # Trim data, to only consider peaks in the area of interest (P-sP arrivals)
        if phase == 'P':
            starttime = origin_time + (bin_P_time * 0.98)
            endtime = origin_time + (bin_sP_time * 1.02)
            x_axis_time_addition = int(((bin_P_time * 0.98) - trim_start)*rsample)
            x_axis_end = int(((bin_sP_time * 1.02) - trim_start)*rsample)
        if phase == 'S':
            starttime = origin_time + (bin_S_time * 0.98)
            endtime = origin_time + (bin_sS_time * 1.02)
            x_axis_time_addition = int(((bin_S_time * 0.98) - trim_start)*rsample)
            x_axis_end = int(((bin_sS_time * 1.02) - trim_start)*rsample)
        
        beam_trimmed = deepcopy(stream[slow_index])
        beam_trimmed.trim(starttime, endtime, pad=True, fill_value=0)
        
        envelope_trimmed = obspy.signal.filter.envelope(beam_trimmed.data)
        envelope_trimmed = envelope_trimmed/np.max(envelope_trimmed)
                
        # Setting the threshold ------------------------------------------------------------------        
        def slope(P1, P2):
            # dy/dx
            # (y2 - y1) / (x2 - x1)
            return(P2[1] - P1[1]) / (P2[0] - P1[0])
        
        def y_intercept(P1, slope):
            # y = mx + b
            # b = y - mx
            # b = P1[1] - slope * P1[0]
            return P1[1] - slope * P1[0]
        
        def line_intersect(m1, b1, m2, b2):
            if m1 == m2:
                print ("These lines are parallel!!!")
                return None
            # y = mx + b
            # Set both lines equal to find the intersection point in the x direction
            # m1 * x + b1 = m2 * x + b2
            # m1 * x - m2 * x = b2 - b1
            # x * (m1 - m2) = b2 - b1
            # x = (b2 - b1) / (m1 - m2)
            x = (b2 - b1) / (m1 - m2)
            # Now solve for y -- use either line, because they are equal here
            # y = mx + b
            y = m1 * x + b1
            return x,y
        
        A1 = [0, np.percentile(envelope_trimmed, 0)]
        A2 = [25, np.percentile(envelope_trimmed, 25)]
        B1 = [80, np.percentile(envelope_trimmed, 80)]
        B2 = [100, np.percentile(envelope_trimmed, 100)]
        
        slope_A = slope(A1, A2)
        slope_B = slope(B1, B2)
        
        y_int_A = y_intercept(A1, slope_A)
        y_int_B = y_intercept(B1, slope_B)
        x,y = line_intersect(slope_A, y_int_A, slope_B, y_int_B)
        x = int(np.round(x, 0))
        
        #line A
        xa = []
        ya = []
        xb = []
        yb = []
        
        # Plot percentiles of amplitude data
        percentiles = []
        for i in range (0,101):
            percentiles.append(np.percentile(envelope_trimmed, i))
            xa.append(i)
            xb.append(i)
            ya.append(slope_A*i + y_int_A)
            yb.append(slope_B*i + y_int_B)

        threshold = np.percentile(percentiles, x)
        
        fig, ax = plt.subplots(1, 2, figsize=(10,3), sharey=True)
        
        x_axis = np.arange(0, 101, 1)
        ax[0].plot(x_axis, percentiles, color='k', zorder = 1)
        ax[0].plot(xa, ya, color='green', label='Fitted lines')
        ax[0].plot(xb, yb, color='blue')
        ax[0].scatter(A1[0], A1[1], color='green', s=20, label='Line defining points')
        ax[0].scatter(A2[0], A2[1], color='green', s=20)
        ax[0].scatter(B1[0], B1[1], color='blue', s=20)
        ax[0].scatter(B2[0], B2[1], color='blue', s=20)
        ax[0].scatter(x, y, marker='o', color='grey', label='Intersection', zorder = 3)
        ax[0].axhline(threshold, label='Threshold', color = 'grey', linestyle = ':', zorder = 2)
        ax[0].vlines(x=x, ymin=-0.5, ymax=threshold, color = 'grey', linestyle = '--', zorder = 2)
        ax[0].set_ylabel('Normalised Amplitude', fontsize=12)
        ax[0].set_xlabel('Percentile', fontsize=12)
        ax[0].set_ylim(-0.1,1.05)
        ax[0].set_xlim(0,100)
        ax[0].legend()
        
        percentile_trace = []
        for i in range (len(envelope_trimmed)):
            percentile_trace.append(percentileofscore(envelope_trimmed, envelope_trimmed[i]))

        ax[1].plot(np.arange(0,len(envelope_trimmed),1), envelope_trimmed, color='k')
        ax[1].axhline(threshold, label='Threshold', color = 'grey', linestyle = ':', zorder = 2)
        ax[1].set_xlabel('Time (s)', fontsize=12)
        ax[1].set_xlim(0, len(envelope_trimmed))
        plt.tight_layout()
        plt.close()
        
        #print('THRESHOLD', threshold)
        return fig
    
    def Picking_Vespagram(self, phase='P'):
        '''Plot showing all picked peaks on vespgram/optimum beam for an ad-hoc array'''        
        vespa_grd = self.array_class.vespa_grd
        relative_time = self.array_class.relative_time 
        trim_start = self.array_class.trim_start
        rsample = self.array_class.resample
        bin_P_time = self.array_class.taup_P_time
        bin_pP_time = self.array_class.taup_pP_time
        bin_sP_time = self.array_class.taup_sP_time
        bin_S_time = self.array_class.taup_S_time
        bin_sS_time = self.array_class.taup_sS_time
        slw = self.array_class.slowness_range
        slow_index = self.array_class.slowness_index
        picks_x = self.array_class.picks 
        envelope = self.array_class.PW_optimum_beam_envelope
        stream = self.array_class.phase_weighted_beams
        threshold = self.array_class.picking_threshold
        
        vmax = np.max(vespa_grd)
        vmin = np.min(vespa_grd)
        if abs(vmin) >= abs(vmax):
            vmax= abs(vmin)
        
        if phase == 'P':
            xlim_start = (self.array_class.taup_P_time-trim_start)*rsample - 400
        if phase == 'S':
            xlim_start = (self.array_class.taup_S_time-trim_start)*rsample - 400
        xlim_end = xlim_start + 1400
        
        if phase == 'P':
            x_axis_time_addition = int(((bin_P_time * 0.98) - trim_start)*rsample)
            x_axis_end = int(((bin_sP_time * 1.02) - trim_start)*rsample)
        if phase == 'S':
            x_axis_time_addition = int(((bin_S_time * 0.98) - trim_start)*rsample)
            x_axis_end = int(((bin_sS_time * 1.02) - trim_start)*rsample)
            
        time = np.arange(0, len(vespa_grd[0].data), 1)
        
        x_tick_interval = 20   
        x_ticks = np.arange(0, len(vespa_grd[0]), x_tick_interval*rsample)
        relative_time = np.arange(relative_time[0], relative_time[-1]+(x_tick_interval*2), x_tick_interval) 
        relative_time = np.round(relative_time,0)
        relative_time = relative_time[:len(x_ticks)]
        relative_time = [int(t) for t in relative_time]
        
        yaxis_p = np.linspace(0, len(slw), 5)
        yaxis_l = np.round(np.linspace(slw[0], slw[-1], 5),3)

        slowness_array = []
        for i in range (len(picks_x)):
            slowness_array.append(slow_index)
        
        picking_figure, ax = plt.subplots(2, 1, sharex=True, figsize=(15,10))
        plt.subplots_adjust(hspace=0.2, wspace=0.2) 
    
        # UPPER PLOT
        #ax1.set_title(str(self.evname) + ' Picking_%s' %self.ev_subarray_gcarc)
        ax[0].pcolormesh(vespa_grd, cmap = 'seismic', vmin = (-vmax*1.05), vmax = (vmax*1.05), rasterized=True)
        ax[0].axhline(slow_index, linestyle = '--', color = 'k', zorder = 1, label = 'Optimum Beam')
        ax[0].set_ylabel('Slowness (s/km)', fontsize = 20)
        ax[0].set_xticks(x_ticks)
        ax[0].set_xlim(xlim_start, xlim_end) # turn off for full trace vespagram
        ax[0].set_ylim(np.min(slw), np.max(slw))
        ax[0].set_yticks(yaxis_p)
        ax[0].set_yticklabels(yaxis_l, fontsize=16)
        ax[0].axvline(x_axis_time_addition)
        ax[0].axvline(x_axis_end)
        #ax1.scatter(15,slow_index , marker='>', color='r', s = 100, label='Expected Slowness (s/km)', zorder=2)
        #if absolute_time == 1:
        #    ax[0].set_xticklabels(time_axis, rotation=-45, ha='left', fontsize=16)
        #else:
        ax[0].set_xticklabels(relative_time, rotation=-45, ha='left', fontsize=16)  
        ax[0].scatter(((bin_P_time-trim_start)*rsample), 1, marker = '^', s = 100, color = 'k', label='Modelled Arrivals')
        ax[0].scatter(((bin_pP_time-trim_start)*rsample), 1, marker = '^', s = 100, color = 'k')
        ax[0].scatter(((bin_sP_time-trim_start)*rsample), 1, marker = '^', s = 100, color = 'k')
        ax[0].scatter(((bin_S_time-trim_start)*rsample), 1, marker = '^', s = 100, color = 'k')
        ax[0].scatter(((bin_sS_time-trim_start)*rsample), 1, marker = '^', s = 100, color = 'k')
        if len(picks_x) >= 1:    
            ax[0].scatter(picks_x, slowness_array , color='gold', edgecolor = 'k',linewidth = 1, s=50, label='Picks', zorder = 2)
        ax[0].tick_params(width=1, length = 8)
        ax[0].legend(loc='upper right', fontsize=15)
        
        # LOWER PLOT
        #vmin = np.min(vespa_grd[slow_index])
        vmax = np.max(envelope)
        
        ax[1].plot(time, stream[slow_index], color = 'k', linewidth = '1', linestyle = '--', zorder = 0.5, label = 'Optimum Beam')
        ax[1].plot(time, envelope, linewidth = '2', zorder = 1, color='k', label = 'Envelope')
        ax[1].set_ylabel('Velocity (m/s)', fontsize = 20)
        ax[1].yaxis.set_tick_params(labelsize=16)
        ax[1].set_yticks([-1,0,1])
        ax[1].set_yticklabels(['-1','0','1'], fontsize=16)
        ax[1].set_ylim(np.min(stream[slow_index].data), np.max(stream[slow_index].data))
        ax[1].axvline(x_axis_time_addition)
        ax[1].axvline(x_axis_end)
        ax[1].set_xticklabels(relative_time, rotation=-45, ha='left', fontsize=16) 
        ax[1].set_xlabel('Time (s)',fontsize = 16)
        #ax[1].scatter(15,slow_index , marker='>', color='r', s = 100, label='Expected Slowness (s/km)', zorder = 2 )
        ax[1].scatter(((bin_P_time-trim_start)*rsample), 0.02 , marker = '^', s = 100, color = 'k', label='Modelled Arrivals', transform=ax[1].get_xaxis_transform())
        ax[1].scatter(((bin_pP_time-trim_start)*rsample), 0.02, marker = '^', s = 100, color = 'k', transform=ax[1].get_xaxis_transform())
        ax[1].scatter(((bin_sP_time-trim_start)*rsample), 0.02, marker = '^', s = 100, color = 'k', transform=ax[1].get_xaxis_transform())
        ax[1].scatter(((bin_S_time-trim_start)*rsample), 0.02, marker = '^', s = 100, color = 'k', transform=ax[1].get_xaxis_transform())
        ax[1].scatter(((bin_sS_time-trim_start)*rsample), 0.02, marker = '^', s = 100, color = 'k', transform=ax[1].get_xaxis_transform())
        if len(picks_x) >= 1:   
            ax[1].scatter(picks_x, envelope[picks_x], color='gold', edgecolor = 'k',linewidth = 1, s=50, label='Picks', zorder = 2)
        ax[1].axhline(threshold, linestyle = '--', label='Threshold')
        ax[1].tick_params(width=1, length = 8)
        ax[1].set_ylim([-vmax*1.5, vmax*1.5])
        ax[1].legend(loc='upper right', fontsize=15)
        plt.close()
        return picking_figure
        
    def beampack_vs_calculated_grid(self, phase='P'):
        '''Plot grid of P wave beamforms in a grid search through calculated to best fitting (beampack found) slowness and backazimuth'''
        if phase=='P':
            bin_slowness = self.array_class.taup_slowness              # calculated subarray slowness
        if phase=='S':
            bin_slowness = self.array_class.taup_S_slowness
        beampack_baz = self.array_class.beampack_backazimuth          # beampacked found backazimuth
        beampack_slow = self.array_class.beampack_slowness             # beampack found slowness     
        stream_baz = self.array_class.array_baz                     # calculated backazimuth
        
        #print(bin_slowness, beampack_slow, stream_baz, beampack_baz)
        
        # Round backazimuth to nearest 0.5
        stream_baz = round((stream_baz*2)/2)
        beampack_baz = np.round((beampack_baz*2)/2)
        
        #print(stream_baz, beampack_baz)
        
        # Round slowness to nearest 0.001
        bin_slowness = round(bin_slowness, 3)
        beampack_slow = np.round(beampack_slow, 3)
    
        #print(bin_slowness, beampack_slow)
        
        if beampack_baz > stream_baz:
            backazimuth = np.arange(stream_baz, beampack_baz, 0.5)
        else:
            backazimuth = np.arange(beampack_baz, stream_baz, 0.5)
            backazimuth = np.flip(backazimuth)
        
        if beampack_slow > bin_slowness:
            slowness = np.arange(bin_slowness, beampack_slow, 0.001)
        else:
            slowness = np.arange(beampack_slow, bin_slowness, 0.001)
            slowness = np.flip(slowness)
        
        print('baz', stream_baz, beampack_baz)
        print('slow', bin_slowness, beampack_slow)
        #print(backazimuth, slowness)
    
        if len(backazimuth) < 2 or len(slowness) < 2:
            print('Beampacked and Calculated Results too Similar to Compare')
            return
    
        # VESPAGRAM/PLANE WAVE BEAMFORMING ----------------------------------------------------------------------------
        print('Setting Up Vespagram')
        figure, ax = plt.subplots(len(backazimuth),len(slowness),  sharex=True, sharey=True, figsize=(len(slowness)*5,len(backazimuth)*5))
        plt.subplots_adjust(hspace=0.2, wspace=0.2)
        #figure.tight_layout()
        figure.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', which='both')
        #plt.tick_params(labelcolor='none', which='both', top=False, bottom=True, right=False, left=True)
        plt.xlabel('Slowness (s/km)', labelpad=100, fontsize=24)
        plt.ylabel('Backazimuth (degrees)', labelpad=100, fontsize = 24)
  
        def make_beam(ax, outputs, backazimuth, slowness, phase):
           
            # ====== BEAMFORMING =====

            # Extract variables
            stream = outputs.stream
            stla = outputs.stations.stla
            stlo = outputs.stations.stlo
            bin_centre_la = outputs.array_latitude
            bin_centre_lo = outputs.array_longitude
            bin_P_time = outputs.taup_P_time
            bin_S_time = self.array_class.taup_S_time
            origin_time = outputs.event.origin_time
            trim_start = outputs.trim_start
            rsample = outputs.resample  
            
            stream.detrend(type='demean')
            stream.normalize()
            stream_vespa = stream.copy()
            
            trima = 7
            trimb = 7
            
            if phase == 'P':
                trima_time = origin_time + bin_P_time - trima
                trimb_time = origin_time + bin_P_time + trimb
            if phase == 'S':
                trima_time = origin_time + bin_S_time - trima
                trimb_time = origin_time + bin_S_time + trimb

            stream_vespa.trim(trima_time, trimb_time, pad=True, fill_value=0)
            rel_time = (np.arange(0,len(stream_vespa[0]),1)/rsample) + trim_start
            stream_vespa.normalize()
              
            baz_rad = math.radians(backazimuth)
            baz_dist = [0] * len(stla)
            baz_dist_km = [0] * len(stla)            
            
            for i in range (len(stla)):
                baz_dist[i] = (bin_centre_lo - stlo[i])*math.sin(baz_rad) + (bin_centre_la - stla[i])*math.cos(baz_rad)
                baz_dist_km[i] = degrees2kilometers(baz_dist[i])
            
            dt = [0] * len(baz_dist_km)
            st_tmp =  stream_vespa.copy()
            exp_slw_beam = stream_vespa[0].copy()
            #npts = stream_vespa[0].stats.npts
            #phi = np.zeros((npts,),dtype=complex)
                
                
            for j in range(len(stla)):
                dt[j] = (slowness*baz_dist_km[j])*-1
                tr = st_tmp[j]
                data = np.roll(tr.data,int(np.round(dt[j]*rsample)))
                tr.data = data
                #phi = phi+np.exp(1j*np.angle(hilbert(data)))
                if j == 0:
                    exp_slw_beam.data = tr.data
                elif j > 0 and j < (len(stla)-1):
                    exp_slw_beam.data = exp_slw_beam.data + tr.data  # Beamforming data into one stream trace
                elif j == (len(stla)-1):
                    exp_slw_beam.data = exp_slw_beam.data + tr.data
                    #exp_slw_beam.data = (exp_slw_beam.data/len(stla)) * (np.abs(phi))**4 # TURN ON FOR PHASE WEIGHTING
                    
            exp_slw_beam.normalize()
            
            envelope = obspy.signal.filter.envelope(exp_slw_beam.data)
             
            # Window around modelled P arrival
            #P_end = ((bin_P_time-trim_start)*rsample) + lamda/2
            #P_start = ((bin_P_time-trim_start)*rsample) - lamda/2

            time = np.arange(0, len(exp_slw_beam.data)/10, 0.1)
            relative_time = np.arange(0, len(exp_slw_beam.data)/10, 1)
            vmax = np.max(envelope)
    
            ax.plot(time, exp_slw_beam, color = 'k', linewidth = '1', linestyle = '--', zorder = 0.5, label = 'Beam')
            ax.plot(time, envelope, linewidth = '2', zorder = 1, color='k', label = 'Envelope')
            ax.set_title(r'$\theta$: ' + str(round(backazimuth, 2))+', u: ' + str(round(slowness,3)), fontsize=20)
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
            ax.set_xticks([0,7,14]) # time_axis alternative, provides UTC time
            ax.set_xticklabels([0,7,14])
            ax.tick_params(labelsize=20)
            #ax.set_xlim(0, trima+trimb) # turn off for full trace vespagr
            #ax.tick_params(width=1, length = 8)
            ax.set_ylim([-vmax*1.5, vmax*1.5])
            ax.set_yticklabels([])
            #ax[1].legend(loc='upper right')           
            return
    
        for j in range (len(backazimuth)):
            for k in range (len(slowness)):              
                make_beam(ax[j,k], self.array_class, backazimuth[j], slowness[k], phase)
        
        #plt.show()
        #plt.savefig('beampack_comparison_grids_%s.png' %beampack_baz, dpi=500, bbox_inches='tight')
        return figure
     
    def x_corr(self, phase='P'):
        '''Plots of cross correlation QC tests - each trace in ad-hoc array compared to ad-hoc array beam'''
        st_beam = self.array_class.x_corr_trimmed_PW_beam
        slowness_index = self.array_class.slowness_index
        stream_trim = self.array_class.x_corr_trimmed_traces
        lag = self.array_class.x_corr_lag
        corr = self.array_class.x_corr
        shift = self.array_class.x_corr_shift
        rsample = self.array_class.resample
        bin_S_time = self.array_class.taup_S_time
        bin_sS_time = self.array_class.taup_sS_time
        
        final_beam = self.array_class.phase_weighted_beams[slowness_index]
        
        # Compare each trace to beam, remove outliers
        if phase == 'P':
            trima_time = self.event.origin_time + self.array_class.taup_P_time - 2
            trimb_time = self.event.origin_time + self.array_class.taup_P_time + 7
        if phase == 'S':
            trima_time = self.event.origin_time + self.array_class.taup_S_time - 10
            trimb_time = self.event.origin_time + self.array_class.taup_S_time + 10

        fn_beam = deepcopy(final_beam)
        
        fn_beam.trim(trima_time, trimb_time, pad=True, fill_value=0)
        fn_beam.normalize()
        
        x_axis = np.arange(0, len(st_beam[0]),1)
        
        max_corr = [0] * len(stream_trim)
        for i in range (len(corr)):
            max_corr[i] = np.max(corr[i]) 
            
        # Find trace with best correlation
        trace_index = np.argmax(max_corr)
        #print('trace', trace_index)
        
        '''
        for i in range (len(stream_trim)):
        #for i in range (0,2):
    
            corr_fig, (ax_beam, ax_trace, ax_corr) = plt.subplots(3,1, figsize=(4.8, 4.8))
            ax_beam.plot(x_axis, st_beam[slowness_index])
            ax_beam.set_title('Beam')
            ax_trace.plot(x_axis, stream_trim[i], label = max_corr[i])
            ax_trace.set_title('Trace')
            ax_trace.legend()
            ax_corr.set_title('Correlation')
            ax_corr.plot(lag, corr[i], label=shift[i])
            ax_corr.set_xlabel('Samples')
            ax_corr.legend()
            ax_beam.margins(0, 0.1)
            corr_fig.tight_layout()
            
            parent_dir = os.path.join(file_path, '01_Sub-Arrays')
            fig_name = 'X_corr_%s.png' %outputs.stname[i]
            path = os.path.join(parent_dir, fig_name)
            corr_fig.savefig(path, dpi=500, bbox_inches = 'tight')'''
        
        corr_fig, (ax_beam, ax_trace, ax_corr) = plt.subplots(3,1, figsize=(5.2, 5.2))
        ax_beam.plot(x_axis, st_beam[slowness_index], color='grey', linewidth = 1.5, linestyle = ':', label = 'Before X-Corr', zorder = 2)
        ax_beam.plot(x_axis, fn_beam, color='k', linewidth=1.5, label = 'After X-Corr', zorder = 1)
        ax_beam.plot(-1,-1, color = 'purple', linewidth=1.5, label='Example Trace')
        #ax_beam.set_title('Beam')
        ax_beam.set_ylim(-1.1,1.1)
        ax_beam.spines['bottom'].set_visible(False)
        ax_beam.spines['top'].set_visible(False)
        ax_beam.spines['right'].set_visible(False)
        ax_beam.axes.get_xaxis().set_visible(False)
        ax_beam.spines['left'].set_bounds(-1,1)
        ax_beam.text(0.015, 0.85, 'Beam', fontsize = 12, transform=ax_beam.transAxes)
        ax_beam.legend(frameon=False, loc='upper right', bbox_to_anchor=(1, 1.1))
        
        for i in range (len(stream_trim)):
            if trace_index != i:
                if shift[i] > 5 or max_corr[i] < 0.3:
                    ax_trace.plot(x_axis, stream_trim[i], color = 'grey',  linestyle = ':', linewidth=1, zorder=1)
                    #ax_trace.set_title('Trace')
                    ax_trace.set_ylim(-1.1,1.1)
                    ax_trace.set_xlim(0,90)
                    ax_trace.spines['bottom'].set_visible(False)
                    ax_trace.spines['top'].set_visible(False)
                    ax_trace.spines['right'].set_visible(False)
                    ax_trace.axes.get_xaxis().set_visible(False)
                    ax_trace.spines['left'].set_bounds(-1,1)
                    ax_trace.text(0.015, 0.85, 'Trace', fontsize = 12, transform=ax_trace.transAxes)
               
                else:
                    ax_trace.plot(x_axis, stream_trim[i], color = 'k', linewidth=0.2, zorder=1)
                    #ax_trace.set_title('Trace')
                    ax_trace.set_ylim(-1.1,1.1)
                    ax_trace.set_xlim(0,90)
                    ax_trace.spines['bottom'].set_visible(False)
                    ax_trace.spines['top'].set_visible(False)
                    ax_trace.spines['right'].set_visible(False)
                    ax_trace.axes.get_xaxis().set_visible(False)
                    ax_trace.spines['left'].set_bounds(-1,1)
                    ax_trace.text(0.015, 0.85, 'Traces', fontsize = 12, transform=ax_trace.transAxes)                                
            
            if trace_index == i:
                if shift[i] > 0.5 or max_corr[i] < 0.3:
                    ax_trace.plot(x_axis, stream_trim[i], color = 'purple', linewidth=1.5, zorder=2)
                    label = shift[i]/rsample
                    ax_corr.plot(lag, corr[i], label='Shift: %s s' %label, color='purple')  
                    ax_corr.plot(-50,0, label = 'Corr coeff.: %s' %np.round(max_corr[i],2))
                else:
                    ax_trace.plot(x_axis, stream_trim[i], color = 'purple', linewidth=1.5, zorder=2)
                    label = shift[i]/rsample
                    ax_corr.plot(lag, corr[i], label='Shift: %s s' %label, color='purple')  
                    ax_corr.plot(-50,0, label = 'Corr coeff.: %s' %np.round(max_corr[i],2))
                    
                #ax_trace.legend(handlelength=0, frameon=False, loc='upper right')
            
                #ax_corr.set_title('Correlation')
                        
                #ax_corr.set_xlabel('Samples')
                ax_corr.legend(handlelength=0, frameon=False, loc='upper right', bbox_to_anchor=(1, 1.1))
                ax_corr.set_ylim(-1.1,1.1)
                ax_corr.set_xticks([-44,-34])
                ax_corr.set_xticklabels([0,'1 second'])
                ax_corr.set_xlim(-44,34)
                ax_corr.spines['bottom'].set_bounds(-44,-34)
                ax_corr.spines['top'].set_visible(False)
                ax_corr.spines['right'].set_visible(False)
                #ax_corr.axes.get_xaxis().set_visible(False)
                ax_corr.spines['left'].set_bounds(-1,1)
                ax_corr.text(0.015, 0.85, 'Correlation', fontsize = 12, transform=ax_corr.transAxes)
            
        ax_beam.margins(0, 0.1)
        corr_fig.tight_layout()        
        plt.close()          
        return corr_fig

    def QC_Vespagram(self, phase='P'):
        core_centre_x = self.array_class.vespa_QC_ccx
        core_centre_y = self.array_class.vespa_QC_ccy
        core_centre_std = self.array_class.vespa_QC_cc_std
        no_points = self.array_class.vespa_QC_npts
        slowness_index = self.array_class.slowness_index
        vespa_grd = self.array_class.vespa_grd
        slow = self.array_class.slowness_range
        core_centre_mean = self.array_class.vespa_QC_cc_mean
        cores = self.array_class.vespa_QC_cores
        rsample = self.array_class.resample
        
        abs_vespa_grd = np.abs(vespa_grd)
    
        figure = plt.figure()
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(core_centre_x))]
        
        yaxis_p = np.linspace(0, len(slow), 5)
        yaxis_l = np.round(np.linspace(slow[0], slow[-1], 5),3)
        
        for i in range (len(cores)):
            x_val = [(x[0]*5)/rsample for x in cores[i]] # into relative time
            y_val = [x[1] for x in cores[i]]
            plt.scatter(x_val, y_val, facecolor=colors[i], edgecolor='k')
    
        #plt.title(str(self.evname) + ' Vespagram_QC_%s' %self.ev_subarray_gcarc)
        for i in range (len(core_centre_x)):
            plt.scatter(core_centre_x[i]*5/rsample, core_centre_y[i], color = 'r', s=no_points[i], marker = 'x')
        plt.scatter(0, 0, color = 'r', s=no_points[i], label = core_centre_std, marker = 'x')
        plt.axhline(slowness_index, color = 'k', linestyle = '--', label = slowness_index)
        plt.axhline(slowness_index+6, color = 'g', linestyle = ':', label = slowness_index+6)
        plt.axhline(slowness_index-6, color = 'g', linestyle = ':', label = slowness_index-6)
        plt.axhline(core_centre_mean, color = 'grey', linestyle = ':', label = core_centre_mean)
        plt.xlabel('Relative Time (s)')
        plt.ylabel('Tested Slownesses')
        plt.xlim(0, len(abs_vespa_grd[0]))
        plt.ylim(0, len(slow))
        #ax.set_yticks(yaxis_p)
        #ax.set_yticklabels(yaxis_l, fontsize=16)
        plt.legend()
        #plt.show()
        plt.close()
        return figure
    
    def Final_Picks(self, phase='P'):
        '''Plot final picks post phase ID function, vespagram and beam'''
        vespa_grid = self.array_class.vespa_grd
        relative_time = self.array_class.relative_time 
        trim_start = self.array_class.trim_start
        rsample = self.array_class.resample
        subarray_P_time = self.array_class.taup_P_time
        subarray_pP_time = self.array_class.taup_pP_time
        subarray_sP_time = self.array_class.taup_sP_time
        subarray_S_time = self.array_class.taup_S_time
        subarray_sS_time = self.array_class.taup_sS_time
        slow = self.array_class.slowness_range
        slowness_index = self.array_class.slowness_index
        threshold_envelope_picks = self.array_class.phase_id_picks 
        envelope = self.array_class.PW_optimum_beam_envelope
        stream = self.array_class.phase_weighted_beams
        threshold = self.array_class.picking_threshold
        if phase == 'P':
            dt_pP_TE = self.array_class.dt_pP_P
            dt_sP_TE = self.array_class.dt_sP_P
        if phase == 'S':
            dt_sS_TE = self.array_class.dt_sS_S
        
        x_tick_interval = 20   
        x_ticks = np.arange(0, len(vespa_grid[0]), x_tick_interval*rsample)
        relative_time = np.arange(relative_time[0], relative_time[-1]+(x_tick_interval*2), x_tick_interval) 
        relative_time = np.round(relative_time,0)
        relative_time = relative_time[:len(x_ticks)]
        relative_time = [int(t) for t in relative_time]
       
        yaxis_p = np.linspace(0, len(slow), 5)
        yaxis_l = np.round(np.linspace(slow[0], slow[-1], 5),3)
        
        Vespagram_FINAL_PICKS_figure, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(30,5))
        plt.subplots_adjust(hspace=0.2, wspace=0.1)
       
        vmax = np.max(vespa_grid)
        vmin = np.min(vespa_grid)
        if abs(vmin) >= abs(vmax):
            vmax= abs(vmin)
            
        ax.axhline(slowness_index, color = 'k', linestyle = ':')
        mesh = ax.pcolormesh(vespa_grid, cmap = 'seismic', vmin = (-vmax*1.05), vmax = (vmax*1.05))
        ax.scatter(threshold_envelope_picks, (np.ones(len(threshold_envelope_picks))*slowness_index), facecolor = 'gold', edgecolor = 'k', linewidths = 1, s=50, label='Final Picks', vmin=0, vmax=1)
        ax.set_ylabel('Slowness (s/km)', fontsize = 20)
        ax.set_xlabel('Time (s)', fontsize = 20)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(relative_time, rotation=-45, ha='left', fontsize=16)   
        ax.set_yticks(yaxis_p)
        ax.set_yticklabels(yaxis_l, fontsize=16)
        if phase == 'P':
            ax.set_xlim(((subarray_P_time-trim_start)*rsample)*0.8, ((subarray_sP_time-trim_start)*rsample)*1.2) # turn off for full trace vespagram 
        if phase == 'S':
            ax.set_xlim(((subarray_S_time-trim_start)*rsample)*0.8, ((subarray_sS_time-trim_start)*rsample)*1.2) # turn off for full trace vespagram 
        ax.set_ylim(0, len(vespa_grid))
        ax.scatter(((subarray_P_time-trim_start)*rsample), 1, marker = '^', s = 100, color = 'k', label='Modelled Arrivals')
        ax.scatter(((subarray_pP_time-trim_start)*rsample), 1, marker = '^', s = 100, color = 'k')
        ax.scatter(((subarray_sP_time-trim_start)*rsample), 1, marker = '^', s = 100, color = 'k')
        ax.scatter(((subarray_S_time-trim_start)*rsample), 1, marker = '^', s = 100, color = 'k')
        ax.scatter(((subarray_sS_time-trim_start)*rsample), 1, marker = '^', s = 100, color = 'k')
        ax.tick_params(width=1, length = 8)
        ax.legend(loc='upper right', fontsize=15)
        if phase == 'P':
            ax.text(0.01, 0.93, 'pP-P: %s seconds' %(dt_pP_TE), fontsize = 16, transform=ax.transAxes)
            ax.text(0.01, 0.86, 'sP-P: %s seconds' %(dt_sP_TE), fontsize = 16, transform=ax.transAxes)
        if phase == 'S':
            ax.text(0.01, 0.93, 'sS-S: %s seconds' %(dt_sS_TE), fontsize = 16, transform=ax.transAxes)
        #ax.set_title('Final_Picks_%s' %subarray.ev_subarray_gcarc)
       
        Vespagram_FINAL_PICKS_figure.colorbar(mesh, ax=ax)
        plt.close()
        
        return Vespagram_FINAL_PICKS_figure
        
    def Beams_and_Beampacking(self, phase='P'):
        '''Plot beampack polar plot and calculated vs beampack based optimum beams'''
        fig = plt.figure(figsize=(20,10)) 
        grid = plt.GridSpec(2, 2, wspace=0.2, hspace=0.2)
        
        top = fig.add_subplot(grid[0,1])
        bottom = fig.add_subplot(grid[1,1])
        main = fig.add_subplot(grid[:, 0], polar=True)

        self.BP_polar_plot(main, fig, phase)

        #exp_slw_beam = self.array_class.subarray_calculated_PW_beam 
        trim_start = self.array_class.trim_start
        bin_P_time = self.array_class.taup_P_time
        bin_S_time = self.array_class.taup_S_time
        bin_sS_time = self.array_class.taup_sS_time
        rel_time= self.array_class.relative_time 
        st_plt_beampws = self.array_class.PW_optimum_beam 
        if phase == 'P':
            subarray_slowness = self.array_class.taup_slowness
        if phase =='S':
            subarray_slowness = self.array_class.taup_S_slowness
        slow_range = self.array_class.slowness_range
        subarray_baz = self.array_class.array_baz
        slowness_index = self.array_class.slowness_index
        rsample = self.array_class.resample

        # Calculate 'calculated' slowness and backazimuth optimum beam
        def make_beam(outputs, backazimuth, slowness):
           
            # ====== BEAMFORMING =====

            # Extract variables
            stream = outputs.stream
            stla = outputs.stations.stla
            stlo = outputs.stations.stlo
            bin_centre_la = outputs.array_latitude
            bin_centre_lo = outputs.array_longitude
            bin_P_time = outputs.taup_P_time
            origin_time = outputs.event.origin_time
            trim_start = outputs.trim_start
            rsample = outputs.resample  
            
            stream.detrend(type='demean')
            stream.normalize()
            stream_vespa = stream.copy()
            
            if phase == 'P':
                trima_time = origin_time + bin_P_time - 40
                trimb_time = origin_time + bin_P_time + 100
            if phase == 'S':
                trima_time = origin_time + bin_S_time - 40
                trimb_time = origin_time + bin_S_time + 100

            stream_vespa.trim(trima_time, trimb_time, pad=True, fill_value=0)
            stream_vespa.normalize()
            
            if phase == 'P':
                rel_time = (np.arange(0,len(stream_vespa[0]),1)/rsample) + (bin_P_time - 40)
            if phase == 'S':
                rel_time = (np.arange(0,len(stream_vespa[0]),1)/rsample) + (bin_S_time - 40)

            baz_rad = math.radians(backazimuth)
            baz_dist = [0] * len(stla)
            baz_dist_km = [0] * len(stla)            

            for i in range (len(stla)):
                baz_dist[i] = (bin_centre_lo - stlo[i])*math.sin(baz_rad) + (bin_centre_la - stla[i])*math.cos(baz_rad)
                baz_dist_km[i] = degrees2kilometers(baz_dist[i])
            
            dt = [0] * len(baz_dist_km)
            st_tmp =  stream_vespa.copy()
            exp_slw_beam = stream_vespa[0].copy()
            npts = stream_vespa[0].stats.npts
            phi = np.zeros((npts,),dtype=complex)    
 
            for j in range(len(stla)):
                dt[j] = (slowness*baz_dist_km[j])*-1
                tr = st_tmp[j]
                data = np.roll(tr.data,int(np.round(dt[j]*rsample)))
                tr.data = data
                phi = phi+np.exp(1j*np.angle(hilbert(data)))
                if j == 0:
                    exp_slw_beam.data = tr.data
                elif j > 0 and j < (len(stla)-1):
                    exp_slw_beam.data = exp_slw_beam.data + tr.data  # Beamforming data into one stream trace
                elif j == (len(stla)-1):
                    exp_slw_beam.data = exp_slw_beam.data + tr.data
                    exp_slw_beam.data = (exp_slw_beam.data/len(stla)) * (np.abs(phi))**4 # TURN ON FOR PHASE WEIGHTING
      
            exp_slw_beam.normalize()
            return exp_slw_beam, rel_time
        
        exp_slw_beam, exp_rel_time = make_beam(self.array_class, subarray_baz, subarray_slowness)

        # PLOTTING BEAMPACK VALUE BEAMFORMING
        if phase == 'P':
            xlim = bin_P_time - 40
            xlim_interval = xlim + 140
        if phase == 'S':
            xlim = bin_S_time - 40
            xlim_interval = xlim + 140
        
        top.scatter(0,0, marker = 'o', facecolor = 'none', edgecolor = 'red', s=50, label = 'Expected Fit Beam')
        top.plot(exp_rel_time, exp_slw_beam, color='k')
        top.yaxis.set_tick_params(labelsize=14)
        top.xaxis.set_tick_params(labelsize=14)
        top.set_xlim(xlim, xlim_interval)
        top.set_ylim(-1.05, 1.05)
        top.legend(loc='upper right', fontsize=12)
        top.set_xlabel('Time (s)', fontsize=14)
    
        bottom.scatter(0,0, marker = 'o', color = 'r', s=50, label = 'Best Fit Beam')    
        bottom.plot(rel_time, st_plt_beampws, color='k')
        bottom.yaxis.set_tick_params(labelsize=14)
        bottom.xaxis.set_tick_params(labelsize=14)
        bottom.set_xlim(xlim, xlim_interval)
        bottom.set_ylim(-1.05,1.05)
        bottom.legend(loc='upper right', fontsize=12)  #, handlelength=0, handletextpad=0
        bottom.set_xlabel('Time (s)', fontsize=14)
        plt.close()
    
        return fig
        
    def Vespagrams(self, phase='P'):
        '''Plot vespagram, optimum beam, all picks and final phase ID'd picks'''
        vespa_grd = self.array_class.vespa_grd
        stream = self.array_class.phase_weighted_beams
        relative_time = self.array_class.relative_time 
        trim_start = self.array_class.trim_start
        rsample = self.array_class.resample
        bin_P_time = self.array_class.taup_P_time
        bin_pP_time = self.array_class.taup_pP_time
        bin_sP_time = self.array_class.taup_sP_time
        bin_S_time = self.array_class.taup_S_time
        bin_sS_time = self.array_class.taup_sS_time
        slow = self.array_class.slowness_range
        slow_index = self.array_class.slowness_index
        picks_x = self.array_class.picks
        threshold = self.array_class.picking_threshold
        if phase == 'P':
            dt_pP_TE = self.array_class.dt_pP_P
            dt_sP_TE = self.array_class.dt_sP_P
        if phase == 'S':
            dt_sS_TE = self.array_class.dt_sS_S
        slw = self.array_class.slowness_range
        threshold = self.array_class.picking_threshold
        envelope = self.array_class.PW_optimum_beam_envelope
        final_picks = self.array_class.phase_id_picks

        if phase == 'P':
            x_axis_time_addition = int(((bin_P_time * 0.98) - trim_start)*rsample)
            x_axis_end = int(((bin_sP_time * 1.02) - trim_start)*rsample)
        if phase == 'S':
            x_axis_time_addition = int(((bin_S_time * 0.98) - trim_start)*rsample)
            x_axis_end = int(((bin_sS_time * 1.02) - trim_start)*rsample)
        time = np.arange(0, len(vespa_grd[0].data), 1)

        slowness_array = []
        for i in range (len(picks_x)):
            slowness_array.append(slow_index)
        
        slowness_array_fp = []
        for i in range (len(final_picks)):
            slowness_array_fp.append(slow_index)
        
        x_tick_interval = 20
        x_ticks = np.arange(0, len(vespa_grd[0]), x_tick_interval*rsample)
        relative_time = np.arange(relative_time[0], relative_time[-1]+(x_tick_interval*2), x_tick_interval) 
        relative_time = np.round(relative_time,0)
        relative_time = relative_time[:len(x_ticks)]
        relative_time = [int(t) for t in relative_time]
        
        yaxis_p = np.linspace(0, len(slow), 5)
        yaxis_l = np.round(np.linspace(slow[0], slow[-1], 5),2)
        
        vmax = np.max(vespa_grd)
        vmin = np.min(vespa_grd)
        if abs(vmin) >= abs(vmax):
            vmax= abs(vmin)
        
        xlim = x_axis_time_addition - 100
        xlim_interval = x_axis_end + 100
        
        picking_figure, ax = plt.subplots(2, 1, sharex=True, figsize=(8,10))
        plt.subplots_adjust(hspace=0.2, wspace=0.2) 
    
        # UPPER PLOT
        #ax1.set_title(str(self.evname) + ' Picking_%s' %self.ev_subarray_gcarc)
        ax[0].pcolormesh(vespa_grd, cmap = 'seismic', vmin = (-vmax*1.05), vmax = (vmax*1.05), rasterized=True)
        ax[0].axhline(slow_index, linestyle = '--', color = 'k', zorder = 1, label = 'Optimum Beam')
        ax[0].set_ylabel('Slowness (s/km)', fontsize = 20)
        ax[0].set_xticks(x_ticks)
        ax[0].set_xlim(xlim, xlim_interval) # turn off for full trace vespagram
        ax[0].set_ylim(np.min(slw), np.max(slw))
        ax[0].set_yticks(yaxis_p)
        ax[0].set_yticklabels(yaxis_l, fontsize=16)
        ax[0].axvline(x_axis_time_addition)
        ax[0].axvline(x_axis_end)
        #ax1.scatter(15,slow_index , marker='>', color='r', s = 100, label='Expected Slowness (s/km)', zorder=2)
        #if absolute_time == 1:
        #    ax[0].set_xticklabels(time_axis, rotation=-45, ha='left', fontsize=16)
        #else:
        ax[0].set_xticklabels(relative_time, rotation=-45, ha='left', fontsize=16)  
        ax[0].scatter(((bin_P_time-trim_start)*rsample), 0.02 , marker = '^', s = 100, color = 'k', label='Modelled Arrivals', transform=ax[0].get_xaxis_transform())
        ax[0].scatter(((bin_pP_time-trim_start)*rsample), 0.02, marker = '^', s = 100, color = 'k', transform=ax[0].get_xaxis_transform())
        ax[0].scatter(((bin_sP_time-trim_start)*rsample), 0.02, marker = '^', s = 100, color = 'k', transform=ax[0].get_xaxis_transform())
        ax[0].scatter(((bin_S_time-trim_start)*rsample), 0.02, marker = '^', s = 100, color = 'k', transform=ax[0].get_xaxis_transform())
        ax[0].scatter(((bin_sS_time-trim_start)*rsample), 0.02, marker = '^', s = 100, color = 'k', transform=ax[0].get_xaxis_transform())
        if len(picks_x) >= 1:    
            ax[0].scatter(picks_x, slowness_array , color='gold', edgecolor = 'k',linewidth = 1, s=50, label='Picks', zorder = 2)
        ax[0].scatter(final_picks, slowness_array_fp, color='green', edgecolor = 'green',linewidth = 1, s=100, label='Final Picks', zorder = 1)
        ax[0].tick_params(width=1, length = 8)
        #ax[0].legend(loc='upper right', fontsize=15)
        ax[0].scatter(final_picks, (np.ones(len(final_picks))*slow_index), facecolor = 'gold', edgecolor = 'k', linewidths = 1, s=50, label='Final Picks', vmin=0, vmax=1)
        if phase == 'P':
            ax[0].text(0.01, 0.92, 'pP-P: %s seconds' %(dt_pP_TE), fontsize = 16, transform=ax[0].transAxes)
            ax[0].text(0.01, 0.85, 'sP-P: %s seconds' %(dt_sP_TE), fontsize = 16, transform=ax[0].transAxes)
        if phase == 'S':
            ax[0].text(0.01, 0.92, 'sS-S: %s seconds' %(dt_sS_TE), fontsize = 16, transform=ax[0].transAxes)
            
        # LOWER PLOT
        #vmin = np.min(vespa_grd[slow_index])
        vmax = np.max(envelope)
        
        ax[1].plot(time, stream[slow_index], color = 'k', linewidth = '1', linestyle = '--', zorder = 0.5, label = 'Optimum Beam')
        ax[1].plot(time, envelope, linewidth = '2', zorder = 1, color='k', label = 'Envelope')
        ax[1].set_ylabel('Velocity (m/s)', fontsize = 20)
        ax[1].set_xlabel('Time (s)', fontsize = 20)
        #ax[1].yaxis.set_tick_params(labelsize=16)
        ax[1].set_yticks([-1,0,1])
        ax[1].set_yticklabels(['-1','0','1'], fontsize=16)
        ax[1].set_ylim(np.min(stream[slow_index].data), np.max(stream[slow_index].data))
        ax[1].axvline(x_axis_time_addition)
        ax[1].axvline(x_axis_end)
        #ax[1].scatter(15,slow_index , marker='>', color='r', s = 100, label='Expected Slowness (s/km)', zorder = 2 )
        ax[1].scatter(((bin_P_time-trim_start)*rsample), 0.02 , marker = '^', s = 100, color = 'k', label='Modelled Arrivals', transform=ax[1].get_xaxis_transform())
        ax[1].scatter(((bin_pP_time-trim_start)*rsample), 0.02, marker = '^', s = 100, color = 'k', transform=ax[1].get_xaxis_transform())
        ax[1].scatter(((bin_sP_time-trim_start)*rsample), 0.02, marker = '^', s = 100, color = 'k', transform=ax[1].get_xaxis_transform())
        ax[1].scatter(((bin_S_time-trim_start)*rsample), 0.02, marker = '^', s = 100, color = 'k', transform=ax[1].get_xaxis_transform())
        ax[1].scatter(((bin_sS_time-trim_start)*rsample), 0.02, marker = '^', s = 100, color = 'k', transform=ax[1].get_xaxis_transform())  
        if len(picks_x) >= 1:   
            ax[1].scatter(picks_x, envelope[picks_x], color='gold', edgecolor = 'k',linewidth = 1, s=50, label='Picks', zorder = 2)
        ax[1].scatter(final_picks, envelope[final_picks], color='green', edgecolor = 'green',linewidth = 1, s=100, label='Final Picks', zorder = 1)
        ax[1].axhline(threshold, linestyle = '--', label='Threshold')
        ax[1].tick_params(width=1, length = 8)
        ax[1].yaxis.set_tick_params(labelsize=16)
        ax[1].set_xticklabels(relative_time, fontsize=16)
        ax[1].set_ylim([-vmax*1.5, vmax*1.5])
        ax[1].legend(fontsize=15, bbox_to_anchor=(1.45, 2.23))
        plt.close()        
        return picking_figure
    
    def Vespagrams_and_QC(self, phase='P'):
        '''Plot vepsgrams and vespagram QC test'''
        vespa_grd = self.array_class.vespa_grd
        stream = self.array_class.phase_weighted_beams
        relative_time = self.array_class.relative_time 
        trim_start = self.array_class.trim_start
        rsample = self.array_class.resample
        bin_P_time = self.array_class.taup_P_time
        bin_pP_time = self.array_class.taup_pP_time
        bin_sP_time = self.array_class.taup_sP_time
        bin_S_time = self.array_class.taup_S_time
        bin_sS_time = self.array_class.taup_sS_time
        slow = self.array_class.slowness_range
        slow_index = self.array_class.slowness_index
        final_picks = self.array_class.phase_id_picks
        envelope = self.array_class.PW_optimum_beam_envelope
        picks_x = self.array_class.picks
        threshold = self.array_class.picking_threshold
        if phase == 'P':
            dt_pP_TE = self.array_class.dt_pP_P
            dt_sP_TE = self.array_class.dt_sP_P
        if phase == 'S':
            dt_sS_TE = self.array_class.dt_sS_S
        slow = self.array_class.slowness_range
        threshold = self.array_class.picking_threshold
        
        core_centre_x = self.array_class.vespa_QC_ccx
        core_centre_y = self.array_class.vespa_QC_ccy
        core_centre_std = self.array_class.vespa_QC_cc_std
        no_points = self.array_class.vespa_QC_npts
        core_centre_mean = self.array_class.vespa_QC_cc_mean
        cores = self.array_class.vespa_QC_cores
        
        if phase == 'P':
            x_axis_time_addition = int(((bin_P_time * 0.98) - trim_start)*rsample)
            x_axis_end = int(((bin_sP_time * 1.02) - trim_start)*rsample)
        if phase == 'S':
            x_axis_time_addition = int(((bin_S_time * 0.98) - trim_start)*rsample)
            x_axis_end = int(((bin_sS_time * 1.02) - trim_start)*rsample)
        time = np.arange(0, len(vespa_grd[0].data), 1)

        slowness_array = []
        for i in range (len(picks_x)):
            slowness_array.append(slow_index)
        
        x_tick_interval = 20
        x_ticks = np.arange(0, len(vespa_grd[0]), x_tick_interval*rsample)
        relative_time = np.arange(relative_time[0], relative_time[-1]+(x_tick_interval*2), x_tick_interval) 
        relative_time = np.round(relative_time,0)
        relative_time = relative_time[:len(x_ticks)]
        relative_time = [int(t) for t in relative_time]
        
        yaxis_p = np.linspace(0, len(slow), 5)
        yaxis_l = np.round(np.linspace(slow[0], slow[-1], 5),3)
        
        vmax = np.max(vespa_grd)
        vmin = np.min(vespa_grd)
        if abs(vmin) >= abs(vmax):
            vmax= abs(vmin)
        
        xlim = x_axis_time_addition - 100
        xlim_interval = x_axis_end + 100
        
        picking_figure, ax = plt.subplots(3, 1, sharex=True,figsize=(6,10))
        plt.subplots_adjust(hspace=0.1, wspace=0.2) 
        
        abs_vespa_grd = np.abs(vespa_grd)
    
        # UPPER PLOT
        #ax1.set_title(str(self.evname) + ' Picking_%s' %self.ev_subarray_gcarc)
        ax[0].pcolormesh(vespa_grd, cmap = 'seismic', vmin = (-vmax*1.05), vmax = (vmax*1.05), rasterized=True)
        ax[0].axhline(slow_index, linestyle = '--', color = 'k', zorder = 1, label = 'Optimum Beam')
        ax[0].set_ylabel('Slowness (s/km)', fontsize = 20)
        ax[0].set_xticks(x_ticks)
        ax[0].set_xlim(xlim, xlim_interval) # turn off for full trace vespagram
        ax[0].set_ylim(np.min(slow), np.max(slow))
        ax[0].set_yticks(yaxis_p)
        ax[0].set_yticklabels(yaxis_l, fontsize=16)
        ax[0].axvline(x_axis_time_addition)
        ax[0].axvline(x_axis_end)
        #ax1.scatter(15,slow_index , marker='>', color='r', s = 100, label='Expected Slowness (s/km)', zorder=2)
        ax[0].set_xticklabels(relative_time, rotation=-45, ha='left', fontsize=16)  
        ax[0].scatter(((bin_P_time-trim_start)*rsample), 0.02 , marker = '^', s = 100, color = 'k', label='Modelled Arrivals', transform=ax[0].get_xaxis_transform())
        ax[0].scatter(((bin_pP_time-trim_start)*rsample), 0.02, marker = '^', s = 100, color = 'k', transform=ax[0].get_xaxis_transform())
        ax[0].scatter(((bin_sP_time-trim_start)*rsample), 0.02, marker = '^', s = 100, color = 'k', transform=ax[0].get_xaxis_transform())
        if len(picks_x) >= 1:    
            ax[0].scatter(picks_x, slowness_array , color='gold', edgecolor = 'k',linewidth = 1, s=50, label='Picks', zorder = 2)
        ax[0].scatter(final_picks, (np.ones(len(final_picks))*slow_index), facecolor = 'green', edgecolor = 'green', linewidths = 1, s=120, label='Final Picks', vmin=0, vmax=1)
        ax[0].tick_params(width=1, length = 8)
        #ax[0].legend(fontsize=15, bbox_to_anchor=(1.45, 2.23))
        if phase == 'P':
            ax[0].text(0.01, 0.92, 'pP-P: %s seconds' %(dt_pP_TE), fontsize = 16, transform=ax[0].transAxes)
            ax[0].text(0.01, 0.82, 'sP-P: %s seconds' %(dt_sP_TE), fontsize = 16, transform=ax[0].transAxes)
        if phase == 'S':
            ax[0].text(0.01, 0.92, 'sS-S: %s seconds' %(dt_sS_TE), fontsize = 16, transform=ax[0].transAxes)
           
        # LOWER PLOT
        vmax = np.max(envelope)
        ax[1].plot(time, stream[slow_index], color = 'k', linewidth = '1', linestyle = '--', zorder = 0.5, label = 'Optimum Beam')
        ax[1].plot(time, envelope, linewidth = '2', zorder = 1, color='k', label = 'Envelope')
        ax[1].set_ylabel('Velocity (m/s)', fontsize = 20)
        ax[1].yaxis.set_tick_params(labelsize=16)
        ax[1].set_yticks([-1,0,1])
        ax[1].set_yticklabels(['-1','0','1'], fontsize=16)
        #ax[1].set_ylim(np.min(stream[slow_index].data), np.max(stream[slow_index].data))
        ax[1].axvline(x_axis_time_addition)
        ax[1].axvline(x_axis_end)
        ax[1].set_xticklabels(relative_time, rotation=-45, ha='left', fontsize=16) 
        #ax[1].scatter(15,slow_index , marker='>', color='r', s = 100, label='Expected Slowness (s/km)', zorder = 2 )
        ax[1].scatter(((bin_P_time-trim_start)*rsample), 0.02 , marker = '^', s = 100, color = 'k', label='Modelled Arrivals', transform=ax[1].get_xaxis_transform())
        ax[1].scatter(((bin_pP_time-trim_start)*rsample), 0.02, marker = '^', s = 100, color = 'k', transform=ax[1].get_xaxis_transform())
        ax[1].scatter(((bin_sP_time-trim_start)*rsample), 0.02, marker = '^', s = 100, color = 'k', transform=ax[1].get_xaxis_transform())
        ax[1].scatter(((bin_S_time-trim_start)*rsample), 0.02, marker = '^', s = 100, color = 'k', transform=ax[1].get_xaxis_transform())
        ax[1].scatter(((bin_sS_time-trim_start)*rsample), 0.02, marker = '^', s = 100, color = 'k', transform=ax[1].get_xaxis_transform())
        if len(picks_x) >= 1:   
            ax[1].scatter(picks_x, envelope[picks_x], color='gold', edgecolor = 'k',linewidth = 1, s=50, label='Picks', zorder = 2)
        #ax[1].scatter(final_picks, envelope[final_picks], facecolor = 'green', edgecolor = 'green', linewidths = 1, s=120, label='Final Picks', vmin=0, vmax=1)
        ax[1].axhline(threshold, linestyle = '--', label='Threshold')
        ax[1].tick_params(width=1, length = 8)
        ax[1].set_ylim([-vmax*1.5, vmax*1.5])
        ax[1].legend(fontsize=15, bbox_to_anchor=(1.6, 2.23))
        #plt.close()
        
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(core_centre_x))]
        
        for i in range (len(cores)):
            #print(cores[i])
            x_val = [x[0]*5/rsample for x in cores[i]]
            y_val = [x[1] for x in cores[i]]
            ax[2].scatter(x_val, y_val, facecolor=colors[i], edgecolor='k', s=100, marker='X')
        ax[2].scatter(0, 0, facecolor=colors[i], edgecolor='k', s=100, marker='X', label = 'Largest Vespagram Peaks')
        
        for i in range(len(core_centre_x)):
            core_centre_x[i] = (core_centre_x[i]*5)/rsample
        
        #print(no_points)
        #ax[0].title(str(self.evname) + ' Vespagram_QC_%s' %self.ev_subarray_gcarc)
            ax[2].scatter(core_centre_x[i], core_centre_y[i], c='red', edgecolor = 'k', marker = 'o', s=50+(no_points[i]*5))
        ax[2].scatter(0, 0, color = 'r', edgecolor = 'k', s=75, label = 'Peak Cluster Centres', marker='o')
        ax[2].axhline(slow_index, color = 'k', linestyle = '--', label = 'Optimum Beam')
        ax[2].axhline(slow_index+6, color = 'grey', linestyle = '--', label = 'Mean Condition Window')
        ax[2].axhline(slow_index-6, color = 'grey', linestyle = '--') # label = slow_index-6)
        ax[2].axhline(core_centre_mean, color = 'g', linestyle = ':', linewidth = 2, label = 'Weighted Mean')
        ax[2].set_xticks(x_ticks)
        ax[2].set_xticklabels(relative_time, rotation=-45, ha='left', fontsize=16)  
        #ax[2].set_xlim(xlim, xlim_interval) # turn off for full trace vespagram
        ax[2].set_yticks(yaxis_p)
        ax[2].set_yticklabels(yaxis_l, fontsize=16)
        #ax[2].set_xlabel('Sample No./50', fontsize = 20)
        ax[2].set_ylabel('Slowness (s/km)', fontsize = 20)
        ax[2].set_xlabel('Time (s)', fontsize=20)
        ax[2].set_xlim(0, len(abs_vespa_grd[0]))
        ax[2].set_ylim(0, len(slow))
        ax[2].legend(fontsize=15, bbox_to_anchor=(1.9, 2.23))
        ax[2].set_xlim(xlim, xlim_interval) # turn off for full trace vespagram 
        ax[2].text(0.01, 0.92, 'Weighted Mean: %s s/km' %np.round(slow[int(np.round(core_centre_mean,0))],2), fontsize = 16, transform=ax[2].transAxes)
        ax[2].text(0.01, 0.82, 'Standard Deviation: %s' %(np.round(core_centre_std/1000, 4)), fontsize = 16, transform=ax[2].transAxes)
        ax[2].scatter(((bin_P_time-trim_start)*rsample), 0.02 , marker = '^', s = 100, color = 'k', label='Modelled Arrivals', transform=ax[2].get_xaxis_transform())
        ax[2].scatter(((bin_pP_time-trim_start)*rsample), 0.02, marker = '^', s = 100, color = 'k', transform=ax[2].get_xaxis_transform())
        ax[2].scatter(((bin_sP_time-trim_start)*rsample), 0.02, marker = '^', s = 100, color = 'k', transform=ax[2].get_xaxis_transform())
        ax[2].scatter(((bin_S_time-trim_start)*rsample), 0.02, marker = '^', s = 100, color = 'k', transform=ax[2].get_xaxis_transform())
        ax[2].scatter(((bin_sS_time-trim_start)*rsample), 0.02, marker = '^', s = 100, color = 'k', transform=ax[2].get_xaxis_transform())
        ax[2].axvline(x_axis_time_addition)
        ax[2].axvline(x_axis_end)
        #ax[2].set_ylim(0, len(vespa_grd))
        #ax[0].show()
        #ax[0].close()
        
        '''
        ax[3].axhline(slow_index, color = 'k', linestyle = ':')
        mesh = ax[3].pcolormesh(vespa_grd, cmap = 'seismic', vmin = (-vmax*1.05), vmax= (vmax*1.05))
        ax[3].scatter(final_picks, (np.ones(len(final_picks))*slow_index), facecolor = 'gold', edgecolor = 'k', linewidths = 1, s=50, label='Final Picks', vmin=0, vmax=1)
        ax[3].set_ylabel('Slowness (s/km)', fontsize = 20)
        ax[3].set_xlabel('Time (s)', fontsize = 20)
        ax[3].set_xticks(xaxis_p)
        ax[3].set_xticklabels(relative_time, rotation=-45, ha='left', fontsize=16)   
        ax[3].set_yticks(yaxis_p)
        ax[3].set_yticklabels(yaxis_l, fontsize=16)
        ax[3].set_xlim(((bin_P_time-trim_start)*rsample)*0.8, ((bin_sP_time-trim_start)*rsample)*1.2) # turn off for full trace vespagram 
        ax[3].set_ylim(0, len(vespa_grd))
        ax[3].scatter(((bin_P_time-trim_start)*rsample), 0.02 , marker = '^', s = 100, color = 'k', label='Modelled Arrivals', transform=ax[3].get_xaxis_transform())
        ax[3].scatter(((bin_pP_time-trim_start)*rsample), 0.02, marker = '^', s = 100, color = 'k', transform=ax[3].get_xaxis_transform())
        ax[3].scatter(((bin_sP_time-trim_start)*rsample), 0.02, marker = '^', s = 100, color = 'k', transform=ax[3].get_xaxis_transform())
        ax[3].tick_params(width=1, length = 8)
        ax[3].legend(loc='upper right', fontsize=15)
        ax[3].text(0.01, 0.92, 'pP-P: %s seconds' %(dt_pP_TE), fontsize = 16, transform=ax[3].transAxes)
        ax[3].text(0.01, 0.82, 'sP-P: %s seconds' %(dt_sP_TE), fontsize = 16, transform=ax[3].transAxes)
        #ax[2].text(0.01, 0.81, 'Pick Weights: %s' %(subarray.pick_weights), fontsize = 16, transform=ax[2].transAxes)
        #ax[2].set_title('Final_Picks_%s' %subarray.ev_subarray_gcarc)'''
        plt.close()
        
        return picking_figure
        
    def Plot_Arrays(self, event_class, binned_stations_lon, binned_stations_lat, centroids, stlo, stla):
        """Return sub-arrays created from a single obspy stream, with a defined diameter and min. no. of stations.
        
        Parameters:
        stream: any obspy stream containing at least one trace (ideally a national or global distribution of stations)
        min_subarray_diameter: diameter for the sub-array
        min_stations: minimum number of stations needed in the sub-array
        stla: array of station latitudes
        stlo: array of station longitudes
        print_sub_arrays: option to print sub-array stations/data out
        
        Returns: 
        sub_array_stream: array of arrays [[],[],[]], each array contains a stream associated with a single sub-array
        sub_array_fig: figure illustrating the sub-arrays created from a global perspective, centred upon the event coordinates
        dbscan_fig: figure illustrating stations clustered by the DBSCAN machine learning algorithm  """ 
        
        evla = event_class.evla
        evlo = event_class.evlo
       
        #Global Plot
        # example: draw circle with 30 degree radius around the North pole
        lat = evla
        lon = evlo
        r = 30
        proj = ccrs.Orthographic(central_longitude=evlo, central_latitude=evla)
        #proj = ccrs.Mercator(central_longitude=-100)         
        
        def compute_radius(ortho, radius_degrees):
            phi1 = lat + radius_degrees if lat <= 0 else lat - radius_degrees
            _, y1 = ortho.transform_point(lon, phi1, ccrs.PlateCarree())
            return abs(y1)
         
        # Compute the required radius in projection native coordinates:
        r_ortho = compute_radius(proj, r)
         
        sub_array_fig = plt.figure(figsize=(6,6))
        ax = sub_array_fig.add_subplot(1, 1, 1, projection=proj)
        ax = plt.axes(projection=proj)
        ax.set_global()
        ax.add_feature(cfeature.NaturalEarthFeature('cultural', 'admin_0_countries', '50m', edgecolor='darkgrey', facecolor='none', linewidth = 1))
        ax.add_patch(mpatches.Circle(xy=[lon, lat], radius=r_ortho, facecolor = 'none', edgecolor='k', linewidth = 0.5, transform=proj, zorder=30))
        ax.scatter(evlo, evla, marker=(5, 1), s=150, edgecolor='k',  facecolor='red', transform=ccrs.PlateCarree(), zorder = 2) #plot EQ event location
        ax.scatter(stlo, stla, marker='1', clip_on=True, color='darkslategrey', s=15, linewidth=1, label=str(len(stlo))+' Stations', transform=ccrs.PlateCarree())
        
        # Add rectangle for USA zoom in
        proj._threshold /= 100.  # the default values are bad, users need to set them manually

        lat_corners = np.array([24, 24, 50., 50.])
        lon_corners = np.array([-130, -65, -65., -130.])
        
        poly_corners = np.zeros((len(lat_corners), 2), np.float64)
        poly_corners[:,0] = lon_corners
        poly_corners[:,1] = lat_corners
        
        poly = mpatches.Polygon(poly_corners, closed=True, ec='k', linestyle='--', linewidth = 1.5, fill=False, transform=ccrs.PlateCarree(), zorder=5)
        ax.add_patch(poly)
        
        # colours
        RR = [0, 0, 70, 44, 255, 255, 255, 255, 128, 255, 0]
        GG = [255, 0, 220, 141, 255, 200, 142, 0, 0, 153, 180]
        BB = [255, 255, 45, 29, 75, 50, 0, 0, 128, 255, 255]
        
        colors = np.c_[RR, GG, BB] / 255
        colors = [colors[0], colors[4], colors[6], colors[9], colors[4], colors[1], colors[2], colors[5], colors[8], colors[7], colors[10]]

        from cycler import cycler
        color_cyc = cycler('color', colors)
        plt.gca().set_prop_cycle(color_cyc)
        
        lats_lons_use = []
        for b in binned_stations_lon:
            lats_lons_use.extend(b)
        lats_lons_use = len(np.unique(lats_lons_use))
            
    
        # Plot twice to create legend with only one sub-array
        for i in range (1):
            #ax = plt.scatter(np.degrees(lon_centre), np.degrees(lat_centre), zorder=2, c='black', transform=ccrs.PlateCarree())    
            ax.scatter(binned_stations_lon[i], binned_stations_lat[i], zorder=2, s=8, marker='v', label='%s Stations in Arrays' %str(lats_lons_use), transform=ccrs.PlateCarree(), clip_on=True)
            ax.scatter(centroids[i][0], centroids[i][1], zorder=4,  edgecolor='k', facecolor='white', linewidth = 1, s=8, marker='o', label = str(len(centroids))+' Core Stations/Arrays', transform=ccrs.PlateCarree(), clip_on=True)
    
        for i in range (1,len(binned_stations_lon)):
            #ax = plt.scatter(np.degrees(lon_centre), np.degrees(lat_centre), zorder=2, c='black', transform=ccrs.PlateCarree())   
            ax.scatter(binned_stations_lon[i], binned_stations_lat[i], zorder=2, s=8, marker='v', transform=ccrs.PlateCarree(), clip_on=True)
        
        for i in range (1,len(centroids)):
            ax.scatter(centroids[i][0], centroids[i][1], zorder=4,  edgecolor='k', facecolor='white', linewidth = 1, s=8, marker='o', transform=ccrs.PlateCarree(), clip_on=True)
    
        plt.legend(fontsize=12, loc='lower right', bbox_to_anchor=(1.1,-0.1))
        plt.close()
        
        # SAVE OUT CENTROIDS AND ASSOCIATED STATIONS, & Plot -----------------------------------------
        
        #USA Plot
        # example: draw circle with 30 degree radius around the North pole
        lat = evla
        lon = evlo
        r = 30
        #proj = ccrs.Orthographic(central_longitude=evlo, central_latitude=evla)
        proj = ccrs.Mercator(central_longitude=-100)         
        def compute_radius(ortho, radius_degrees):
            phi1 = lat + radius_degrees if lat <= 0 else lat - radius_degrees
            _, y1 = ortho.transform_point(lon, phi1, ccrs.PlateCarree())
            return abs(y1)
         
        # Compute the required radius in projection native coordinates:
        r_ortho = compute_radius(proj, r)
         
        sub_array_fig_US = plt.figure(figsize=(7,3.5))
        ax = sub_array_fig_US.add_subplot(1, 1, 1, projection=proj)
        ax = plt.axes(projection=proj)
        ax.set_extent([-130,-65,24,50], crs=ccrs.PlateCarree())
        #ax.set_global()
        ax.add_feature(cfeature.NaturalEarthFeature('cultural', 'admin_0_countries', '50m', edgecolor='darkgrey', facecolor='none', linewidth=1, zorder=0))
        #ax.add_patch(mpatches.Circle(xy=[lon, lat], radius=r_ortho, facecolor = 'none', edgecolor='k', transform=proj, zorder=30))
        #ax = plt.plot(evlo, evla, marker=(5, 1), markersize='10', color='red', transform=ccrs.PlateCarree()) #plot EQ event location
        ax.scatter(stlo, stla, marker='1', color='darkslategrey', s=15, linewidth=1, label=str(len(stlo))+' Stations', transform=ccrs.PlateCarree())
        
        
        lat_corners = np.array([24, 50, 24, 50])
        lon_corners = np.array([-130, -130, -65, -65])
        
        poly_corners = np.zeros((len(lat_corners), 2), np.float64)
        poly_corners[:,0] = lon_corners
        poly_corners[:,1] = lat_corners
        
        poly = mpatches.Polygon(poly_corners, zorder = 8, closed=True, ec='k', linestyle='--', linewidth = 2, fill=False, transform=ccrs.PlateCarree())
        ax.add_patch(poly)
        
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels = True)
        cartopy.mpl.geoaxes.GeoSpine(ax, linestyle='--', linewidth=2) 
        
        plt.gca().set_prop_cycle(color_cyc)
        for i in range (0,len(binned_stations_lon)):
            ax.scatter(binned_stations_lon[i], binned_stations_lat[i], zorder=2, s=20, ec='dimgrey', linewidth=0.5, marker='v', transform=ccrs.PlateCarree(), clip_on=True)     
        for i in range (0,len(centroids)):
            ax.scatter(centroids[i][0], centroids[i][1], zorder=4, edgecolor='k', facecolor='white', linewidth = 2, s=20, marker='o', transform=ccrs.PlateCarree())
    
       	plt.close()
        
        '''# plot one subarray
        #outputs = self.output
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(1, 1, 1)  
        ax.scatter(binned_stations_lon[0], binned_stations_lat[0], edgecolor = 'dimgrey', facecolor=colors[0], linewidth = 0.5, zorder=2, s=100, marker='v')
        ax.scatter(np.mean(binned_stations_lon[0]), np.mean(binned_stations_lat[0]), marker=(4,1), color='red', s=200, label=str(np.mean(binned_stations_lon[0]))+ ', ' + str(np.mean(binned_stations_lat[0])), zorder=1)
        ax.set_xlim(-120,-117.5)
        ax.set_ylim(32.6, 34.2)

        #for i in range(len(binned_stations_lon[0])):
        #    plt.annotate(outputs.stname[i], (outputs.stlo[i]+0.03, outputs.stla[i]))

        plt.xlabel('Longitude (degrees)')
        plt.ylabel('Latitude (degrees)')
        #plt.title('Station Locations')
        #plt.legend(fontsize=12)
        plt.grid()
        
        plt.show()'''
        
        # Find number of stations not used:
        used_lo = []
        used_la = []
            
        for k in range (len(stlo)):
            for i in range (len(binned_stations_lon)):
                if stlo[k] in binned_stations_lon[i] and stla[k] in binned_stations_lat[i]:
                    #print(stlo[k], binned_stations_lon[i])
                    used_lo.append(stlo[k])
                    used_la.append(stla[k])
                    continue

        no_used = len(used_lo)
        no_not_used = len(stlo) - no_used
        print('UNUSED STATIONS', no_not_used)       
        return  sub_array_fig, sub_array_fig_US







        
