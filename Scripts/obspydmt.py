#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 11:34:11 2022

@author: ee18ab
"""

from obspy import read_events
from obspy.clients.fdsn.client import Client
from obspy.core.utcdatetime import UTCDateTime
import re
import os
import sys
import pickle
import numpy as np

# OUTPUT FILE NAME
#name = 'ObspyDMT_Events_1995'

# Flags
#make_catalogue = True
#split_catalogue = True
#download_data_Z = False
#download_data_NEZ = True
#single_event_download = False # requires event number (row) in catalogue as an input in the command line 'python 0_ObspyDMT.py n', n starts from 1. Unless you have 1 event in the catalogue.

def run_obspyDMT(name, make_catalogue, split_catalogue, download_data_Z, download_data_NEZ, single_event_download, event=False):
    if download_data_Z==True and download_data_NEZ==True:
        print()
        print('If you would like to donwload data, please choose either Z or NEZ (1 or 3) component data with flags at the top of the script.')
        sys.exit()

    if single_event_download==True and split_catalogue==False:
        print()
        print('WARNING: If you want to download multiple events in parallel using the single_event_download method, you need to split the catalogue. Ignore this if you have already split the catalogue.')
        print('If you are working with a single event catalogue, use single_event_download = False. This will download the entire catalogue in sequence, which is fine because there is only one event in there.')
        print()

    '''# TRY IMPORTING EVENTS DIRECTLY
    t1 = str(UTCDateTime("2010-05-23T22:00:00"))
    t2 = str(UTCDateTime("2010-05-23T22:59:59"))
    mlat = '-58.0'
    maxlat = '14.0'
    mlon = '-83.0'
    maxlon = '-60.0'
    mdepth = '40.0'
    maxdepth = '350.0'
    mmag = '4.7'
    maxmag = '6.5'
    magtype = 'mb'
    cat = 'ISC'
    sort = 'magnitude'
    #limit = '3'
    file = '%s.xml' %name
    client = Client('ISC')
    #station_separation = '0.01' #degrees'''

    t1 = str(UTCDateTime("1995-08-19T21:40:00"))
    t2 = str(UTCDateTime("1995-08-19T21:45:59"))
    mlat = '-58.0'
    maxlat = '14.0'
    mlon = '-83.0'
    maxlon = '-60.0'
    mdepth = '40.0'
    maxdepth = '350.0'
    mmag = '4.7'
    maxmag = '6.5'
    magtype = 'mb'
    cat = 'ISC'
    sort = 'magnitude'
    #limit = '3'
    file = '%s.xml' %name
    client = Client('ISC')
    #station_separation = '0.01' #degrees


    if make_catalogue == True:
        # Load events which fit the search criteria, and save out as an xml file
        command = 'obspyDMT --datapath ' + name
        command += ' --event_rect ' + mlon + '/' + maxlon + '/' + mlat + '/' + maxlat
        command += ' --min_depth ' + mdepth + ' --max_depth ' + maxdepth
        command += ' --mag_type ' + magtype
        command += ' --min_mag ' + mmag +  ' --max_mag ' + maxmag 
        command += ' --min_date ' + t1 + ' --max_date ' + t2 
        command += ' --event_catalog ' + cat 
        command += ' --isc_catalog COMPREHENSIVE'
        
        # Load Catalogue
        command_events = command + ' --event_info'
        print(command_events)
        os.system(command_events)
        
        # Load event xml files to make text file from
        data_file = name + '/EVENTS-INFO/catalog.ml.pkl'
        
        with open(data_file, 'rb') as f:
            catalogue = pickle.load(f)
            
        # Read in new catalogue
        #catalogue = read_events(events)
        print(type(catalogue))
        
        # Plot event locations
        #catalogue.plot(projection = 'local')
        
        # Make txt file of events
        f = open('%s.txt' %name, 'w')
        f.write('EQ Name'.ljust(12) +'\t'+ 'Event_ID'.ljust(9) + '\t' + 'mb'.ljust(4) +'\t'+ 'Lat'.ljust(8)+ '\t'+ 'Lon'.ljust(8) + '\t' + 'Depth'.ljust(8) + '\t' + 'yyyy'.ljust(4) + '\t' + 'mn'.ljust(2) + '\t' + 'dd'.ljust(2) + '\t' + 'hh'.ljust(2) + '\t' + 'mm'.ljust(2) + '\t' + 'ss'.ljust(2) + '\n')
        
        for i in range (len(catalogue)):
            event = catalogue[i]
            event_id = re.sub("[^0-9]", "", str(event.resource_id))
            #print(str(event_id))
            lat = event.origins[0].latitude
            lon = event.origins[0].longitude
            dp = event.origins[0].depth/1000
            mag = event.magnitudes[0].mag
            time = event.origins[0].time
            
            yyyy = time.year
            mn = time.month
            dd = time.day
            hh = time.hour
            mm = time.minute
            ss = time.second
            evname=str(yyyy)+str(mn)+str(dd)+str(hh)+str(mm)+str(ss)
            
            f.write(str(evname).ljust(12) +'\t'+ str(event_id).ljust(9) + '\t' + str(mag).ljust(4) +'\t'+ str(lat).ljust(8) + '\t' + str(lon).ljust(8) + '\t' + str(np.round(dp,4)).ljust(8) + '\t' + str(yyyy).ljust(4) + '\t' + str(mn).ljust(2) + '\t' + str(dd).ljust(2) + '\t' + str(hh).ljust(2) + '\t' + str(mm).ljust(2) + '\t' + str(ss).ljust(2) + '\n')
            
        f.close()
        
        with open(name + '/EVENTS-INFO/catalog.txt') as f:
            lines = f.readlines()
            headers, lines = [lines[4]], lines[5:]
            with open(name + '/EVENTS-INFO/working_catalogue.txt', "w") as f:
                f.writelines(headers + lines)

    if split_catalogue == True:
        
        # Make new directory for split catalogues
        path = os.path.join(name, 'EVENTS-INFO/individual_catalogues')
        if os.path.isdir(path):
        	pass
        else:
            os.mkdir(path)
        
        #Split the catalogue into single lines for task arrays/parallel data downloading
        LINES_PER_FILE = 1

        def write_to_file(name, headers, lines):
            with open(name, "w") as f:
                print(headers + lines)
                f.writelines(headers + lines)
        
        with open(name + '/EVENTS-INFO/catalog.txt') as f:
            lines = f.readlines()
            headers, lines = [lines[4]], lines[5:]
            [write_to_file(name + '/EVENTS-INFO/individual_catalogues/' + f'{i+1}.txt', headers, lines[i: i+LINES_PER_FILE]) for i in range(0, len(lines)-1, LINES_PER_FILE)]


    if download_data_Z == True:
        
        if single_event_download == True:
            #input_no = sys.argv[1:]
            input_no = event
            
            # If a specific event in the catalogue is not specified
            if input_no == False:
                print('Event number not provided -- event arguement')
                sys.exit()
                
            cat_file = name + '/EVENTS-INFO/individual_catalogues/' + str(int(input_no)) + '.txt'    
        
        else:
            cat_file = name + '/EVENTS-INFO/working_catalogue.txt'
        
        command = 'obspyDMT --datapath ' + name
        command += ' --data_source AusPass,BGR,EMSC,ETH,GEOFON,GEONET,GFZ,ICGC,IESDMC,IGN,INGV,IPGP,ISC,KNMI,KOERI,LMU,NCEDC,NIEP,NOA,NRCAN,ODC,ORFEUS,RASPISHAKE,RESIF,SCEDC,TEXNET,UIB-NORSAR,USGS,USP,IRIS'
        command += ' --min_epi 30 --max_epi 90'
        #command += ' --net TA'
        command += ' --cha BHZ,HHZ'
        command += ' --preset 200 --offset 1500'
        #command += ' --req_parallel --req_np 4 --parallel_process --process_np 4'
        command_load = command
        command_load += ' --read_catalog ' + cat_file
        command_load += ' --corr_unit=VEL'
        command_load += ' --instrument_correction'
        command_load += ' --sampling_rate=10'
        
        # Load data
        print(command_load)
        os.system(command_load)


    if download_data_NEZ == True:
        
        if single_event_download == True:
            #input_no = sys.argv[1:]
            input_no = event
            
            # If a specific event in the catalogue is not specified
            if input_no == False:
                print('Event number not provided -- event arguement')
                sys.exit()
                
            cat_file = name + '/EVENTS-INFO/individual_catalogues/' + str(int(input_no)) + '.txt' 

        else:
            cat_file = name + '/EVENTS-INFO/working_catalogue.txt'
                

        # Load data for events in P wave code
        command = 'obspyDMT --datapath ' + name
        #command += ' --local'  # turn on to process without retreiving data
        command += ' --data_source BGR,EMSC,ETH,GEONET,GFZ,ICGC,INGV,IPGP,ISC,KNMI,KOERI,LMU,NCEDC,NIEP,NOA,ODC,ORFEUS,RASPISHAKE,RESIF,SCEDC,TEXNET,UIB-NORSAR,USGS,USP,IRIS'
        command += ' --min_epi 30 --max_epi 90'
        #command += ' --cha BHE,HHE,BHN,HHN'
        command += ' --cha BHE,HHE,BHN,HHN,BH1,HH1,BH2,HH2,BHZ,HHZ'
        command += ' --preset 200 --offset 1500'
        #command += ' --req_parallel --req_np 4 --parallel_process --process_np 4'
        command_load = command
        command_load += ' --read_catalog ' + cat_file
        command_load += ' --corr_unit=VEL'
        command_load += ' --instrument_correction'
        command_load += ' --sampling_rate=10'
        
        # Load data
        print(command_load)
        os.system(command_load)

    if make_catalogue!=True and split_catalogue!=True and download_data_Z!=True and download_data_NEZ!=True:
        print('You need to select a mode by changing the input flags.')
    else:
        print('Script complete. Catalogue/Data located in %s.' %name)
    return

