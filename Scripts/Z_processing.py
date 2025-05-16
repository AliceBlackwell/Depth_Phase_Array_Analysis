#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 10:39:53 2022

Author: Alice Blackwell

Script to load in an xml file of earthquake events (generated from script 0), 
download teleseismic data available using Obspy's massdownloader, and process the data.

Raw data (MSEED and xml) are saved into a file called ISC_Unpro_DATA and processed data (MSEED and xml) are saved into ISC_Pro_DATA.

Opportunity to update:
+ Parent Directory
+ Data type ('DISP', 'VEL', 'ACC')
+ Components
+ Processing variables (frequency bands, sample rate, tapering)  

"""

# ## IMPORT MODULES
import warnings
warnings.filterwarnings("ignore")

import obspy
import os
import numpy as np
import sys
import shutil
import pickle
import glob

from copy import deepcopy

import pandas as pd

from obspy.signal.filter import bandpass
from obspy.clients.fdsn.mass_downloader import CircularDomain, Restrictions, MassDownloader
from obspy import read_events
from obspy.core.stream import Stream


# INPUT EVENT ==================================================================================================

def QC_Stream(stream, evname, origin_time, datatype, data_file, stats_output_file):
    """Returns stream (and metadata) with traces that only contain all associated metadata (instrument response optional),
    and without any NaN or Infinity traces.
    
    Parameters:
    stream: any obspy stream containing at least one trace
    
    Returns: 
    stream: cleaned stream with traces that have all associated metadata and no Nan or Infinity values
    stname: array of station names
    ststring: array of station seed IDs 
    stla: array of station latitudes
    stlo: array of station longitudes
    stel: array of station elevations (m)
    stnet: array of station networks
    nsamples: array of sample points per traces
    inv: stream inventory 
    ntraces: number of tractes/stations in the stream """ 
   
    f = stats_output_file
    
    ntraces = len(stream)
    print('Initial no. of traces =', ntraces)
    
    # Check for NaNs or Infs, delete trace
    for i in range(0,ntraces):
        try:
            if (np.isfinite(stream[i]).any() == False):
                print ('Traces contain non-numeric values!')
                print ('Failure at trace ', i,';','stname[i]')
                del stream[i]
            else:
                pass
        except IndexError as id_err:
                print(id_err)
                pass 
    
    ntraces = len(stream)
    print('Number of traces remaining after NaNs/Infs check =', ntraces)
    f.write('Number of traces remaining after NaNs/Infs check = ' + str(ntraces) + '\n')
    
    # Check for zero traces, delete trace
    for i in range(0,ntraces):
        try:
            if (not np.any(stream[i]) == True):
                print ('Traces contain only zero values!')
                print ('Failure at trace ', i,';','stname[i]')
                del stream[i]
            else:
                pass
        except IndexError as id_err:
                print(id_err)
                pass 
    
    ntraces = len(stream)
    print('Number of traces remaining after zeros check =', ntraces)
    f.write('Number of traces remaining after zeros check = ' + str(ntraces) + '\n')
    
    #**************************************************************************
    
    # Extract metadata from the stream

    stname_tmp = [0] * ntraces # Station name
    ststring_tmp = [0] * ntraces # Station SEED ID
    stla_tmp = [0] * ntraces # Station latitude
    stlo_tmp = [0] * ntraces # Station longitude
    stel_tmp = [0] * ntraces # Station elevation
    stnet_tmp = [0] * ntraces # Station network 
    nsamples_tmp = [0] * ntraces #no. sample points per trace
    stream_tmp = Stream()
    
    print('data_file', data_file)
    pathname_mseed = data_file + '/processed'
    pathname_xml = data_file + '/resp'
    print(pathname_mseed, pathname_xml)

    for i in range(0,ntraces):   
        if glob.glob(pathname_xml+ '/STXML.' + stream[i].stats.network + '.' + stream[i].stats.station+'*.*HZ'):
            print(pathname_xml+ '/STXML.' + stream[i].stats.network + '.' + stream[i].stats.station+'*.*HZ')
            try:
                stname_tmp[i] = stream[i].stats.station
                ststring_tmp[i] = stream[i].get_id()
                rname = pathname_xml+ '/STXML.' + stream[i].stats.network + '.' + stream[i].stats.station+'*.*HZ'
                inv_tmp=obspy.read_inventory(rname) #creates a one trace inventory 
                stnet_tmp[i]=stream[i].stats.network
                nsamples_tmp[i] = stream[i].stats.npts
                statcoords = inv_tmp.get_coordinates(ststring_tmp[i],origin_time)
                stla_tmp[i] = (statcoords[u'latitude'])
                stlo_tmp[i] = (statcoords[u'longitude'])
                stel_tmp[i] = (statcoords[u'elevation'])
                
                stream_tmp.append(stream[i])
            except:
                rname = pathname_xml+ '/STXML.' + stream[i].stats.network + '.' + stream[i].stats.station+'*.*HZ'
                print('No metadata for', rname)
                pass

        else:
            rname = pathname_xml+ '/STXML.' + stream[i].stats.network + '.' + stream[i].stats.station+'*.*HZ'
            print('Missing xml file for', rname)
            pass

    stream = deepcopy(stream_tmp)
    ntraces = len(stream)
    print('Number of traces remaining after metadata check =', ntraces)
    f.write('Number of traces remaining after metadata check = ' + str(ntraces) + '\n')
    
    #*************************************************************************************************
    
    # Check for Instrument Response, Create Inventory
    '''
    instr_resp = [0]*ntraces
    f.write(str(ntraces) + '\n')
    
    for i in range(0,ntraces):
        rname = pathname_xml + '/' + stream[i].stats.network + '.' + stream[i].stats.station+'.xml'
        print(i+1, 'out of', ntraces,':', rname, end='\r')
        if i==0:
            inv = obspy.read_inventory(rname)
        else:
            inv=inv + obspy.read_inventory(rname) #creates the inventory for the stream
    
    f.write(str(ntraces) + '\n')
    for i in range(0,ntraces):
        try:
            stream[i].remove_response(inventory=inv, output=datatype, pre_filt=None, zero_mean=True, taper=True)
            instr_resp[i]=1
        except:
            print('Instrument response not removed', i, stream[i].stats.station)
            f.write('Instrument response not removed: '+ str(stream[i].stats.station) +'\n')
            pass
      
    stream_tmp = Stream()
    for i in range (ntraces):
        if instr_resp[i]==1:
             stream_tmp.append(stream[i])
    
    stream = stream_tmp

    ntraces = len(stream)
    print('Number of traces remaining after instrument response check =', ntraces)
    print('Number of traces in inventory =', len(inv))
    f.write('Number of traces remaining after instrument response check = ' + str(ntraces) + '\n')
    f.write('Number of traces in inventory = '+ str(len(inv)) + '\n')'''
    
    #**************************************************************************************************
    
    # Extract metadata from the stream

    stname = [0] * ntraces # Station name
    ststring = [0] * ntraces # Station SEED ID
    stla = [0] * ntraces # Station latitude
    stlo = [0] * ntraces # Station longitude
    stel = [0] * ntraces # Station elevation
    stnet = [0] * ntraces # Station network 
    nsamples = [0] * ntraces #no. sample points per trace

    for i in range(0,ntraces):
        stname[i] = stream[i].stats.station
        ststring[i] = stream[i].get_id()   #array containing SEED identifiers for each trace with network, station, location & channel code

    for i in range(0,ntraces):
        rname = pathname_xml+ '/STXML.' + stream[i].stats.network + '.' + stream[i].stats.station+'*.*HZ'
        print(i, 'out of', ntraces-1,':', rname, end='\r')
        inv=obspy.read_inventory(rname) #creates a one trace inventory 
        stnet[i]=stream[i].stats.network
        nsamples[i] = (stream[i].stats.npts)
        statcoords = inv.get_coordinates(ststring[i],origin_time)
        stla[i] = (statcoords[u'latitude'])
        stlo[i] = (statcoords[u'longitude'])
        stel[i] = (statcoords[u'elevation'])

    ntraces = len(stream)
    print('Number of traces metadata is extracted for =', ntraces)
    f.write('Number of traces metadata is extracted for = ' + str(ntraces) + '\n')
    
    inv = 0
    for i in range(0,ntraces):
        rname = pathname_xml+ '/STXML.' + stream[i].stats.network + '.' + stream[i].stats.station+'*.*HZ'
        print(i+1, 'out of', ntraces,':', rname, end='\r')
        if i==0:
            inv = obspy.read_inventory(rname)
        else:
            inv=inv + obspy.read_inventory(rname) #creates the inventory for the stream
    
    f.write('Number of traces in final inventory = '+ str(len(inv)) + '\n')
            
    stream.normalize()
        
    return stream, stname, ststring, stla, stlo, stel, stnet, nsamples, inv, ntraces

def import_process_data(catalogue, re_processing, data_dir, parent_dir):
    
    try:
        event = catalogue
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

        # SET VARIABLES ===================================================================================================
        
        parent_dir = parent_dir
        
        # Data Type ('DISP', 'VEL', 'ACC')
        datatype = 'VEL'
        
        # Processing Variable
        rsample = 10
        tpratio = 0.05
        frqmin = 1/10
        frqmax = 1/1        
        
        # Set distance limits for the data download (degrees from event)
        distlims = [30, 90]
        
        # Set the array to download data for:
        # Requires the user to check the the array is within the distance range specified, otherwise no data will be downloaded
        arr_name = str('*') 
        arr_net = str('*')
        # Download data and save
        # Set array components to request
        arr_comp = ['BH[Z]', 'HH[Z]']
        
        # Create File Structure ============================================
        '''try:
            # Create folder
            directory = 'ISC_Unpro_DATA'
            unpro_data = os.path.join(parent_dir, directory)
            os.mkdir(unpro_data)
            #unpro_data = '/nobackup/ee18ab/10_Peruvian_data'
            print('Directory %s created' %directory )
            
        except FileExistsError:
            directory = 'ISC_Unpro_DATA'
            unpro_data = os.path.join(parent_dir, directory)
            #unpro_data = '/nobackup/ee18ab/10_Peruvian_data'
            pass'''   
            
        try:
            directory = 'Processed_DATA'
            pro_data = os.path.join(parent_dir, directory)
            os.mkdir(pro_data)
            print('Directory %s created' %directory )
        
        except FileExistsError:
            directory = 'Processed_DATA'
            pro_data = os.path.join(parent_dir, directory)
            pass
            
        un_pro_ev_dir = data_dir + '/' + evname_obspyDMT + '.a'
        if os.path.isdir(un_pro_ev_dir) == True:
            pass
        else:
            txt_file_name = 'Events_with_no_data.txt'
            outputs_txt = os.path.join(pro_data, txt_file_name)
            f = open(outputs_txt, 'a+')
            f.write(str(evname_obspyDMT) + '\n')
            f.close()
            print('NO PRE-LOADED DATA')
            return
        
        '''try:    
            directory = '%s' %evname
        
            un_pro_ev_dir = os.path.join(unpro_data, directory)
            os.mkdir(un_pro_ev_dir)
            print('Directory %s created' %directory )
        
        except FileExistsError:
            directory = '%s' %evname
            un_pro_ev_dir = os.path.join(unpro_data, directory)
            pass'''
        
        try:
            directory = '%s' %evname
        
            pro_ev_dir = os.path.join(pro_data, directory)
            os.mkdir(pro_ev_dir)
            print('Directory %s created' %directory )
            
            directory = 'Data'
            path = os.path.join(pro_ev_dir, directory)
            os.mkdir(path)
            print('Directory %s created' %directory )
            
            directory = 'Stations'
            path = os.path.join(pro_ev_dir, directory)
            os.mkdir(path)
            print('Directory %s created' %directory )
            
            directory = 'Arrays'
            path = os.path.join(pro_ev_dir, directory)
            os.mkdir(path)
            print('Directory %s created' %directory )
        
            directory = 'Initially_Processed'
            path = os.path.join(pro_ev_dir, directory)
            os.mkdir(path)
            print('Directory %s created' %directory )
        
        except FileExistsError:
            directory = '%s' %evname
            pro_ev_dir = os.path.join(pro_data, directory)
            pass
        
        # Set up summary txt file
        txt_file_name = 'Script_1_Summary.txt'
        outputs_txt = os.path.join(pro_ev_dir, txt_file_name)
        f = open(outputs_txt, 'w')
        
        # Load Data ==================================================================== 
                
        # Read in the downloaded waveforms
        arr_net='*' # Reset Array network to a wildcard
        #stream = obspy.read(un_pro_ev_dir+"/processed/" + '*Z')
        #ntraces = len(stream)
        #stream.resample(rsample)
        
        #print ("Number of traces read in = ",ntraces)
        #f.write("Number of traces initially read in = " + str(ntraces) + '\n')
    
        #QC Data and Map Stations ===========================================================================================
        '''new_MSEED = True
        if new_MSEED == False:
            print('Data already processed')
            f.write('Data already processed' + '\n')
            f.write('Final no. of traces post processing = ' + str(len(stream_z)) + '\n')
            f.close()
            pass'''
    
        if re_processing == True:
        
            f.write('Data going through QC checks' + '\n')
            
            '''try:
                os.remove(pro_ev_dir+'/*')  # Remove processed data file to prevent data issues later
            except OSError:
                pass'''
            
            try:               
                directory = 'Data'
                path = os.path.join(pro_ev_dir, directory)
                os.mkdir(path)
                print('Directory %s created' %directory )
                                
                directory = 'Stations'
                path = os.path.join(pro_ev_dir, directory)
                os.mkdir(path)
                print('Directory %s created' %directory )
                
                directory = 'Arrays'
                path = os.path.join(pro_ev_dir, directory)
                os.mkdir(path)
                print('Directory %s created' %directory )
                
                directory = 'Initially_Processed'
                path = os.path.join(pro_ev_dir, directory)
                os.mkdir(path)
                print('Directory %s created' %directory )
                
            except FileExistsError:
                pass
            
            f.write('Data going through QC checks' + '\n')       
            stream_z = obspy.read(un_pro_ev_dir + "/processed/" + '*Z')  
            print(un_pro_ev_dir + "/processed/" + '*Z')

            # Remove extra traces =================================================================================       
            '''ststring = [0] * len(stream_z)
            ststring_og = [0] * len(stream_z)
            for i in range(0, len(stream_z)):
                ID = stream_z[i].get_id()[:-3]
                ststring_og[i] = stream_z[i].get_id()
                if '..' in ID:
                    ststring[i] = ID
                else:
                    ststring[i] = ID[:-3]
                
            stations = np.unique(ststring)'''
            
            ststring = [0] * len(stream_z)
            ststring_og = [0] * len(stream_z)
            for i in range(0, len(stream_z)):
                print(stream_z[i])
                ststring[i] = stream_z[i].stats.network + '.' + stream_z[i].stats.station
                ststring_og[i] = stream_z[i].get_id()
                print('ID', ststring[i], ststring_og[i])
        
            stations = np.unique(ststring)
            final_traces = [0]*len(stations)
            
            for j in range (len(stations)):
                Z = 0
                Z_pop = 0
                failure = 0
                
                print()
                sta = stations[j][3:]
                st = stream_z.select(network = stations[j][:2], station = stations[j][3:])
                print(st)
                
                try:
                    if len(st.select(component='Z')) == 1:
                        Z = st.select(component='Z')
                    elif len(st.select(component='Z')) > 1:
                        Z = st.select(channel='BHZ')
                        if len(Z) > 1:
                            channels = []
                            for i in range (len(Z)):
                                channels.append(Z[i].get_id()[-6:-4])
                                if '..' in Z[i].get_id(): 
                                    Z = Z[i]
                                    Z = Stream(traces=[Z])
                                    Z_pop = 1
                                    break
                            if Z_pop == 0:
                                index = np.argmin(channels)
                                Z = Z[index]
                                Z = Stream(traces=[Z])
                    final_traces[j] = Z[0].get_id()
                except:
                    print('No Z component')
            
            stream_z = Stream()
            for i in range (len(final_traces)):
                if final_traces[i] != 0:
                    print(un_pro_ev_dir + "/processed/" + str(final_traces[i]))
                    stream_z.extend(obspy.read(un_pro_ev_dir + "/processed/" + str(final_traces[i])))  
            print('Length of Z stream, pre-QC:', len(stream_z))
            
            stream_Z, stname_Z, ststring_Z, stla_Z, stlo_Z, stel_Z, stnet_Z, nsamples_Z, inv_Z, ntraces_Z = QC_Stream(stream_z, evname_obspyDMT, origin_time, datatype, un_pro_ev_dir, f)

            print('Length of Z stream, post-QC:', len(stream_z))            

            for i in range(0, len(stream_Z)):
                tr = stream_Z[i] 
                tr.write(pro_ev_dir+ "/Initially_Processed/"+tr.stats.network+"."+tr.stats.station + "." + tr.stats.channel + '.MSEED', format = 'MSEED')
            
            # Create table of station network, name, longitude and latitude           
            data= {
                'Station_Network': stnet_Z,
                'Station_Name': stname_Z,
                'Station_Longitude': stlo_Z,
                'Station_Latitude': stla_Z
            }
            df=pd.DataFrame(data)
            #print(df)
            
            # Plot loaded stations relative to earthquake ('flat', or 'global')
            #figure = ev_station_plot(df, evla, evlo)
            
            # WRITE OUT PROCESSED DATA ======================================================================================
            name = 'stname.npy'
            path = os.path.join(pro_ev_dir + '/Arrays', name)
            np.save(path, stname_Z, allow_pickle=True)
            
            name = 'ststring.npy'
            path = os.path.join(pro_ev_dir + '/Arrays', name)
            np.save(path, ststring_Z, allow_pickle=True)
                    
            name = 'stla.npy'
            path = os.path.join(pro_ev_dir + '/Arrays', name)
            np.save(path, stla_Z, allow_pickle=True)
            
            name = 'stlo.npy'
            path = os.path.join(pro_ev_dir + '/Arrays', name)
            np.save(path, stlo_Z, allow_pickle=True)
            
            name = 'stnet.npy'
            path = os.path.join(pro_ev_dir+ '/Arrays', name)
            np.save(path, stnet_Z, allow_pickle=True)    
            
            name = 'stel.npy'
            path = os.path.join(pro_ev_dir+ '/Arrays', name)
            np.save(path, stel_Z, allow_pickle=True) 
            
        print('Loading processed data')
        #pro_ev_dir = './ISC_Pro_DATA/200175135350'
        
        # Load saved outputs numpy array
        name = 'stla.npy'
        path = os.path.join(pro_ev_dir+ '/Arrays', name)
        stla = np.load(path, allow_pickle=True)
    
        name = 'stlo.npy'
        path = os.path.join(pro_ev_dir+ '/Arrays', name)
        stlo = np.load(path, allow_pickle=True)
        
        name = 'stnet.npy'
        path = os.path.join(pro_ev_dir+ '/Arrays', name)
        stnet = np.load(path, allow_pickle=True)

        name = 'stname.npy'
        path = os.path.join(pro_ev_dir + '/Arrays', name)
        stname = np.load(path, allow_pickle=True)
        
        name = 'stel.npy'
        path = os.path.join(pro_ev_dir + '/Arrays', name)
        stel = np.load(path, allow_pickle=True)
        
        print(len(stla), len(stlo), len(stnet), len(stname), len(stel))
        
        stream_Z = Stream()
        for i in range (len(stnet)):
            #print(i)
            #print(pro_ev_dir + "/Initially_Processed/" + str(stnet[i]) + '.' + str(stname[i]) + '.*HZ.MSEED')
            stream_Z.extend(obspy.read(pro_ev_dir + "/Initially_Processed/" + str(stnet[i]) + '.' + str(stname[i]) + '.*HZ.MSEED'))
        
        print('Length of Z stream, post load:', len(stream_Z))

        # Save MSEEDS
        for i in range(0, len(stream_Z)):
            try:
                tr = stream_Z[i] 
                tr.detrend(type='simple') # Detrend waveforms
                tr.detrend(type='demean') # Demean waveforms
                tr.taper(tpratio) # Taper waveforms
                tr.data = bandpass(tr.data, frqmin, frqmax,tr.stats.sampling_rate, 3, True) # Filter waveforms
                tr.normalize() # Normalise waveforms
                tr.resample(rsample)
                tr.write(pro_ev_dir+ "/Data/"+tr.stats.network+"."+tr.stats.station+  "." + tr.stats.channel + '.MSEED', format = 'MSEED')
                
                # Save out inventory xml file
                rname =  un_pro_ev_dir + '/resp/STXML.' + tr.stats.network + '.' + tr.stats.station+'*.*HZ'
                inv=obspy.read_inventory(rname) #creates a one trace inventory 
                inv.write(pro_ev_dir+ "/Stations/"+tr.stats.network+"."+tr.stats.station + "." + tr.stats.channel +'.xml', format = 'stationxml')
            except Exception as e:
                print(e, sys.exc_info()[-1].tb_lineno)
                f.write('trace failed: ' + str(tr) + '\n') # NOTE THIS WILL MAKE SAVED .npys INVALID
                continue
        
        f.write('Final no. of traces post processing = ' + str(len(stream_Z)) + '\n')
        f.close()
        print('Final no. of traces post processing = ' + str(len(stream_Z)))

        try:
            shutil.rmtree(pro_ev_dir + "/Initially_Processed/")  # Remove initially processed data - no longer needed
        except OSError:
            print('Initially processed directory not deleted')
            pass
        return
    
    except Exception as e:
        print('Data for event %s failed to download/be processed' %evname_obspyDMT)
        print(e)
        txt_file_name = 'Failed_Events_from_Z_Processing.txt'
        outputs_txt = os.path.join(pro_data, txt_file_name)
        
        f = open(outputs_txt, 'a+')
        f.write(str(evname_obspyDMT) + '\t' + str(e) + '\n')
        f.close()
        pass

# ======= MAIN ==========
def process_Z_components(catalogue, event, re_processing, data_dir, parent_dir):
    # input_no indexes an event from the catalogue
    input_no = event

    # Option to write out processed traces again, without preliminary checks (False)
    re_processing = re_processing

    event = catalogue[input_no-1]
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
    evname=str(yyyy)+str(mn)+str(dd)+str(hh)+str(mm)+str(ss)
    print("Event names is:",evname)

    import_process_data(catalogue[input_no-1], re_processing, data_dir, parent_dir)
    
    print('Z component data for %s processed.' %evname)
    return
