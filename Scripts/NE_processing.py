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

import pandas as pd

from obspy.signal.filter import bandpass
from obspy.clients.fdsn.mass_downloader import CircularDomain, Restrictions, MassDownloader
from obspy import read_events
from obspy.core.stream import Stream

# INPUT EVENT ==================================================================================================

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
        
        # Set Parent Directory
        parent_dir = parent_dir
        
        # Data Type ('DISP', 'VEL', 'ACC')
        datatype = 'VEL'
        
        # Processing Variable
        rsample = 10
        tpratio = 0.05
        frqmin = 0.03 # Molnar et al. (1973), Langston (2014) 
        frqmax = 0.2       
        
        # Set distance limits for the data download (degrees from event)
        distlims = [30, 90]
        
        # Set the array to download data for:
        # Requires the user to check the the array is within the distance range specified, otherwise no data will be downloaded
        arr_name = str('*') 
        arr_net = str('*')
        # Download data and save
        # Set array components to request
        arr_comp = ['BH[NE]', 'HH[NE]']
        
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
            
        un_pro_ev_dir = data_dir + '/' + evname_obspyDMT + '.a'
        if os.path.isdir(un_pro_ev_dir) == True:
            pass
        else:
            txt_file_name = 'Events_with_no_data.txt'
            outputs_txt = os.path.join('.', txt_file_name)
            f = open(outputs_txt, 'a+')
            f.write(str(evname_obspyDMT) + '\n')
            f.close()
            print('NO PRE-LOADED DATA')
            return
            
        try:
            directory = 'Processed_DATA'
            pro_data = os.path.join(parent_dir, directory)
            os.mkdir(pro_data)
            print('Directory %s created' %directory )
        
        except FileExistsError:
            directory = 'Processed_DATA'
            pro_data = os.path.join(parent_dir, directory)
            pass
        
        '''try:    
            directory = '%s' %evname
        
            un_pro_ev_dir = os.path.join(unpro_data, directory)
            os.mkdir(un_pro_ev_dir)
            print('Directory %s created' %directory )
        
        except FileExistsError:
            directory = '%s' %evname
            un_pro_ev_dir = os.path.join(unpro_data, directory)
            pass'''
        '''
        try:
            os.remove(un_pro_ev_dir+'/Data_unmatched')  # Remove processed data file to prevent data issues later
        except OSError:
            pass'''
        
       
        try:    
            directory = '%s' %evname      
            pro_ev_dir = os.path.join(pro_data, directory)
            os.mkdir(pro_ev_dir)
            print('Directory %s created' %directory )
                
        except FileExistsError:
            directory = '%s' %evname
            pro_ev_dir = os.path.join(pro_data, directory)
            pass
        
        # Set up summary txt file
        txt_file_name = 'Script_1S_Summary.txt'
        outputs_txt = os.path.join(pro_ev_dir, txt_file_name)
        f = open(outputs_txt, 'w')
        
        # Load Data ==================================================================== 

        #QC Data and Map Stations ===========================================================================================
        '''new_MSEED = True
        if new_MSEED == False:
            print('Data already processed')
            f.write('Data already processed' + '\n')
            f.write('Final no. of traces post processing = ' + str(len(stream)) + '\n')
            f.close()
            pass'''
        
        '''if os.path.isfile(pro_ev_dir + '/Arrays/stname_S.npy') == True:
            print('Already processed')
            return'''
        
        if re_processing == True:
                    
            f.write('Data going through QC checks' + '\n')
            '''
            try:
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
                
            except FileExistsError:
                pass
                
            try:
                directory = 'ZNE_Components'
                path = os.path.join(pro_ev_dir, directory)
                os.mkdir(path)
                print('Directory %s created' %directory )
            except FileExistsError:
                pass
            
            stream = obspy.read(un_pro_ev_dir+"/processed/" + '*')
            
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

            #stream = stream+stream_2
            print(stream)
            
            
            ststring = [0] * len(stream)
            ststring_og = [0] * len(stream)
            for i in range(0, len(stream)):
                ststring[i] = stream[i].stats.network + '.' + stream[i].stats.station
                #ststring_og[i] = stream[i].get_id() #includes component, problematic to remove.
                print('ID', ststring[i])
        
            stations = np.unique(ststring)
            #print(stations[0][3:])
            
            failure = 0
            int_components = 0
            NE_components = 0
                
            for j in range (len(stations)):
                N = 0
                E = 0
                Z = 0
                N1 = 0
                E2 = 0
                N_pop = 0
                E_pop = 0
                Z_pop = 0
                
                print()
                sta = stations[j][3:]
                #print(stations[j][:2])
                st = stream.select(network = stations[j][:2], station = stations[j][3:])
                #st = stream.select(network = 'US', station = 'KSU1')
                print(st)
                
                try:
                    if len(st.select(component='N')) == 1:
                        N = st.select(component='N')
                    elif len(st.select(component='N')) > 1:
                        if len(st.select(channel='BHN')) > 0:
                            N = st.select(channel='BHN')
                        else:
                            N = st.select(channel='HHN')
                        
                        if len(N) > 1:
                            channels = []
                            for i in range (len(N)):
                                channels.append(N[i].get_id()[-6:-4])
                                if '..' in N[i].get_id(): 
                                    N = N[i]
                                    N = Stream(traces=[N])
                                    N_pop = 1
                                    break
                            if N_pop == 0:
                                index = np.argmin(channels)
                                N = N[index]
                                N = Stream(traces=[N])

    
                except:
                    print('No N component')
                    
                try:
                    if len(st.select(component='E')) == 1:
                        E = st.select(component='E')
                    elif len(st.select(component='E')) > 1:
                        if len(st.select(channel='BHE')) > 0:
                            E = st.select(channel='BHE')
                        else:
                            E = st.select(channel='HHE')
                        
                        if len(E) > 1:
                            channels = []
                            for i in range (len(E)):
                                channels.append(E[i].get_id()[-6:-4])
                                if '..' in E[i].get_id(): 
                                    E = E[i]
                                    E = Stream(traces=[E])
                                    E_pop = 1
                                    break
                            if E_pop == 0:
                                index = np.argmin(channels)
                                E = E[index]
                                E = Stream(traces=[E])

    
                except:
                    print('No E component')
                    
                try:
                    if len(st.select(component='Z')) == 1:
                        Z = st.select(component='Z')
                    elif len(st.select(component='Z')) > 1:
                        if len(st.select(channel='BHZ')) > 0:
                            Z = st.select(channel='BHZ')
                        else:
                            Z = st.select(channel='HHZ')
                            
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
                except:
                    print('No Z component')
                
                              
                if N == 0 and E ==0:
                    try:
                        if len(st.select(component='1')) == 1:
                            N1 = st.select(component='1')
                        elif len(st.select(component='1')) > 1:
                            if len(st.select(channel='BH1')) > 0:
                                N1 = st.select(channel='BH1')
                            else:
                                N1 = st.select(channel='HH1')
                            
                            if len(N1) > 1:
                                channels = []
                                for i in range (len(N1)):
                                    channels.append(N1[i].get_id()[-6:-4])
                                    if '..' in N1[i].get_id(): 
                                        N1 = N1[i]
                                        N1 = Stream(traces=[N1])

                                        N_pop = 1
                                        break
                                if N_pop == 0:
                                    index = np.argmin(channels)
                                    N1 = N1[index]
                                    N1 = Stream(traces=[N1])

                    except Exception as e:
                        print('No 1 component', e)
                        
                    try:
                        if len(st.select(component='2')) == 1:
                            E2 = st.select(component='2')
                        elif len(st.select(component='2')) > 1:
                            if len(st.select(channel='BH2')) > 0:
                                E2 = st.select(channel='BH2')
                            else:
                                E2 = st.select(channel='HH2')
                            
                            if len(E2) > 1:
                                channels = []
                                for i in range (len(E2)):
                                    channels.append(E2[i].get_id()[-6:-4])
                                    if '..' in E2[i].get_id(): 
                                        E2 = E2[i]
                                        E2 = Stream(traces=[E2])
                                        E_pop = 1
                                        break
                                if E_pop == 0:
                                    index = np.argmin(channels)
                                    E2 = E2[index]
                                    E2 = Stream(traces=[E2])

                        #print('E2', E2)
        
                    except:
                        print('No 2 component')
                                            
                    print()
                    print('Final Traces:')
                    print('N1', N1)
                    print('E2', E2)
                    print('Z', Z)
                    
                else:
                    print()
                    print('Final Traces:')
                    print('N', N)
                    print('E', E)
                    print('Z', Z)
                    
                if (N == 0 or E == 0) and (N1 == 0 and E2 == 0) or Z == 0:
                    failure += 1
                    print('FAILED - MISSING TRACE')
                    f.write('FAILURE: MISSING TRACE for ' + str(stations[j]) +'\n')
                    f.write(str(N) + '\n')
                    f.write(str(E) + '\n')
                    f.write(str(N1) + '\n')
                    f.write(str(E2) + '\n')
                    f.write(str(Z) + '\n')
                    f.write('\n')
                    print('==================================================================================================')
                    continue
                    
                if N != 0 and E != 0:
                    stream_rotate = Z + N + E
                    NE_components += 1
                    try:
                        for k in range (len(stream_rotate)):
                            #print(stream_rotate[k].get_id())
                            rname = un_pro_ev_dir + '/resp/STXML.' + stream_rotate[k].get_id()
                            #print(rname)
                            if k == 0:
                                inv = obspy.read_inventory(rname)
                            if k > 0:
                                inv = inv + obspy.read_inventory(rname)
                        print()
                        print('Final *non-Rotated* Traces:')
                        print(stream_rotate)
                        f.write(str(stream_rotate) + '\n')
                        f.write('\n')
                    except Exception as e:
                        print(e)
                        failure += 1
                        f.write('FAILURE: ' + str(stations[j]) + '\n')
                        f.write(str(e))
                        f.write(str(N) + '\n' + str(E) + '\n')
                        f.write('\n')
                        continue
                    
                    print('---------------------------------------------------------------------------------------------------')
                # Rotate BH1/BH2 into ZNE using inventory held correction angle                 
                if N1 != 0 and E2 != 0:
                    # rotate N1, E1
                    print()
                    print('ROTATION')
                    try:
                        #stream_tmp = Stream(traces=[N1,E2])
                        stream_rotate = Z + N1 + E2
                        f.write(str(stream_rotate) + '\n')
                        f.write('\n')
                        int_components += 1
                        inv = 0
                        for k in range (len(stream_rotate)):
                            print(stream_rotate[k].get_id())
                            rname = un_pro_ev_dir + '/resp/STXML.' + stream_rotate[k].get_id()
                            print(rname)
                            if k == 0:
                                inv = obspy.read_inventory(rname)
                            if k > 0:
                                inv = inv + obspy.read_inventory(rname)
                        #stream_rotate.plot()
                        stream_rotate.rotate('->ZNE', inventory = inv)
                        #stream_rotate.plot()
                        print()
                        print('Final Rotated Traces:')
                        print(stream_rotate)
                        print('---------------------------------------------------------------------------------------------------')
                        
                    except Exception as e:
                        print(e)
                        failure += 1
                        f.write('FAILURE: ' + str(stations[j]) + '\n')
                        f.write(str(e))
                        f.write(str(N1) + '\n' + str(E2) + '\n')
                        f.write('\n')
                        continue
                    
                    #print(stream_components)
                    
                print('Stream to save out: ', stream_rotate)
                #print(N, E, N1, E2)
                #print(len(stream_rotate))
                if (N != 0 and E != 0) or (N1 != 0 and E2 != 0):
                # Save out rotated streams
                    # Save MSEEDS
                    for i in range(0, len(stream_rotate)):
                        tr = stream_rotate[i] 
                        tr.write(pro_ev_dir+ "/ZNE_Components/"+tr.stats.network+"."+tr.stats.station + "." + tr.stats.channel + '.MSEED', format = 'MSEED')
                    inv.write(pro_ev_dir+ "/ZNE_Components/"+tr.stats.network+"."+tr.stats.station +'.xml', format = 'stationxml')
                print('==================================================================================================')
                
            print('Failed: ', failure) 
            f.write('N or E components: ' + str(NE_components) + '\n')
            f.write('1 or 2 components: '+ str(int_components) + '\n')
            f.write('Failed stations: '+ str(failure) + '\n')
                
        # Process N and E data
        stream_N = obspy.read(pro_ev_dir + "/ZNE_Components/*.*HN.MSEED")
        stream_E = obspy.read(pro_ev_dir + "/ZNE_Components/*.*HE.MSEED")
        stream = stream_N + stream_E
        ntraces = len(stream)
        print(stream[0], stream[1])
        print(stream_E[0],stream_E[1])
                
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
            
        print(ststring)

        for i in range(0,ntraces):
            try:
                #rname = un_pro_ev_dir + '/resp/STXML.' + stream[i].stats.network + '.' + stream[i].stats.station+'*BHE'
                tr = stream[i]
                rname = pro_ev_dir+ "/ZNE_Components/"+tr.stats.network+"."+tr.stats.station +'.xml'
                #rname = un_pro_ev_dir + '/resp/STXML.' + ststring[i]
                print(i, 'out of', ntraces-1,':', rname)  #, end='\r')
                print(ststring[i])
                print('1')
                inv=obspy.read_inventory(rname) #creates a one trace inventory 
            except:
                try:
                    rname = un_pro_ev_dir + '/resp/STXML.' + ststring[i][:-1]+'1'
                    print(i, 'out of', ntraces-1,':', rname)  #, end='\r')
                    print(ststring[i])
                    print('2')
                    inv=obspy.read_inventory(rname) #creates a one trace inventory 
                except:
                    rname = un_pro_ev_dir + '/resp/STXML.' + stream[i].stats.network + '.' + stream[i].stats.station + '*'
                    print(i, 'out of', ntraces-1,':', rname)  #, end='\r')
                    print(ststring[i])
                    print('3')
                    inv=obspy.read_inventory(rname) #creates a one trace inventory 
            print(inv)
            stnet[i]=stream[i].stats.network
            nsamples[i] = (stream[i].stats.npts)
            try:
                print('4')
                statcoords = inv.get_coordinates(ststring[i],origin_time)
            except:
                try:
                    print('5')
                    statcoords = inv.get_coordinates(str(ststring[i][:-1])+'1',origin_time)
                except:
                    print('6')
                    statcoords = inv.get_coordinates(str(ststring[i][:-1])+'Z')
                    #statcoords = inv.get_coordinates('AZ.FLV2.30.BHZ')
                    
            stla[i] = (statcoords[u'latitude'])
            stlo[i] = (statcoords[u'longitude'])
            stel[i] = (statcoords[u'elevation'])
            #print(len(inv))
        
        # Plot loaded stations relative to earthquake ('flat', or 'global')
        #figure = ev_station_plot(df, evla, evlo)
        
        # WRITE OUT PROCESSED DATA ======================================================================================
        print('pro dir', pro_ev_dir)
        
        name = 'stname_S.npy'
        path = os.path.join(pro_ev_dir + '/Arrays', name)
        np.save(path, stname, allow_pickle=True)
        
        #name = 'ststring.npy'
        #path = os.path.join(pro_ev_dir + '/Arrays', name)
        #np.save(path, ststring, allow_pickle=True)
                
        name = 'stla_S.npy'
        path = os.path.join(pro_ev_dir + '/Arrays', name)
        np.save(path, stla, allow_pickle=True)
        
        name = 'stlo_S.npy'
        path = os.path.join(pro_ev_dir + '/Arrays', name)
        np.save(path, stlo, allow_pickle=True)
        
        name = 'stnet_S.npy'
        path = os.path.join(pro_ev_dir+ '/Arrays', name)
        np.save(path, stnet, allow_pickle=True)    
        
        name = 'stel_S.npy'
        path = os.path.join(pro_ev_dir+ '/Arrays', name)
        np.save(path, stel, allow_pickle=True) 
        
        # Save MSEEDS
        for i in range(0, len(stream)):
            tr = stream[i]
            print(tr) 
            tr.detrend(type='simple') # Detrend waveforms
            tr.detrend(type='demean') # Demean waveforms
            tr.taper(tpratio) # Taper waveforms
            tr.data = bandpass(tr.data, frqmin, frqmax,tr.stats.sampling_rate, 3, True) # Filter waveforms
            tr.normalize() # Normalise waveforms
            tr.resample(rsample)
            tr.write(pro_ev_dir+ "/Data/"+tr.stats.network+"."+tr.stats.station + "." + tr.stats.channel + '.MSEED', format = 'MSEED')
        
        # Save XML for Inventory 
        #pro_stream = obspy.read(pro_ev_dir+ "/Data/*"+'.MSEED')
        for i in range(0, len(stream)):
            tr = stream[i]
            try:
                rname = pro_ev_dir+ "/ZNE_Components/"+tr.stats.network+"."+tr.stats.station +'.xml'
                print(rname)
                inv=obspy.read_inventory(rname) #creates a one trace inventory
            except:
                rname = un_pro_ev_dir + '/resp/STXML.' + tr.get_id()
                print(rname)
                inv=obspy.read_inventory(rname) #creates a one trace inventory
            inv.write(pro_ev_dir+ "/Stations/"+tr.stats.network+"."+tr.stats.station + "." + tr.stats.channel +'.xml', format = 'stationxml')
        
        f.write('Final no. of traces post processing = ' + str(len(stream)) + '\n')
        f.close()

        try:
            shutil.rmtree(pro_ev_dir + "/ZNE_Components")  # Remove initially processed data - no longer needed
        except OSError:
            print('ZNE Components directory not deleted')
            pass
        return

    except Exception as e:
        print('Data for event %s failed to download/be processed' %evname)
        print(e, ' Error on line {}'.format(sys.exc_info()[-1].tb_lineno))
        txt_file_name = 'Failed_Events_from_1S.txt'
        outputs_txt = os.path.join('.', txt_file_name)
        
        f = open(outputs_txt, 'a+')
        f.write(str(evname) + '\t' + str(e) + '\n')
        f.close()
        pass

# ======= MAIN ==========
def process_NE_components(catalogue, event, re_processing, data_dir, parent_dir):
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

    print('N/E component data for %s processed.' %evname)
    return
