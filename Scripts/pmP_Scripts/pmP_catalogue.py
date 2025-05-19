import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
import matplotlib.ticker as mticker
import matplotlib.colors as clrs
import scipy
from matplotlib import cm
from matplotlib.path import Path
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import rasterio
from scipy.interpolate import griddata

import cartopy as ctp # for making maps
import cartopy.crs as ccrs
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter,
                                LatitudeLocator, LongitudeLocator)
from math import radians
from sklearn.neighbors import BallTree
import numpy as np
import pandas as pd
import re
import math
import copy
import struct
from obspy.geodetics.base import gps2dist_azimuth

def assemble_clean_pmP_cat(uncleaned_pmP_cat, final_3D_EQ_cat, obspydmt_cat_name, cleaned_pmP_cat):

    #Load data points
    evname, eventid, baz, gcarc, crust1_depth, lat, lon, pmP_pw_amp, pP_pw_amp, pmP_amp, pP_amp, delay_time, moho_depth = np.loadtxt(uncleaned_pmP_cat, skiprows=1, unpack=True, usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12))

    print('No. of Data Points:', len(lat))
    print()

    # Extract event details per pmP data point
    ev_x = []
    ev_y = []
    ev_z = []
    for i in range (len(eventid)):
        flag = False
        ev = str(int(eventid[i]))
        #print(ev)
        
        with open(final_3D_EQ_cat) as f:
            df = f.readlines()
        for line in df:
            if ev in line:
                flag = True
                l = re.split('\t', line)
                x = float(l[5])
                y = float(l[4])
                z = float(l[10])
                
                x_pmP = lon[i]
                y_pmP = lat[i]            
            f.close()

        if flag == False:
            with open(obspydmt_cat_name) as f:
                df = f.readlines()
            for line in df:
                if ev in line:
                    flag = True
                    l = re.split('\t', line)
                    x = float(l[4])
                    y = float(l[3])
                    z = float(l[5])
                    
                    x_pmP = lon[i]
                    y_pmP = lat[i]                
            f.close()
            
        ev_x.append(x)
        ev_y.append(y)
        ev_z.append(z)

        #print(x, y, z)

    # Filter out events less than 40 km
    ev_x = np.array(ev_x)
    ev_y = np.array(ev_y)
    ev_z = np.array(ev_z)
    rel_pw_amp = pmP_pw_amp/pP_pw_amp

    evname = evname[ev_z >= 40]
    eventid = eventid[ev_z >= 40]
    baz = baz[ev_z >= 40]
    gcarc = gcarc[ev_z >= 40]
    crust1_depth = crust1_depth[ev_z >= 40]
    lat = lat[ev_z >= 40]
    lon = lon[ev_z >= 40]
    pmP_amp = pmP_amp[ev_z >= 40]
    pP_amp = pP_amp[ev_z >= 40]
    moho_depth = moho_depth[ev_z >= 40]
    delay_time = delay_time[ev_z >= 40]
    ev_x = ev_x[ev_z >= 40]
    ev_y = ev_y[ev_z >= 40]
    rel_pw_amp = rel_pw_amp[ev_z >= 40]
    ev_z = ev_z[ev_z >= 40]

    print('No. of Data Points:', len(ev_x))
    print()

    # Filter out events more than 350 km
    evname = evname[ev_z <= 350]
    eventid = eventid[ev_z <= 350]
    baz = baz[ev_z <= 350]
    gcarc = gcarc[ev_z <= 350]
    crust1_depth = crust1_depth[ev_z <= 350]
    lat = lat[ev_z <= 350]
    lon = lon[ev_z <= 350]
    pmP_amp = pmP_amp[ev_z <= 350]
    pP_amp = pP_amp[ev_z <= 350]
    moho_depth = moho_depth[ev_z <= 350]
    delay_time = delay_time[ev_z <= 350]
    ev_x = ev_x[ev_z <= 350]
    ev_y = ev_y[ev_z <= 350]
    rel_pw_amp = rel_pw_amp[ev_z <= 350]
    ev_z = ev_z[ev_z <= 350]


    print('No. of Data Points:', len(ev_x))
    print('No. Events with pmP:', len(set(eventid)))
    
    # Remove data points which are likely to have picked unwanted P wave coda
    evname = evname[rel_pw_amp <= 1.72]
    eventid = eventid[rel_pw_amp <= 1.72]
    baz = baz[rel_pw_amp <= 1.72]
    gcarc = gcarc[rel_pw_amp <= 1.72]
    crust1_depth = crust1_depth[rel_pw_amp <= 1.72]
    lat = lat[rel_pw_amp <= 1.72]
    lon = lon[rel_pw_amp <= 1.72]
    pmP_amp = pmP_amp[rel_pw_amp <= 1.72]
    pP_amp = pP_amp[rel_pw_amp <= 1.72]
    moho_depth = moho_depth[rel_pw_amp <= 1.72]
    delay_time = delay_time[rel_pw_amp <= 1.72]
    ev_y = ev_y[rel_pw_amp <= 1.72]
    ev_z = ev_z[rel_pw_amp <= 1.72]
    ev_x = ev_x[rel_pw_amp <= 1.72]
    rel_pw_amp = rel_pw_amp[rel_pw_amp <= 1.72]

    print('No. of Data Points:', len(ev_x))
    print('No. Events with pmP:', len(set(eventid)))

    ## Write out new final catalogue ##
    f = open(cleaned_pmP_cat, 'w+')
    f.write('Event'.ljust(15) + '\t' + 'Event_ID'.ljust(10) + '\t' + 'Baz'.ljust(8) + '\t' + 'Gcarc'.ljust(8) + '\t' + 'Cr1_thk'.ljust(8) + '\t' + 'Bpt_Lat'.ljust(8) + '\t' + 'Bpt_Lon'.ljust(8) + '\t' + 'pmP_amp_pw'.ljust(8) + '\t' + 'pP_amp_pw'.ljust(8) + '\t' + 'pmP_amp'.ljust(8) + '\t' + 'pP_amp'.ljust(8) + '\t' + 'pP-pmP_dt'.ljust(10) + '\t' + 'New_Cr_thk'.ljust(12) + '\n')

    for i in range (len(evname)):
        #if (subarray.pmP_pw_amplitude/subarray.pP_pw_amplitude) <= 1.72:
        f.write(str(int(evname[i])).ljust(15) + '\t')
        f.write(str(eventid[i]).ljust(10) + '\t')
        f.write(str(np.round(baz[i],2)).ljust(8) + '\t')
        f.write(str(np.round(gcarc[i],2)).ljust(8) + '\t')
        f.write(str(np.round(crust1_depth[i],2)).ljust(8) + '\t')
        f.write(str(np.round(lat[i],4)).ljust(8) + '\t')
        f.write(str(np.round(lon[i],4)).ljust(8) + '\t')
        f.write(str(np.round(pmP_amp[i], 4)).ljust(8) + '\t')
        f.write(str(np.round(pP_amp[i], 4)).ljust(8) + '\t')             
        f.write(str(np.round(delay_time[i],4)).ljust(10) + '\t')
        f.write(str(np.round(moho_depth[i],4)).ljust(12) + '\n')

    f.close() 
    return

