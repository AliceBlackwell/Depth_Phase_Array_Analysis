#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 13:43:04 2023

Author: Hanna-Riia Allas (earha@leeds.ac.uk)

Functions to use in Crustal_thickness_code
"""

# Importing modules

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import csv
import os
import rasterio

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

import obspy
from obspy.taup import taup_create

# List of functions starts here:

#==============================================================================
def Get_crustal_thickness(evla, evlo, event_name, vel_dir):
    """
    This function extracts the crustal thickness value at event coordinates from Crust1 data
    Inputs:
        evla: event latitude in degrees
        evlo: event longitude in degrees
    Returns:
        crthk: crustal thickness at specified coordinates
    """
    
    # Load in crustal model data
    
    #ev_dir = os.path.join(res_dir, '%s' %event_name)
    #vel_models = os.path.join(ev_dir, 'Velocity_model_files')
    
    crm_path = os.path.join(vel_dir, '%s' %'crsthk.xyz')                              # filepath to the crustal thickness model Crust1.0
    crthk_df = pd.read_csv(crm_path, skiprows=1, delimiter="\s+",                        # read the xyz file into Python dataframe
                           header=None, names=["Longitude", "Latitude", "Depth"])                               

    # Create arrays of Crust1 coordinates to use in search for event coordinates

    lon_array = np.array(crthk_df["Longitude"])
    lat_array = np.array(crthk_df["Latitude"])

    # Find coordinates closest to event coordinates to extract crustal thickness value

    lonidx = []
    abslondiff = []
    for i in range(0, len(lon_array)):                                                   # create an array of row indices in crust1 which correspond to longitude values within 1 degree of event longitude
        diff = abs(lon_array[i]-evlo)
        abslondiff.append(diff)
        if abslondiff[i] < 0.5:
            lonidx.append(i)

    latidx = []
    abslatdiff =[]
    for i in range(0, len(lat_array)):                                                   # create an array of row indices in crust1 which correspond to latitude values within 1 degree of event latitude
        diff = abs(lat_array[i]-evla)
        abslatdiff.append(diff)
        if abslatdiff[i] < 0.5:
            latidx.append(i)

    latlonidx = np.intersect1d(lonidx, latidx)                                           # find index of row in crustal model which corresponds to event coordinates by finding the index present in both arrays above
                                          
    crthk = crthk_df["Depth"].values[latlonidx][0]                                       # extract crustal thickness at event coordinates

    return crthk

#==============================================================================

def Build_velocity_model(crthk, event_name, crustal_velocity_model, vel_dir):
   """ 
   This function builds the velocity model used for phase arrival time predictions,
   with a custom-defined crustal thickness, using ak135 as base.
   
   Inputs:
       crthk: crustal thickness at event coordinates
   Returns:
       custom_model: taup model used for arrival time prediction
   """ 
   # Define filepaths
   
   #ev_dir = os.path.join(res_dir, '%s' %event_name)
   #vel_models = os.path.join(output_dir, 'Velocity_model_files')
   
   # Load in the ak135 model

   ak135_path = os.path.join(vel_dir, '%s' %'ak135.tvel')                            # filepath to the ak135 model text file
   ak135_arr = np.loadtxt(ak135_path, skiprows = 2)                                     # load the velocity model into an ndarray; columns are depth, Vp, Vs and density. Skip first 2 rows of text

   # Load in the crustal model

   crust_path = os.path.join(vel_dir, crustal_velocity_model)                    # filepath to the text file with crustal velocity profile
   crust_arr = np.loadtxt(crust_path, delimiter=",")                                    # load crustal velocities into an ndarray
   #print(crust_arr)
   
   # Set the Moho depth by changing the Moho depth in the array to crthk

   crust_arr[-1, 0] = crthk
   crust_arr[-2, 0] = crthk

   # Remove anything above the new Moho from the ak135 model
   
   if crthk > 35:                                                                       # check that the custom crustal thickness is greater than ak135 default value of 35km
       mask = ak135_arr[:, 0] >= crthk                                                  # check whether values in column 0 (i.e. depth) are smaller than or equal to the new value of crustal thickness, create a boolean mask
       ak135_nocrust = ak135_arr[mask]                                                  # create a version of the ak135f model with all layers above the crustal thickness value removed
   else:
       mask = ak135_arr[:, 0] >= 36                                                     # if crust is thinner, need to remove everything above and including 35km from the ak135
       ak135_nocrust = ak135_arr[mask]

   # Replace bottom of Crust 1.0 averages model with top of ak135f values, at crthk depth
   crust_arr[-1][1] = ak135_nocrust[0][1]
   crust_arr[-1][2] = ak135_nocrust[0][2]
   crust_arr[-1][3] = ak135_nocrust[0][3]
   
   # Append the new crustal model to the crust-less ak135f

   ak_w_crt = np.vstack((crust_arr, ak135_nocrust))
   #print(ak_w_crt)
   
   # Write into a tvel file for ObsPy

   ak_modified_path = os.path.join(vel_dir, '%s' %'ak135_modified.tvel')             # define path to  tvel file with custom velocity data

   with open(ak_modified_path, 'w') as txt_file:
       opening_lines_to_write = ["ak135 - P\n",                                         # need to include two text lines to conform to ObsPy tvel file formatting requirements
                                 "ak135 - S\n"]
       txt_file.writelines(opening_lines_to_write)
       csv.writer(txt_file, delimiter=' ').writerows(ak_w_crt)

   # Convert the new model tvel file into an ObsPy custom model

   taup_create.build_taup_model(ak_modified_path, output_folder=vel_dir)
   custom_model = obspy.taup.TauPyModel(model=os.path.join(vel_dir, '%s' %'ak135_modified.npz'))
    
   return custom_model

#==============================================================================
def Crustal_thickness_forward_modelling(subarray_list, first_pass_crthk, evdepth, event_name, crustal_velocity_model, vel_dir):
    """
    Finds best-fit crustal thickness by using a forward modelling approach. 
    The function builds a velocity model for a range of crustal thickness values
    around the first-pass value and calculates the pmP-pP arrival time 
    differential for each event-subarray pair in each case using Taup. It then finds
    the best-fit value and saves it into each Cr_subarray object.
    
    This step not carried out within the Cr_subarray class because of computing 
    efficiency (building velocity models for each crustal thickness test value is
    time-consuming, best done once per event in the main code).
    
    Inputs:
        
        subarray_list: list of the instances of Cr_subarray objects for which crustal thickness will be calculated
        from subarray_list[idx]:
            subarray.outputs.rsample: rate of sampling
            subarray.outputs.final_picks: Alice's final depth phase picks
            subarray.pmP_pick: pmP arrival picked for the subarray
            subarray.outputs.ev_subarray_gcarc: event-subarray distance in degrees
            
        first_pass_crthk: first-pass crustal thickness value
        evdepth: event depth
            
    Returns:
        
        subarray.new_crthk_FM: new improved crustal thickness value from forward modelling approach, saved into the Cr_subarray object      
    """
    
    # Extract variables from function
    
    crthk = first_pass_crthk
    evdepth = evdepth
    event_name = event_name
    
    # Generate range of crustal thicknesses to test, +-10km around first pass value
    
    test_crthk = np.arange((crthk-15), (crthk+15.2), 0.2) # ~0.2 km intervals
    
    test_crthk[test_crthk < 0] = np.nan
    test_crthk = test_crthk[~np.isnan(test_crthk)]
    
    test_crthk[test_crthk > evdepth] = np.nan
    test_crthk = test_crthk[~np.isnan(test_crthk)]
 
    # Create an array to store model - observation residuals for each subarray being iterated over
    
    subarray_residuals = np.zeros((len(subarray_list), len(test_crthk)))
    
# -------- Enter model-building loop --------   
     
    # Build model for each test crustal thickness value, then calculate model pmP-pP difference for each subarray using that crustal thickness
    
    for i in range(0, len(test_crthk)):
        model = Build_velocity_model(test_crthk[i], event_name, crustal_velocity_model, vel_dir)
        
        for j in range(len(subarray_list)):
            
            subarray = subarray_list[j]
            #print(subarray.outputs.ev_subarray_gcarc)
            #print("pmP-pP difference for first-pass crthk:", subarray.pmP_pP_diff_m, "\npmP-pP difference from data:", subarray.pP_pmP_delay_time)
            
            # Extract variables from each Cr_subarray object
    
            pmP_pick_exists = subarray.pmP_pick_exists
            ev_sa_dist = subarray.outputs.ev_array_gcarc
            pP_pmP_delay_time = subarray.pP_pmP_delay_time
        
            # Check that a pmP pick exists within the subarray (shouldn't be an issue at this point but just in case)
        
            if pmP_pick_exists == False:
                print("Cannot calculate crustal thickness, no pmP pick is stored in subarray.")
                return
            else: 
                pass
    
            # Predict model pmP-pP delay time

            arrivals = model.get_travel_times(source_depth_in_km=evdepth,
                                              distance_in_degree=ev_sa_dist,
                                              phase_list=["p^"+str(int(test_crthk[i]))+"P","pP"])
            pmP = arrivals[0].time
            pP = arrivals[1].time
            dt = pP - pmP
            
            # Calculate model-pick residual, save value in residuals array
            
            residual = abs(dt - pP_pmP_delay_time)
            subarray_residuals[j][i] = residual
            
# -------- Exit model-building loop --------
            
    # Extract best-fit crustal thickness for each subarray by finding the minimum residual crustal thickness value
        
    min_col_idx = np.argmin(subarray_residuals, axis=1)
    
    for i in range(len(subarray_list)):
        
        best_fit_crthk = test_crthk[min_col_idx[i]]
            
        # Save new crustal thickness value into the Cr_subarray object
            
        subarray_list[i].new_crthk_FM = best_fit_crthk
        #print("Best fit crustal thickness from forward modelling for subarray at", subarray_list[i].outputs.ev_array_gcarc, "distance is: ", best_fit_crthk)
        
    return

#==============================================================================
def Plot_delay_time_histogram(subarray_list, event_name, fig_dir):
    """
    Plots a histogram of all pP-pmP delay times to see if there's any trends in the data.
    """

    # Create delay time data array to plot
    
    pmP_pP_delay_times = []

    for subarray in subarray_list:
        pmP_pP_delay_times.append(subarray.pP_pmP_delay_time)
        
    # Plot the histogram
    
    plt.hist(pmP_pP_delay_times, bins=10, color='red', edgecolor='black')
    
    plt.xlabel('delay times (s)')
    plt.ylabel('Frequency')
    plt.title('Histogram of pP-pmP delay times for event ' + str(event_name))
    
    # Save figure    
    fig_name = str(event_name) + '_Delay_time_histogram.png'
    path = os.path.join(fig_dir, fig_name)
    plt.savefig(path, dpi=500)
    
    plt.close()
        
    return

#==============================================================================
def Plot_crthk_histogram(subarray_list, event_name, first_pass_crthk, fig_dir):
    """
    Plots a histogram of all calculated crustal thickness values to see if there's any trends in the data.
    """
    
    # Create crustal thickness data array to plot
    
    crthk_values = []
    
    for subarray in subarray_list:
        crthk_values.append(subarray.new_crthk_FM)
        
    # Plot the histogram
        
    plt.hist(crthk_values, bins=5, color='red', edgecolor='black')
    
    plt.xlabel('Crustal thickness (km)')
    plt.ylabel('Frequency')
    plt.title('Histogram of crustal thickness values calculated for event ' + str(event_name))
    
    plt.axvline(x=first_pass_crthk, color='black', linestyle='dashed', linewidth=2, label="First-pass crustal thickness")
    
    plt.legend()
    
    # Save figure    
    fig_name = str(event_name) + '_Crthk_histogram.png'
    path = os.path.join(fig_dir, fig_name)
    plt.savefig(path, dpi=500)
    
    plt.close()
    
    return

#==============================================================================
def Plot_bounce_points(subarray_list, event_name, evla, evlo, first_pass_crthk, fig_dir, gen_dir):
    """
    Plots bounce point coordinates for a list of subarrays with a pmP pick. 
    First subplot shows bounce points from first-pass crustal thickness, second
    subplot shows bounce points calculated using the recalculated crustal thickness values
    for the subarrays, with the crustal thicknesses displayed as annotations.
    """
    
    print("Now plotting the bounce point figure for subarrays with pmP picks only.")
    
    # Initialise variables
    
    crthk = first_pass_crthk
    
    # Initialise data array for first-pass plot
    
    BP_data_single_picks_m = np.zeros((len(subarray_list), 4))

    for i in range(len(subarray_list)):
        BP_data_single_picks_m[i][0] = subarray_list[i].pmP_bounce_lat_m
        BP_data_single_picks_m[i][1] = subarray_list[i].pmP_bounce_lon_m
        BP_data_single_picks_m[i][2] = subarray_list[i].pP_bounce_lat_m   
        BP_data_single_picks_m[i][3] = subarray_list[i].pP_bounce_lon_m
           
    # Initialise lists of coordinates for first-pass plot
    
    pmP_BP_lat_m = [0] * len(subarray_list)
    pmP_BP_lon_m = [0] * len(subarray_list)   
    pP_BP_lat_m = [0] * len(subarray_list)
    pP_BP_lon_m = [0] * len(subarray_list)
    
    for i in range(len(subarray_list)):
        pmP_BP_lat_m[i] = BP_data_single_picks_m[i][0]
        pmP_BP_lon_m[i] = BP_data_single_picks_m[i][1]
        pP_BP_lat_m[i] = BP_data_single_picks_m[i][2]
        pP_BP_lon_m[i] = BP_data_single_picks_m[i][3]
    
    # Initialise data array for improved BP plot
    
    BP_data_single_picks = np.zeros((len(subarray_list), 5))

    for i in range(len(subarray_list)):
        BP_data_single_picks[i][0] = subarray_list[i].pmP_bounce_lat
        BP_data_single_picks[i][1] = subarray_list[i].pmP_bounce_lon
        BP_data_single_picks[i][2] = subarray_list[i].pP_bounce_lat  
        BP_data_single_picks[i][3] = subarray_list[i].pP_bounce_lon
        BP_data_single_picks[i][4] = subarray_list[i].new_crthk_FM # CHANGE BACK TO FM
        
    # Initialise lists of coordinates for improved BP plot

    pmP_BP_lat = [0] * len(subarray_list)
    pmP_BP_lon = [0] * len(subarray_list)   
    pP_BP_lat = [0] * len(subarray_list)
    pP_BP_lon = [0] * len(subarray_list)
    
    for i in range(len(subarray_list)):
        pmP_BP_lat[i] = BP_data_single_picks[i][0]
        pmP_BP_lon[i] = BP_data_single_picks[i][1]
        pP_BP_lat[i] = BP_data_single_picks[i][2]
        pP_BP_lon[i] = BP_data_single_picks[i][3]   
        
    # List crustal thickness values for improved BP plot

    crthk_list = [0] * len(subarray_list)
    
    for i in range(len(subarray_list)):
        crthk_list[i] = BP_data_single_picks[i][4]
    
# ---------- PLOT STARTS HERE ----------
        
    # Initialise figure

    BP_fig = plt.figure(figsize=(20, 10))
    
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.3, figure = BP_fig)
    
    ax0 = plt.subplot(gs[0], projection = ccrs.PlateCarree())
    ax1 = plt.subplot(gs[1], projection = ccrs.PlateCarree())
    
    cax = plt.subplot(gs[2])
    
    cax.set_frame_on(True)
    cax.xaxis.set_visible(True)
    cax.yaxis.set_visible(True)

    ax = [ax0, ax1, cax]                          
                  
    # Make lists of coordinates to search for min/max values to set map limits
    
    longitudes = pmP_BP_lon + pP_BP_lon + pmP_BP_lon_m + pP_BP_lon_m
    longitudes.append(evlo)
    
    latitudes = pmP_BP_lat + pP_BP_lat + pmP_BP_lat_m + pP_BP_lat_m
    latitudes.append(evla)     
    
    # Define bounding box for the map region
    
    lon_min = np.round(min(longitudes)-0.25, 1)
    lon_max = np.round(max(longitudes)+0.25, 1)
    lat_min = np.round(min(latitudes)-0.25, 1)
    lat_max = np.round(max(latitudes)+0.25, 1)
       
    # Import topography data
    
    elevation_file = os.path.join(gen_dir, 'Elevation_data_files/output_SRTMGL3.tif') # digital elevation data for all of Peru
    
    with rasterio.open(elevation_file) as dataset: # read Peru elevation data into an array
        elevation_data = dataset.read(1)
    
    # Downsample the elevation dataset (because it is massive and the code won't run with a 90m resolution elevation dataset)
    
# -----------------------------------------------------------------------------   
    def downsample(input_array, factor):
        
        # get the shape of input array
        rows, cols = input_array.shape
        
        # calculate number of blocks in each dimension
        new_rows = rows // factor
        new_cols = cols // factor
        
        # reshape input array into blocks of the specified factor
        blocks = input_array[:new_rows * factor, :new_cols * factor].reshape(new_rows, factor, new_cols, factor)
        
        # compute mean within each block
        downsampled_array = blocks.mean(axis=(1, 3))
        
        return downsampled_array
# -----------------------------------------------------------------------------
    
    factor = 5 # set downsampling factor for the data
    downsampled_elevation = downsample(elevation_data, factor)
    elevation_data = downsampled_elevation
        
    # Create a coordinate array corresponding to the (downsampled) Peru elevation data. 
    # Coordinates are from data extraction metadata, need to change if changing the input DEM file

    elevation_data_coords = np.zeros((np.shape(elevation_data)[0], np.shape(elevation_data)[1], 2))
        
    elevation_data_latitudes = np.linspace(0.627581444629, -18.8010181096, num = np.shape(elevation_data)[0])    
    elevation_data_longitudes = np.linspace(-81.8481445312, -67.6977539062, num = np.shape(elevation_data)[1])
    
    for i in range(np.shape(elevation_data)[0]):
        for j in range(np.shape(elevation_data)[1]):
            elevation_data_coords[i][j] = (elevation_data_latitudes[i], elevation_data_longitudes[j])

    # Find the indices corresponding to the bounding box in the elevation data array by creating a boolean mask
        
    lon_mask = np.logical_and(elevation_data_longitudes >= lon_min, elevation_data_longitudes <= lon_max)
    lat_mask = np.logical_and(lat_max >= elevation_data_latitudes, lat_min <= elevation_data_latitudes)

    # Extract the subset of the elevation data for desired region

    subset_elevation_data = elevation_data[lat_mask][:, lon_mask]
    
    # Add the topographic data to the map
    
    ax[0].imshow(subset_elevation_data,
                 extent=[lon_min, lon_max, lat_min, lat_max],
                 cmap='terrain',
                 origin='upper',
                 alpha = 0.4,
                 transform=ccrs.PlateCarree())
    
    img = ax[1].imshow(subset_elevation_data,
                 extent=[lon_min, lon_max, lat_min, lat_max],
                 cmap='terrain',
                 origin='upper',
                 alpha = 0.4,
                 transform=ccrs.PlateCarree())
    
    # Customise axes
    
    for axis in [ax[0], ax[1]]:
        
        axis.set_xlim(lon_min, lon_max)
        axis.set_ylim(lat_min, lat_max)

        axis.set_xticks(np.linspace(lon_min, lon_max, 6))
        axis.set_yticks(np.linspace(lat_min, lat_max, 6))

        axis.set_xlabel('Longitude', fontsize = 15)
        axis.set_ylabel('Latitude', fontsize = 15)

        axis.xaxis.set_label_coords(0.5, -0.05)
        axis.yaxis.set_label_coords(-0.1, 0.5)
    
        axis.grid(True, zorder=2, linestyle='--', linewidth=0.25, color='gray', alpha=0.5)
        
    # Add colour bar

    cbar = plt.colorbar(img, cax=cax, pad = 0.02)
    cbar.set_label('Elevation (m)', fontsize = 15, labelpad=-80)
    
    # Add country borders
       
#    ax[0].add_feature(cfeature.NaturalEarthFeature('cultural', 'admin_0_countries', '50m', edgecolor='dimgrey', facecolor='none'))
#    ax[1].add_feature(cfeature.NaturalEarthFeature('cultural', 'admin_0_countries', '50m', edgecolor='dimgrey', facecolor='none'))
    
    # Add coastlines
    
    ax[0].coastlines()
    ax[1].coastlines()
    
    # First-pass plot data
    
    ax[0].scatter(pmP_BP_lon_m, pmP_BP_lat_m, zorder=6, c='red', s=20, marker='o', label = "pmP bouncepoints", transform=ccrs.PlateCarree())
    ax[0].scatter(pP_BP_lon_m, pP_BP_lat_m, zorder=5, c='black', s=20, marker='o', label = "pP bouncepoints", transform=ccrs.PlateCarree())
    ax[0].scatter(evlo, evla, c='yellow', s=100, marker='*', label="Event location", transform=ccrs.PlateCarree())
    
    # Improved BP plot data
    
    ax[1].scatter(pmP_BP_lon, pmP_BP_lat, zorder=6, c='red', s=20, marker='o', label = "pmP bouncepoints", transform=ccrs.PlateCarree())
    ax[1].scatter(pP_BP_lon, pP_BP_lat, zorder=5, c='black', s=20, marker='o', label = "pP bouncepoints", transform=ccrs.PlateCarree())
    ax[1].scatter(evlo, evla, c='yellow', s=100, marker='*', label="Event location", transform=ccrs.PlateCarree())
              
    # Plot lines that connect pmP-pP bounce points
    
    for i in range(len(subarray_list)):
        ax[0].plot([pP_BP_lon_m[i], pmP_BP_lon_m[i]], [pP_BP_lat_m[i], pmP_BP_lat_m[i]], zorder = 3, color='black', linestyle='dotted', transform=ccrs.PlateCarree())        
        ax[1].plot([pP_BP_lon[i], pmP_BP_lon[i]], [pP_BP_lat[i], pmP_BP_lat[i]], zorder = 3, color='black', linestyle='dotted', transform=ccrs.PlateCarree())
            
    # Plot titles
    
    ax[0].set_title('First-pass bounce points', fontsize=15)
    ax[1].set_title('Recalculated bounce points', fontsize=15)
    
    BP_fig.suptitle(str(event_name) + ' Bounce Points', fontsize = 18, x=0.5, y=0.95, ha='center')
    
    # Crustal thickness labels
    
    for i in range(len(crthk_list)):
        ax[1].annotate(np.around(crthk_list[i], 1), 
                      ((pmP_BP_lon[i]+pP_BP_lon[i])/2, (pmP_BP_lat[i]+pP_BP_lat[i])/2),
                      zorder=6, 
                      label = 'New crustal thickness values', 
                      transform=ccrs.PlateCarree())
    
    # Text box for first pass plot
    
    bbox_props = dict(boxstyle='square', edgecolor='k', facecolor = 'none')
    ax[0].text(0.95, 0.95, "First-pass crustal thickness = " + str(crthk) + " km", transform = ax[0].transAxes, fontsize=12, ha='right', va='top', color='k', bbox=bbox_props)

    # Legend
    
    ax[0].legend(loc = 'lower left',
                 fontsize=12)
    ax[1].legend(loc = 'lower left',
                 fontsize=12)
    
# ---------- SAVE FIGURE ----------
    
    fig_name = str(event_name) + '_Bounce_points.png'
    path = os.path.join(fig_dir, fig_name)
    BP_fig.savefig(path, dpi=500)
    
    plt.close()

    print("Plotted all bounce point locations for subarrays with pmP picks.")    
    
    return

#==============================================================================
def Plot_all_event_data(subarray_list, event_name, evla, evlo, first_pass_crthk, fig_dir, gen_dir):
    
    """
    This function plots all event data (subarrays bounce points with and without
    pmP picks, crustal thickness values, receiver function crustal thicknesses) onto one map.
    
    Arguments:
        subarray_list: list of all subarrays that entered the pmP search
        evname: event name
        evla: event latitude
        evlo: event longitude
        first_pass_crthk: Crust1 crustal thickness value
        fig_dir: path to figure directory to save the plot
    """
    
    print("Now plotting the summary map with all crustal thickness results.")

    crthk = first_pass_crthk
                 
# ---------- PLOT STARTS HERE ----------
        
    # Initialise figure

    fig_crthk_all = plt.figure(figsize=(15, 10))
    
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 0.05], wspace=0.1, figure = fig_crthk_all)
    
    ax = plt.subplot(gs[0], projection = ccrs.PlateCarree())
    
    cax = plt.subplot(gs[1])
    
    cax.set_frame_on(True)
    cax.xaxis.set_visible(True)
    cax.yaxis.set_visible(True)

    # Make lists of coordinates to search for min/max values to set map limits
    
    longitudes = []
    
    for subarray in subarray_list:
        if subarray.pmP_pick_exists == True:
            longitudes.append(subarray.pmP_bounce_lon)
            longitudes.append(subarray.pP_bounce_lon)
        else:
            longitudes.append(subarray.pmP_bounce_lon_m)
            longitudes.append(subarray.pP_bounce_lon_m)
            
    longitudes.append(evlo)
    
    latitudes = []
    
    for subarray in subarray_list:
        if subarray.pmP_pick_exists == True:
            latitudes.append(subarray.pmP_bounce_lat)
            latitudes.append(subarray.pP_bounce_lat)
        else:
            latitudes.append(subarray.pmP_bounce_lat_m)
            latitudes.append(subarray.pP_bounce_lat_m)
            
    latitudes.append(evla)
    
    # Define limits for the map region
    
    lon_min = np.round(min(longitudes)-0.25, 1)
    lon_max = np.round(max(longitudes)+0.25, 1)
    lat_min = np.round(min(latitudes)-0.25, 1)
    lat_max = np.round(max(latitudes)+0.25, 1)

# ---------- TOPOGRAPHY ----------
       
    # Import topography data
    
    elevation_file = os.path.join(gen_dir, 'Elevation_data_files/output_SRTMGL3.tif') # digital elevation data for all of Peru
    
    with rasterio.open(elevation_file) as dataset: # read Peru elevation data into an array
        elevation_data = dataset.read(1)
    
    # Downsample the elevation dataset (because it is massive and the code won't run with a 90m resolution elevation dataset)
    
# -----------------------------------------------------------------------------   
    def downsample(input_array, factor):
        
        # get the shape of input array
        rows, cols = input_array.shape
        
        # calculate number of blocks in each dimension
        new_rows = rows // factor
        new_cols = cols // factor
        
        # reshape input array into blocks of the specified factor
        blocks = input_array[:new_rows * factor, :new_cols * factor].reshape(new_rows, factor, new_cols, factor)
        
        # compute mean within each block
        downsampled_array = blocks.mean(axis=(1, 3))
        
        return downsampled_array
# -----------------------------------------------------------------------------
    
    factor = 5 # set downsampling factor for the data
    downsampled_elevation = downsample(elevation_data, factor)
    elevation_data = downsampled_elevation
        
    # Create a coordinate array corresponding to the (downsampled) Peru elevation data.
    # Coordinates are from data extraction metadata, need to change if changing the input file

    elevation_data_coords = np.zeros((np.shape(elevation_data)[0], np.shape(elevation_data)[1], 2))
        
    elevation_data_latitudes = np.linspace(0.627581444629, -18.8010181096, num = np.shape(elevation_data)[0])    
    elevation_data_longitudes = np.linspace(-81.8481445312, -67.6977539062, num = np.shape(elevation_data)[1])
    
    for i in range(np.shape(elevation_data)[0]):
        for j in range(np.shape(elevation_data)[1]):
            elevation_data_coords[i][j] = (elevation_data_latitudes[i], elevation_data_longitudes[j])

    # Find the indices corresponding to the bounding box in the elevation data array by creating a boolean mask
        
    lon_mask = np.logical_and(elevation_data_longitudes >= lon_min, elevation_data_longitudes <= lon_max)
    lat_mask = np.logical_and(lat_max >= elevation_data_latitudes, lat_min <= elevation_data_latitudes)

    # Extract the subset of the elevation data for desired region

    subset_elevation_data = elevation_data[lat_mask][:, lon_mask]
    
    # Add the topographic data to the map
    
    img = ax.imshow(subset_elevation_data,
                 extent=[lon_min, lon_max, lat_min, lat_max],
                 cmap='terrain',
                 origin='upper',
                 alpha = 0.4,
                 transform=ccrs.PlateCarree())

# ---------- RECEIVER FUNCTION DATA ----------

    # Load in the receiver function dataset
    
    RF_path = os.path.join(gen_dir, 'stephenson_moho.txt')
    RF_arr = np.loadtxt(RF_path, skiprows = 0)                                  # read the dataset into and array; columns are longitude, latitude, depth
    
    # Choose coordinates within map limits, extract corresponding indices

    lon_cond = (RF_arr[:, 0] <= lon_max) & (RF_arr[:, 0] >= lon_min)
    lat_cond = (RF_arr[:, 1] <= lat_max) & (RF_arr[:, 1] >= lat_min)
    
    subset_RF_data = RF_arr[lon_cond & lat_cond]
    
    # Plot the receiver function data
    
    for i in range(subset_RF_data.shape[0]):
        ax.scatter(subset_RF_data[i, 0], subset_RF_data[i, 1], c='black', s=5, marker='o', alpha=0.5, transform=ccrs.PlateCarree())
        ax.annotate(str(subset_RF_data[i, 2]), (subset_RF_data[i, 0], subset_RF_data[i, 1]), textcoords="offset points", xytext = (5, 5), color = 'black', alpha = 0.5, fontsize = 10, transform = ccrs.PlateCarree())

    RF_legend = ax.scatter([], [], color='black', s= 5, marker = 'o', alpha=1.0, label="Receiver function crustal thicknesses")
    
# -----------------------------------------------------------------------------
       
    # Customise axes
    
    def generate_ticks(start, end):
        tick_values = np.arange(np.ceil(start*2) / 2, np.floor(end*2) / 2 + 0.5, 0.5)
        tick_labels = [f'{val:.1f}' for val in tick_values]
        return tick_values, tick_labels

    xtick_values, xtick_labels = generate_ticks(lon_min, lon_max)
    ytick_values, ytick_labels = generate_ticks(lat_min, lat_max)
                
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    
    ax.set_xticks(xtick_values)
    ax.set_xticklabels(xtick_labels)
    
    ax.set_yticks(ytick_values)
    ax.set_yticklabels(ytick_labels)

    #ax.set_xticks(np.linspace(lon_min, lon_max, 6))
    #ax.set_yticks(np.linspace(lat_min, lat_max, 6))
    
    ax.set_xlabel('Longitude', fontsize = 20)
    ax.set_ylabel('Latitude', fontsize = 20)
        
    ax.xaxis.set_label_coords(0.5, -0.05)
    ax.yaxis.set_label_coords(-0.1, 0.5)

    ax.grid(True, zorder=2, linestyle='--', linewidth=0.25, color='gray', alpha=0.6)  
        
    # Add colour bar

    cbar = plt.colorbar(img, cax=cax, pad = 0.02)
    cbar.set_label('Elevation (m)', fontsize = 15, labelpad=-100)
    
    # Add country borders
       
    #ax.add_feature(cfeature.NaturalEarthFeature('cultural', 'admin_0_countries', '50m', edgecolor='dimgrey', facecolor='none'))
    
    # Add coastlines
    
    ax.coastlines()
    
    # Plot event location
    
    event = ax.scatter(evlo, evla, c='yellow', s=100, marker='*', label="Event location", transform=ccrs.PlateCarree())
        
    # Plot bounce point locations
    
    for subarray in subarray_list:
        if subarray.pmP_pick_exists == False:
            pmP_no_pick = ax.scatter(subarray.pmP_bounce_lon_m, subarray.pmP_bounce_lat_m, zorder=6, s=40, marker='o', facecolors='none', edgecolors='red', alpha=0.8,
                       label = "first-pass pmP bouncepoints, subarrays with no pmP pick", transform=ccrs.PlateCarree())
            pP_no_pick = ax.scatter(subarray.pP_bounce_lon_m, subarray.pP_bounce_lat_m, zorder=5, s=40, marker='o', facecolors='none', edgecolors='black', alpha=0.8,
                       label = "pP bouncepoints, subarrays with no pmP pick", transform=ccrs.PlateCarree())
        else:
            pmP_with_pick = ax.scatter(subarray.pmP_bounce_lon, subarray.pmP_bounce_lat, zorder=6, c='red', s=40, marker='o', alpha=0.8,
                       label = "pmP bouncepoints, subarrays with pmP picks", transform=ccrs.PlateCarree())
            pP_with_pick = ax.scatter(subarray.pP_bounce_lon, subarray.pP_bounce_lat, zorder=5, c='black', s=40, marker='o', alpha=0.8,
                       label = "pP bouncepoints, subarrays with pP picks", transform=ccrs.PlateCarree())
              
    # Plot lines that connect pmP-pP bounce points
    
    subarray_list_no_picks = []
    subarray_list_picks = []
    
    for subarray in subarray_list:
        if subarray.pmP_pick_exists == True:
            subarray_list_picks.append(subarray)
        else:
            subarray_list_no_picks.append(subarray)
    
    for i in range(len(subarray_list_no_picks)):
        ax.plot([subarray_list_no_picks[i].pP_bounce_lon_m, subarray_list_no_picks[i].pmP_bounce_lon_m], [subarray_list_no_picks[i].pP_bounce_lat_m, subarray_list_no_picks[i].pmP_bounce_lat_m],
                zorder = 3, alpha = 0.5, color='black', linestyle='dotted', transform=ccrs.PlateCarree())
    for i in range(len(subarray_list_picks)):        
        ax.plot([subarray_list_picks[i].pP_bounce_lon, subarray_list_picks[i].pmP_bounce_lon], [subarray_list_picks[i].pP_bounce_lat, subarray_list_picks[i].pmP_bounce_lat],
                zorder = 3, alpha = 0.5, color='black', linestyle='dotted', transform=ccrs.PlateCarree())
            
    # Plot title
    
    fig_crthk_all.suptitle(str(event_name) + ' all BPs, crustal thicknesses and RF data', fontsize = 18, x=0.5, y=0.95, ha='center')

    # Text box with first pass crthk value
    
    bbox_props = dict(boxstyle='square', edgecolor='k', facecolor = 'none')
    ax.text(0.95, 0.95, "First-pass crustal thickness = " + str(crthk) + " km",
            transform = ax.transAxes, fontsize=12, ha='right', va='top', color='k', bbox=bbox_props)
    
    # DP crustal thickness labels
    
    for subarray in subarray_list_picks:
        ax.scatter(((subarray.pmP_bounce_lon+subarray.pP_bounce_lon)/2), ((subarray.pmP_bounce_lat+subarray.pP_bounce_lat)/2), c='red', s=20, marker='x', transform=ccrs.PlateCarree())
        ax.annotate(np.around(subarray.new_crthk_FM, 1), ((subarray.pmP_bounce_lon+subarray.pP_bounce_lon)/2, (subarray.pmP_bounce_lat+subarray.pP_bounce_lat)/2), textcoords="offset points", xytext = (5, 5), color = 'black', fontsize = 10, label = 'DP crustal thickness values', transform=ccrs.PlateCarree())
    
    crthk_legend = ax.scatter([], [], color='red', s= 20, marker = 'x', alpha=1.0, label="Calculated crustal thicknesses")

    
    # Legend
    
    if len(subarray_list_no_picks) == 0:
        legend_handles = [event,
                          pmP_with_pick,
                          pP_with_pick,
                          RF_legend,
                          crthk_legend]
    else:
        legend_handles = [event,
                          pmP_with_pick,
                          pP_with_pick,
                          pmP_no_pick,
                          pP_no_pick,
                          RF_legend,
                          crthk_legend]
    
    ax.legend(handles = legend_handles,
              loc = 'lower left',
              fontsize=12)
    
# ---------- SAVE FIGURE ----------
    
    fig_name = str(event_name) + '_Crthk_summary.png'
    path = os.path.join(fig_dir, fig_name)
    fig_crthk_all.savefig(path, dpi=500)
    
    plt.close()

    print("Plotted the summary map.")    
    
    return

#==============================================================================

def write_out_results(subarray_list, event_name, event_id, res_dir, output_file):

    ev_dir = os.path.join(res_dir, str(event_name))
    f = open(ev_dir + '/pmP_catalogue.txt', 'w+')
    f.write('Event'.ljust(15) + '\t' + 'Event_ID'.ljust(10) + '\t' + 'Baz'.ljust(8) + '\t' + 'Gcarc'.ljust(8) + '\t' + 'Cr1_thk'.ljust(8) + '\t' + 'Bpt_Lat'.ljust(8) + '\t' + 'Bpt_Lon'.ljust(8) + '\t' + 'pmP_amp_pw'.ljust(8) + '\t' + 'pP_amp_pw'.ljust(8) + '\t' + 'pmP_amp'.ljust(8) + '\t' + 'pP_amp'.ljust(8) + '\t' + 'pP-pmP_dt'.ljust(10) + '\t' + 'New_Cr_thk'.ljust(12) + '\n')
       
    for subarray in subarray_list:
        if subarray.pmP_pick_exists == True:
            #if (subarray.pmP_pw_amplitude/subarray.pP_pw_amplitude) <= 1.72:
            f.write(str(event_name).ljust(15) + '\t')
            f.write(str(event_id).ljust(10) + '\t')
            f.write(str(np.round(subarray.outputs.beampack_backazimuth,2)).ljust(8) + '\t')
            f.write(str(np.round(subarray.outputs.ev_array_gcarc,2)).ljust(8) + '\t')
            f.write(str(np.round(subarray.crthk,2)).ljust(8) + '\t')
            f.write(str(np.round(subarray.pmP_bounce_lat,4)).ljust(8) + '\t')
            f.write(str(np.round(subarray.pmP_bounce_lon,4)).ljust(8) + '\t')
            f.write(str(np.round(subarray.pmP_pw_amplitude, 4)).ljust(8) + '\t')
            f.write(str(np.round(subarray.pP_pw_amplitude, 4)).ljust(8) + '\t') 
            f.write(str(np.round(subarray.pmP_amplitude, 4)).ljust(8) + '\t')
            f.write(str(np.round(subarray.pP_amplitude, 4)).ljust(8) + '\t')             
            f.write(str(np.round(subarray.pP_pmP_delay_time,4)).ljust(10) + '\t')
            f.write(str(np.round(subarray.new_crthk_FM,4)).ljust(12) + '\n')
    
    f.close() 
    
    file_exists = os.path.isfile(res_dir + output_file)
    
    f_all = open(res_dir + output_file, 'a+')
    if not file_exists:
        f.write('Event'.ljust(15) + '\t' + 'Event_ID'.ljust(10) + '\t' + 'Baz'.ljust(8) + '\t' + 'Gcarc'.ljust(8) + '\t' + 'Cr1_thk'.ljust(8) + '\t' + 'Bpt_Lat'.ljust(8) + '\t' + 'Bpt_Lon'.ljust(8) + '\t' + 'pmP_amp_pw'.ljust(8) + '\t' + 'pP_amp_pw'.ljust(8) + '\t' + 'pmP_amp'.ljust(8) + '\t' + 'pP_amp'.ljust(8) + '\t' + 'pP-pmP_dt'.ljust(10) + '\t' + 'New_Cr_thk'.ljust(12) + '\n')
    
    for subarray in subarray_list:
        if subarray.pmP_pick_exists == True:
            #if (subarray.pmP_pw_amplitude/subarray.pP_pw_amplitude) <= 1.72:
            f_all.write(str(event_name).ljust(15) + '\t')
            f_all.write(str(event_id).ljust(10) + '\t')
            f_all.write(str(np.round(subarray.outputs.beampack_backazimuth,2)).ljust(8) + '\t')
            f_all.write(str(np.round(subarray.outputs.ev_array_gcarc,2)).ljust(8)+ '\t')
            f_all.write(str(np.round(subarray.crthk,2)).ljust(8) + '\t')
            f_all.write(str(np.round(subarray.pmP_bounce_lat,4)).ljust(8) + '\t')
            f_all.write(str(np.round(subarray.pmP_bounce_lon,4)).ljust(8) + '\t')
            f_all.write(str(np.round(subarray.pmP_pw_amplitude, 4)).ljust(8) + '\t')
            f_all.write(str(np.round(subarray.pP_pw_amplitude, 4)).ljust(8) + '\t')
            f_all.write(str(np.round(subarray.pmP_amplitude, 4)).ljust(8) + '\t')
            f_all.write(str(np.round(subarray.pP_amplitude, 4)).ljust(8) + '\t')
            f_all.write(str(np.round(subarray.pP_pmP_delay_time,4)).ljust(10) + '\t')
            f_all.write(str(np.round(subarray.new_crthk_FM,4)).ljust(12) + '\n')
    
    f_all.close()

    
    
    
    















