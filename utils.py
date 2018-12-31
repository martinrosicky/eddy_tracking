# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 2018

@author: Martin

version: 1.0

Utility functions:
    - dist
    - vidst
    - loadVelocity
    - loadLatLon
    - dxdy
    - dvo (Divergence, Vorticity, Okubo-Weiss parameter)
    
"""

# libraries
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def dist(lat1, lon1, lat2, lon2):
    
    '''
    Approximate distance between two points by lat and lon in meter

    Input:
        lat1 (float): latitude of point 1
        lon1 (float): longitude of point 1
        lat2 (float): latitude of point 2
        lon2 (float): longitude of point 2
        
    Output:
        distance (float): euclidean distance between the two points
        
    '''
    # radius of the Earth in km
    R = 6371.0
    
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1) 
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return R * c * 1000


# vectorize dist
vdist = np.vectorize(dist)


def loadVelocity_GOM(year, time,):
    
    """
    Load daily average velocity in Gulf of Mexico
    
    Input:
        year      (int): 1993 ~ 2012
        time      (int): datetime in hours since 2000/01/01 00:00:00 
        
    Output:
        u (2d array): zonal velocity field
        v (2d array): meridional velocity field
        
    """
    
    u = np.ma.zeros((time.size, 346, 541))
    v = np.ma.zeros((time.size, 346, 541))
    
    # Load zonal velocity
    with Dataset('demo_' + str(year) + '_water_u_GOM3.nc', 'r') as u_GOM3:
        _, m3, n3 = u_GOM3['water_u'].shape
        u[:, 0:m3, 0:n3] = u_GOM3['water_u'][time, :, :]
        u[:, 0:m3, 0:n3].mask = u_GOM3['water_u'][time, :, :].mask
    
    with Dataset('demo_' + str(year) + '_water_u_GOM4.nc', 'r') as u_GOM4:
        _, m4, n4 = u_GOM4['water_u'].shape
        u[:, 0:m4, n3:n3 + n4] = u_GOM4['water_u'][time, :, :]
        u[:, 0:m4, n3:n3 + n4].mask = u_GOM4['water_u'][time, :, :].mask
    
    with Dataset('demo_' + str(year) + '_water_u_GOM1.nc', 'r') as u_GOM1:
        _, m1, n1 = u_GOM1['water_u'].shape
        u[:, m3:m3 + m1, 0:n1] = u_GOM1['water_u'][time, :, :]
        u[:, m3:m3 + m1, 0:n1].mask = u_GOM1['water_u'][time, :, :].mask
    
    with Dataset('demo_' + str(year) + '_water_u_GOM2.nc', 'r') as u_GOM2:
        _, m2, n2 = u_GOM2['water_u'].shape
        u[:, m4:m4 + m2, n1:n1 + n2] = u_GOM2['water_u'][time, :, :]
        u[:, m4:m4 + m2, n1:n1 + n2].mask = u_GOM2['water_u'][time, :, :].mask
        
    # Load meridional velocity
    with Dataset('demo_' + str(year) + '_water_v_GOM3.nc', 'r') as v_GOM3:
        _, m3, n3 = v_GOM3['water_v'].shape
        v[:, 0:m3, 0:n3] = v_GOM3['water_v'][time, :, :]
        v[:, 0:m3, 0:n3].mask = v_GOM3['water_v'][time, :, :].mask
    
    with Dataset('demo_' + str(year) + '_water_v_GOM4.nc', 'r') as v_GOM4:
        _, m4, n4 = v_GOM4['water_v'].shape
        v[:, 0:m4, n3:n3 + n4] = v_GOM4['water_v'][time, :, :]
        v[:, 0:m4, n3:n3 + n4].mask = v_GOM4['water_v'][time, :, :].mask
    
    with Dataset('demo_' + str(year) + '_water_v_GOM1.nc', 'r') as v_GOM1:
        _, m1, n1 = v_GOM1['water_v'].shape
        v[:, m3:m3 + m1, 0:n1] = v_GOM1['water_v'][time, :, :]
        v[:, m3:m3 + m1, 0:n1].mask = v_GOM1['water_v'][time, :, :].mask
    
    with Dataset('demo_' + str(year) + '_water_v_GOM2.nc', 'r') as v_GOM2:
        _, m2, n2 = v_GOM2['water_v'].shape
        v[:, m4:m4 + m2, n1:n1 + n2] = v_GOM2['water_v'][time, :, :]
        v[:, m4:m4 + m2, n1:n1 + n2].mask = v_GOM2['water_v'][time, :, :].mask
    
    # Assign NaN to outliers
    u[(abs(u) > 5) & (u.mask == False)] = np.nan
    v[(abs(v) > 5) & (v.mask == False)] = np.nan
        
    # Calculate daily average
    u = np.nanmean(u, axis=0)
    v = np.nanmean(v, axis=0)
    
    # Slice the GOM Basin
    u = u[0:325, 0:430]; v =v[0:325, 0:430]
    
    # Masking outside the region of interest (the Gulf Basin)
    u.mask[0:75, 250:] = True
    u.mask[:88, 275:] = True
    u.mask[:120, 355:] = True
    u.mask[:96, 326:350] = True
    u.mask[:105, 350:360] = True
    u.mask[250:, 410:] = True
    
    v.mask[0:75, 250:] = True
    v.mask[:88, 275:] = True
    v.mask[:120, 355:] = True
    v.mask[:96, 326:350] = True
    v.mask[:105, 350:360] = True
    v.mask[250:, 410:] = True
    
    return u, v

def loadLatLon(region):
    
    """
    Load latitude and longitude of input region
    
    Input  
        region (str): 'USW', 'GOM', 'USE', 'Hawaii'    
    
    Output 
        Y (float): latitude in degrees
        X (float): longitude in degrees
        
    """
    if region == 'GOM':
        with Dataset('gom_reanalysis.nc', 'r') as dataset:
            Y, X = dataset['lat/GOM'][0:325], dataset['lon/GOM'][0:430]    
    
    elif region == 'USW':
        with Dataset('global_reanalysis.nc', 'r') as dataset:
            Y, X = dataset['lat/USW'][:], dataset['lon/USW'][:]
            
    return Y, X


def dxdy(region):
    
    """
    Calculate dx and dy 
    
    Input:
        region (str): USW, GOM
        
    Ouput:
        dx (float): zonal distance between each grid point in m
        dy (float): meridional distance between each grid point in m

        
    """    

    if region == 'GOM':    
    
        # Load latitude and longitude
        Y, X = loadLatLon('GOM')
        
        # Calculate dx and dy
        dx = np.empty((0, 430)); dy = np.empty((325, 0))
        for j in np.arange(325):
            dx = np.vstack([dx, np.gradient(vdist(Y[j], X[0], Y[j], X[0:None]))])
        for i in np.arange(430):
            dy = np.hstack([dy, np.gradient(vdist(Y[0], X[i], Y[0:None], X[i])).reshape(325, 1)])

    elif region == 'USW':    
    
        # Load latitude and longitude
        Y, X = loadLatLon('USW')
        
        # Calculate dx and dy
        dx = np.empty((0, 153)); dy = np.empty((230, 0))
        for j in np.arange(230):
            dx = np.vstack([dx, np.gradient(vdist(Y[j], X[0], Y[j], X[0:None]))])
        for i in np.arange(153):
            dy = np.hstack([dy, np.gradient(vdist(Y[0], X[i], Y[0:None], X[i])).reshape(230, 1)])
            
            
    return dx, dy


def dvo(u, v, dx, dy):
    
    """
    Compute divergence, vorticity, and Okubo-Weiss parameter
    
    Input:
        u  (int, 2D array): zonal velocity
        v  (int, 2D array): meriodial velocity
        dx (int, 1D array): zonal distance between each grid point on Earth
        dy (int, 1D array): meridional distance between each grid point on Earth
        
    Ouput:
        div   (2D array): divergence  
        vort  (2D array): vorticity 
        ow  (2D array): Okubo-Weiss parameter
        
        
    """    
    
    # Calculate unitary gradient
    du_dy, du_dx = np.gradient(u)
    dv_dy, dv_dx = np.gradient(v)
    
    # Divide by distance between each grid point on Earth
    du_dy /= dy; du_dx /= dx; dv_dy /= dy; dv_dx /= dx
    
    div = du_dx + dv_dy    # divergence 
    vort = dv_dx - du_dy    # vorticity
    
    # Okubo-Weiss Parameter
    s_n = du_dx - dv_dy    # normal strain
    s_s = dv_dx + du_dy    # shear strain
    
    # Compute and normalize ow
    ow_raw = s_n**2 + s_s**2 - vort**2
    ow = ow_raw / ow_raw.std() 
     
    return div, vort, ow


def plotDetection(detection_result, vort, ow, u, v, Y, X):
    
    """
    Visualize detection with velocity field and vorticity dominant regions
    
    Input:
        constraint4_result (int): 1 for cyclonic eddy, -1 for anti-cyclonic eddy
                                  0 for none
        vort (float): vorticity 
        ow   (float): Okubo-Weiss parameter
        u    (float): zonal velocity field
        v    (float): meridional velocity field
            
    Output:
        no direct output. figure            
            
    """
    
    # Generate 2d meshgrid with lat and lon
    x, y = np.meshgrid(X, Y)
    
    # Label vorticity dominant region by negative and positive vorticity
    vortDominantRegion = np.ma.zeros(ow.shape)
    vortDominantRegion.mask = ow.mask
    vortDominantRegion.fill_value = -30000
    vortDominantRegion[(ow < -0.2) & (np.sign(vort) == 1)] = 1
    vortDominantRegion[(ow < -0.2) & (np.sign(vort) == -1)] = -1
    
    plt.figure(figsize=(6, 5))
    plt.pcolor(x, y, vortDominantRegion, cmap='seismic_r')
    plt.clim(-1, 1)
    plt.legend()
    stepsize = 4
    plt.scatter(X[np.where((detection_result.mask == False) & (detection_result != 0))[1]],
                Y[np.where((detection_result.mask == False) & (detection_result != 0))[0]], c='C1', marker='o', s=30)
    plt.quiver(x[::stepsize, ::stepsize], y[::stepsize, ::stepsize], u[::stepsize, ::stepsize], v[::stepsize, ::stepsize], color='k', width=0.002, scale=50, minlength=0)
    plt.rcParams.update({'font.size': 16})
    
    # Custom legend
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='eddy center', markerfacecolor='C1', markersize=9),
                       Line2D([0], [0], marker='o', color='w', label='cyclonic', markerfacecolor='navy', markersize=15),
                       Line2D([0], [0], marker='o', color='w', label='anti-cyclonic', markerfacecolor='darkred', markersize=15)]
    
    plt.legend(handles=legend_elements, loc='lower right')

    # fix axes
    plt.xlim(-98, -82)
    plt.ylim(18, 32)
    
    plt.tight_layout()
    return None
