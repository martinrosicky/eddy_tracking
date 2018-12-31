# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 2018

@author: Martin

version: 1.0

"""

#libraries
import numpy as np

# functions
def values_along_edge(array, i_center, j_center, dist):
    """
    Create a velocity vector along the edge of a square specified by center and dist
    
    Input:
        array    (2D array): a rectangular patch of matrix
        i_center (int): row index of square center
        j_center (int): column index of square center
        dist     (int): one half the width of square
    Output:
        values   (1D array): values along the edge of the square
        
    """
    if ((i_center - dist) < 0) | ((j_center - dist) < 0):
        raise IndexError
    
    values = np.array([])
    values = np.append(values, array[i_center - (dist - 1) : i_center + (dist), (j_center + (dist - 1))][::-1])
    values = np.append(values, array[i_center - (dist - 1), j_center - (dist - 1) : j_center + (dist)][:-1][::-1])
    values = np.append(values, array[i_center - (dist - 1) : i_center + (dist), j_center - (dist - 1)][1:])
    values = np.append(values, array[i_center + (dist - 1), j_center - (dist - 1) : j_center + (dist)][1:-1])
    
    return values


def detect_center(u, v, a=3, b=3):
    """
    Detect eddy centers based on 4 constraints described in Necioloi et al(2010). 
    Refer to each constraint description below for more details.
    
    Input:
        u (2D array): velocity field in zonal direction
        v (2D array): velocity field in meridional direction
        a      (int): parameter 1 (number of grid pts away the increase in magnitude are checked) 
        b      (int): parameter 2 (number of grid pts away the local minimum are checked)
    
    Output:
        constraint4_result (2D array): matrix of 0's, 1's and -1's where

                                       1 indicating cyclocnic eddy center, 
                                      -1 indicating anti-cyclonic eddy center, and
                                       0 none of the above
    
    """
             
    ###########################################################################
    #       
    # Constraint1: 
    #
    #   along an east–west (EW) section, v has to reverse in sign across
    #   the eddy center and its magnitude has to increase away from it  
    ###########################################################################
    
    # Sign change
    nrow, ncol = v.shape
    sign_change_mat1 = np.ma.zeros((nrow, ncol))
    v_sign = np.sign(v)
    for i in np.arange(nrow):
        sign_change_cy = ((np.roll(v_sign[i, :], 1) - v_sign[i, :]) == -2).astype(int)    #cyclonic or ccw: - to + as moving from W to E
        sign_change_anticy = ((np.roll(v_sign[i, :], 1) - v_sign[i, :]) == 2).astype(int)    #anti-cyclonic or cw: + to -
        sign_change_mat1[i, :] = sign_change_cy + -1 * sign_change_anticy    
    
    sign_change_mat1[:, 0] = 0
    sign_change_mat1.mask = u.mask
    sign_change_mat1.fill_value = -30000    
    
    # increase away in magnitude and no sign change left and right within a
    constraint1_result = sign_change_mat1.copy()
    v_mag = abs(v)
    for i, j in zip(*np.where((sign_change_mat1 != 0) & (sign_change_mat1.mask == False))):
        if np.any((v_mag[i, j-a : j] - np.roll(v_mag[i, j-a : j], 1))[1:] > 0) or np.any((v_mag[i, j : j+a] - np.roll(v_mag[i, j : j+a], 1))[1:] < 0) or np.any(sign_change_mat1[i, j-a : j] != 0) or np.any(sign_change_mat1[i, j+1 : j+a] != 0):
            constraint1_result[i, j] = 0
    
    # Add another point next to each sign change locations
    for i, j in zip(*np.where((constraint1_result != 0) & (constraint1_result.mask == False))):
        if constraint1_result[i, j] == 1:
            constraint1_result[i, j-1] = 1
        elif constraint1_result[i, j] == -1:
            constraint1_result[i, j-1] = -1
            
    
    ##############################################################################
    #
    # Constraint2: 
    #
    #   along a north-sourth (NW) section, u has to reverse in sign across
    #   the eddy center and its magnitude has to increase away from it  
    #
    ##############################################################################

    # Sign change
    nrow, ncol = u.shape
    sign_change_mat2 = np.ma.zeros((nrow, ncol))
    u_sign = np.sign(u)
    #for j in np.arange(ncol):
    for j in np.unique(np.where(abs(constraint1_result) == True)[1]):
        sign_change_cy = ((np.roll(u_sign[:, j], 1) - u_sign[:, j]) == 2).astype(int)    #cyclonic: + to - as moving from S to N
        sign_change_anticy = ((np.roll(u_sign[:, j], 1) - u_sign[:, j]) == -2).astype(int)    #anticyclonic: - to +
        sign_change_mat2[:, j] = sign_change_cy + -1 * sign_change_anticy 
    
    sign_change_mat1[0, :] = 0
    sign_change_mat2.mask = u.mask
    sign_change_mat2.fill_value = -30000    
    
    constraint2_result = sign_change_mat2.copy()
    # increase away in magnitude and no sign change left and right within a
    u_mag = abs(u)
    for i, j in zip(*np.where(abs(constraint2_result) == True)):
        if np.any((u_mag[i-a : i, j] - np.roll(u_mag[i-a : i, j], 1))[1:] > 0) or np.any((u_mag[i : i+a, j] - np.roll(u_mag[i : i+a, j], 1))[1:] < 0) or np.any(sign_change_mat2[i-a : i, j] != 0) or np.any(sign_change_mat2[i+1 : i+a, j] != 0):
            constraint2_result[i, j] = 0
    
    # Add another point next to each sign change locations
    for i, j in zip(*np.where((constraint2_result != 0) & (constraint2_result.mask == False))):
        if constraint2_result[i, j] == 1:
            constraint2_result[i-1, j] = 1
        elif constraint2_result[i, j] == -1:
            constraint2_result[i-1, j] = -1
    
    sign_change_mat1 = constraint1_result.copy()
    sign_change_mat2 = constraint2_result.copy()
    # Check sign change is consistent with sense of rotation
    for i, j in zip(*np.where((sign_change_mat2 == 1) | (sign_change_mat2 == -1))):
        if (sign_change_mat1[i, j] != sign_change_mat2[i, j]):
            constraint2_result[i, j] = 0
     
    ###############################################################################       
    #      
    # Constraint3: 
    #
    #   velocity magnitude has a local minimum at the eddy center  
    #
    ###############################################################################
    
    m = np.hypot(u, v)
    constraint3_result = np.ma.zeros((m.shape))
    
    # first search
    search1_result = np.ma.zeros((m.shape))
    for i, j in zip(*np.where(abs(constraint2_result) == True)): 
        i_bot, i_top = i - b, i + b
        if i_bot < 0: i_bot = 0 
        if i_top > (nrow-1): i_top = nrow-1
        j_bot, j_top = j - b, j + b
        if j_bot < 0: j_bot = 0 
        if j_top > (ncol-1): j_top = ncol-1
        box1 = m[i_bot : i_top, j_bot : j_top]
        search1_result[i_bot + np.where(box1 == box1.min())[0], j_bot + np.where(box1 == box1.min())[1]] = 1
        
    search1_result.mask = u.mask
    search1_result.fill_value = -30000
        
    # Second search
    search2_result = np.ma.zeros((m.shape))
    for i, j in zip(*np.where(search1_result == True)): 
        i_bot, i_top = i - b, i + b
        if i_bot < 0: i_bot = 0 
        if i_top > (nrow-1): i_top = nrow-1
        j_bot, j_top = j - b, j + b
        if j_bot < 0: j_bot = 0 
        if j_top > (ncol-1): j_top = ncol-1
        box1 = m[i_bot : i_top, j_bot : j_top]
        search2_result[i_bot + np.where(box1 == box1.min())[0], j_bot + np.where(box1 == box1.min())[1]] = 1
    
    search2_result.mask = m.mask
    search2_result.fill_value = -30000
    
    # Compare the two search results
    for i, j in zip(*np.where(search1_result == True)): 
        if search1_result[i, j] == search2_result[i, j]:
            constraint3_result[i, j] = 1
    
    constraint3_result.mask = u.mask
    constraint3_result.fill_value = -30000
    
    ##############################################################################
    #
    # Constraint4: 
    #
    #   around, +/-(a-1) away, the eddy center, the directions of the velocity vectors have to 
    #   change with a constant sense of rotation and the directions of two 
    #   neighboring velocity vectors have to lay within the same or two adjacent 
    #   quadrants.
    #
    ##############################################################################    
    
    # Find on what quadrant each edge vectors falls
    constraint4_result = np.ma.zeros((m.shape))
    for i, j in zip(*np.where(abs(constraint3_result) == True)):
        try:
            edge_vector_u, edge_vector_v = values_along_edge(u, i, j, a), values_along_edge(v, i, j, a)
        except IndexError:
            constraint4_result[i, j] = 0
            continue
        edge_vector_quad = np.zeros(edge_vector_u.size)
        for k in np.arange(edge_vector_u.size):
            if  (edge_vector_u[k] >= 0) & (edge_vector_v[k] >= 0): edge_vector_quad[k] = 1
            elif  (edge_vector_u[k] <= 0) & (edge_vector_v[k] >= 0): edge_vector_quad[k] = 2
            elif  (edge_vector_u[k] <= 0) & (edge_vector_v[k] <= 0): edge_vector_quad[k] = 3
            elif  (edge_vector_u[k] >= 0) & (edge_vector_v[k] <= 0): edge_vector_quad[k] = 4
    
        # Find the change in quadrant is in one direction and the edge vector passes all four quadrants
        grad = edge_vector_quad - np.roll(edge_vector_quad, 1)
        if (set(np.unique(np.delete(grad, np.where(abs(grad) == abs(grad).max())[0]))) == set([-1, 0]) or set(np.unique(np.delete(grad, np.where(abs(grad) == abs(grad).max())[0]))) == set([1, 0])) and (1 in np.unique(edge_vector_quad)) and (2 in np.unique(edge_vector_quad)) and (3 in np.unique(edge_vector_quad)) and (4 in np.unique(edge_vector_quad)): 
            constraint4_result[i, j] = 1
    constraint4_result.mask = u.mask
    constraint4_result.fill_value = -30000
    
    # Find orientation as each eddy center
    constraint4_result[np.where(abs(constraint4_result)==1)] = constraint2_result[np.where(abs(constraint4_result)==1)]
            
    return constraint4_result


def search(t, lat, lon, rot, searching_width, eddy_centers, time_idx):
    """
    Perfom search for eddy center inside a box that has center at the input 
    location and is input width wide at consecutive time step
    
    Input:
        t                 (int): day of year [0 : 365]
        i                 (int): latitude index
        j                 (int): longitude index
        rot               (int): 1 if cyclonic and -1 if anti-cyclonic
        searching_width   (int): width of square box; odd number
        eddy_centers           : 3D array containing eddy centers over time
    
    Output:
        search_time      (int): time index of search result
        closest_search_y (int): row index of search result 
        closest_search_x (int): column index of search result
    
    """
    # Eliminate search from going out of bounds
    if t < (time_idx - 1):
        # First, search for cyclic eddy in search window at "t + 1" 
        search_r = (searching_width -1) / 2
        search_y, search_x = np.where(eddy_centers[t + 1, (lat - search_r):(lat + search_r +1), (lon - search_r):(lon + search_r) +1] == rot)
        
        # Correct the search idx from window space to the original space
        search_y += (lat - search_r); search_x += (lon - search_r)
        
        if search_y.size >= 1:
            closest_idx = np.argsort(np.hypot(search_y - lat, search_x - lon))[0]
            closest_search_y = search_y[closest_idx]
            closest_search_x = search_x[closest_idx]
            search_time = t + 1
            return search_time, closest_search_y, closest_search_x  
    
        
        # Perform a second search at "t + 2" if result of the first search is null 
        elif t < (time_idx - 2):
            search_r2 = np.ceil(search_r * 1.5).astype(int)
            search_y, search_x = np.where(eddy_centers[t + 2, (lat - search_r2):(lat + search_r2 + 1), (lon - search_r2):(lon + search_r2 + 1)] == rot)
            
            # Correct the search idx from window space to the original space
            search_y += (lat - search_r2); search_x += (lon - search_r2)
        
            if search_y.size >= 1:
                closest_idx = np.argsort(np.hypot(search_y - lat, search_x - lon))[0]
                closest_search_y = search_y[closest_idx]
                closest_search_x = search_x[closest_idx]
                search_time = t + 2
                return search_time, closest_search_y, closest_search_x
            else:
                return None, None, None
        else:
            return None, None, None
    else:
        return None, None, None