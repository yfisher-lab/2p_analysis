import numpy as np
import pandas as pd
from scipy.stats import f_oneway
from scipy.stats import zscore
from scipy.stats import circmean
from scipy.stats import circvar
import seaborn as sns
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.interpolate import UnivariateSpline
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
import os
from os.path import sep
import math
from scipy.signal import correlate
from fly2p_function_TQ.imaging_2p_preprocessing import low_pass_filter_TQ, get_dff_array, normalizing_dff_array
from fly2p_function_TQ.imaging_2p_fictrac_imaging_alignment import moving_wrapped_plot_by_offset,fictrack_signal_decoding,offset_calculation 
from fly2p_function_TQ.imaging_2p_PVA_functions import calcualteBumpAmplitude, calcualteBumpAmplitude_V3, calculateBumpWidth_v1, PVA_radian_to_angle,calcualteBumpAmplitude_V4,PVAangleToRoi,strong_PVA_duration,strong_PVA_index,get_behavior_state_of_strong_PVA,get_bump_shape_at_strong_signal,get_PVA_at_strong_signal,strong_signal_index,weak_signal_index






def circmean_degrees(degree_data):
    # Convert degrees to radians
    radian_data = np.deg2rad(degree_data)
    
    # Calculate the circular mean in radians
    mean_radians = circmean(radian_data, high=np.pi, low=-np.pi)
    
    # Convert the mean back to degrees
    mean_degrees = np.rad2deg(mean_radians)
    
    return mean_degrees






# using the downsampled heading array to find the period when fly stop walking 
def find_persistence_period(head_array, degree_of_tolerance, shortest_stopFrame):
    #Convert tolerance to radian since heading array is in radians
    radian_tol = degree_of_tolerance * np.pi/180
    
    persistenceArray = np.zeros(head_array.size)
    for current_frame in range(1,len(head_array)):
        head_current = head_array[current_frame]
        
        previous_bar_positions = head_array[0:current_frame]
        #Absolute value of different from current heading
        absDiffFromCurrent = np.abs(previous_bar_positions - head_current)
        
        aboveThresholdIndex = np.argwhere(absDiffFromCurrent > radian_tol)
        
        if len(aboveThresholdIndex) == 0:
            persistenceArray[current_frame] = current_frame - 1
        else:
            mostRecentAboveThreshold = np.max(aboveThresholdIndex)
            persistenceArray[current_frame] = current_frame - mostRecentAboveThreshold 
       

    return persistenceArray





def find_stop_period_on_heading(head_velocity_array,degree_of_tolerance,shortest_stopFrame):
    radian_cutoff = degree_of_tolerance * np.pi/180
    stop_index = []
    stop_length = []
    count = 0
    
    for current_volume in range(len(head_velocity_array)):
        #Special case for last index in the array
        if current_volume == len(head_velocity_array) -1:
            if np.abs(head_velocity_array[current_volume]) < radian_cutoff:
                count = count + 1 
                if count >= shortest_stopFrame:
                    stop_index.append(current_volume)
                    stop_length.append(count)
            elif count >= shortest_stopFrame:
                stop_index.append(current_volume-1)
                stop_length.append(count)               
        elif current_volume < len(head_velocity_array) -1:
            if np.abs(head_velocity_array[current_volume]) < radian_cutoff:
                count = count + 1
            elif count < shortest_stopFrame:
                count = 0
            else:
                stop_index.append(current_volume-1)
                stop_length.append(count)
                count = 0
    
    stop_index_length_combined = np.zeros((len(stop_index),2))
    stop_index_length_combined[:,0] = stop_index
    stop_index_length_combined[:,1] = stop_length
    stop_index_length_combined = stop_index_length_combined.astype(int)
    return stop_index_length_combined



def max_raw_F_during_stopPeriod(stop_index_array,max_raw_f_array,volume_time):
    F_at_stop = []
    F_at_10s_later = []
 
    frame_10s_after_stop = int(np.ceil(10/volume_time))
    
        
    
    for current_index in range(len(stop_index_array)):
        restart_index_current = stop_index_array[current_index,0]
        stop_points_current = stop_index_array[current_index,0]-stop_index_array[current_index,1]+1
        index_of_stop = stop_index_array[current_index,0]-stop_index_array[current_index,1]+1
        
        
        if index_of_stop + frame_10s_after_stop - 1 <= stop_index_array[current_index,0]:
            index_of_10s_after_stop = index_of_stop + frame_10s_after_stop - 1 
            
            F_at_10s_later.append(np.mean(max_raw_f_array[index_of_10s_after_stop-4:index_of_10s_after_stop+1]))
            F_at_stop.append(np.mean(max_raw_f_array[stop_points_current:stop_points_current+5]))
            
    return F_at_stop,F_at_10s_later
    

def Bump_shape_during_stopPeriod(stop_index_array,Bump_shape_array,volume_time,ROI_number):
    
    if ROI_number == 8:
        bump_shape_at_stop =  np.empty((8, 0))
        bump_shape_at_stop_more_than_10s_trial =  np.empty((8, 0))
        bump_shape_at_5s_after_stop =  np.empty((8, 0))
        bump_shape_at_10s_after_stop =  np.empty((8, 0))
        bump_shape_at_20s_after_stop =  np.empty((8, 0))
    else:
        bump_shape_at_stop =  np.empty((16, 0))
        bump_shape_at_stop_after_more_than_10s_tria =  np.empty((16, 0))
        bump_shape_at_5s_after_stop =  np.empty((16, 0))
        bump_shape_at_10s_after_stop =  np.empty((16, 0))
        bump_shape_at_20s_after_stop =  np.empty((16, 0))
    
    
    stop_points_average_5_frame_after = []
    stop_after_5s_average_5_frame_before = []
    stop_after_10s_average_5_frame_before = []
    stop_after_20s_average_5_frame_before = []
    frame_5s_after_stop = int(np.ceil(5/volume_time))
    frame_10s_after_stop = int(np.ceil(10/volume_time))
    frame_20s_after_stop = int(np.ceil(20/volume_time))
    
    
    for current_index in range(len(stop_index_array)):
        restart_index_current = stop_index_array[current_index,0]
        stop_points_current = stop_index_array[current_index,0]-stop_index_array[current_index,1]+1
        index_of_stop = stop_index_array[current_index,0]-stop_index_array[current_index,1]+1
        
        bump_shape_at_stop_current = np.mean(Bump_shape_array[:,stop_points_current:stop_points_current+5], axis = 1)
        
        bump_shape_at_stop = np.hstack((bump_shape_at_stop, bump_shape_at_stop_current.reshape(-1, 1)))
        
        if index_of_stop + frame_5s_after_stop - 1 <= stop_index_array[current_index,0]:
            index_of_5s_after_stop = index_of_stop + frame_5s_after_stop - 1 
            bump_shape_at_5s_after_stop_current = np.mean(Bump_shape_array[:,index_of_5s_after_stop-4:index_of_5s_after_stop+1], axis = 1)
            bump_shape_at_5s_after_stop  =  np.hstack((bump_shape_at_5s_after_stop, bump_shape_at_5s_after_stop_current.reshape(-1, 1)))
        
        if index_of_stop + frame_10s_after_stop - 1 <= stop_index_array[current_index,0]:
            index_of_10s_after_stop = index_of_stop + frame_10s_after_stop - 1 
            bump_shape_at_10s_after_stop_current = np.mean(Bump_shape_array[:,index_of_10s_after_stop-4:index_of_10s_after_stop+1], axis = 1)
            bump_shape_at_10s_after_stop  =  np.hstack((bump_shape_at_10s_after_stop, bump_shape_at_10s_after_stop_current.reshape(-1, 1)))
            bump_shape_at_stop_more_than_10s_trial = np.hstack((bump_shape_at_stop_more_than_10s_trial, bump_shape_at_stop_current .reshape(-1, 1)))
        
    
        if index_of_stop + frame_20s_after_stop - 1 <= stop_index_array[current_index,0]:
            index_of_20s_after_stop = index_of_stop + frame_20s_after_stop - 1 
            bump_shape_at_20s_after_stop_current = np.mean(Bump_shape_array[:,index_of_20s_after_stop-4:index_of_20s_after_stop+1], axis = 1)
            bump_shape_at_20s_after_stop  =  np.hstack((bump_shape_at_20s_after_stop, bump_shape_at_20s_after_stop_current.reshape(-1, 1)))
    if bump_shape_at_stop.shape[1] > 0:
        bump_shape_at_stop = np.mean(bump_shape_at_stop,axis=1)
    if bump_shape_at_5s_after_stop.shape[1] > 0:
        bump_shape_at_5s_after_stop = np.mean(bump_shape_at_5s_after_stop,axis=1)
    if bump_shape_at_10s_after_stop.shape[1] > 0:
        bump_shape_at_10s_after_stop = np.mean(bump_shape_at_10s_after_stop,axis=1)
    if bump_shape_at_stop_more_than_10s_trial.shape[1] > 0:
        bump_shape_at_stop_more_than_10s_trial = np.mean(bump_shape_at_stop_more_than_10s_trial,axis=1)  
    if bump_shape_at_20s_after_stop.shape[1] > 0:
        bump_shape_at_20s_after_stop = np.mean(bump_shape_at_20s_after_stop,axis=1)
     
    return bump_shape_at_stop, bump_shape_at_5s_after_stop,bump_shape_at_10s_after_stop,bump_shape_at_20s_after_stop,bump_shape_at_stop_more_than_10s_trial 
        

    
    
    
def PVA_velocity_right_before_stopPeriod(PVA_array, volume_time, stop_index_array, duration):
    """
    Calculates PVA angular velocity in 1-second bins before each stop period.

    Parameters:
    - PVA_array (numpy array): The array of PVA angles per frame.
    - volume_time (float): Frame rate (time per frame).
    - stop_index_array (numpy array): An Nx2 array where:
        - stop_index_array[:,0] is the index of stop start (restart index).
        - stop_index_array[:,1] is the duration of the stop period in frames.
    - duration (int): Time window (in seconds) before each stop period to analyze.

    Returns:    
    - angular_velocity_per_sec_before_stop: List of NumPy arrays containing 
      angular velocity (deg/s) in 1s bins before each stop period.
    """

    # Convert chosen duration to frame count
    frames_before_stop = int(np.ceil(duration / volume_time))

    angular_velocity_per_sec_before_stop = []  # Final list of velocity arrays

    for i in range(len(stop_index_array)):
        stop_start_index = stop_index_array[i, 0]  # Start of stop

        # Ensure there is enough data before the stop period
        if stop_start_index - frames_before_stop < 0:
            angular_velocity_per_sec_before_stop.append(np.full((1,), np.nan))  # Store NaN array if not enough data
            continue

        # Extract PVA values in the selected time window
        PVA_before_stop = PVA_array[(stop_start_index - frames_before_stop):stop_start_index]

        # Compute angular velocity per frame
        angle_differences = np.diff(PVA_before_stop)

        # Correct for circular values (handling wrap-around at 180/-180)
        angle_differences = np.where(np.abs(angle_differences) > 180,
                                     angle_differences - np.sign(angle_differences) * 360,
                                     angle_differences)

        # Convert frame-wise differences to per-second angular velocity
        angular_velocity_per_frame = angle_differences / volume_time  # Convert to deg/s

        # Bin into 1s chunks
        frames_per_second = int(1 / volume_time)
        valid_length = (len(angular_velocity_per_frame) // frames_per_second) * frames_per_second

        if valid_length > 0:
            reshaped_velocity = angular_velocity_per_frame[:valid_length].reshape(-1, frames_per_second)
            avg_velocity_per_second = np.mean(reshaped_velocity, axis=1)
        else:
            avg_velocity_per_second = np.full((1,), np.nan)  # Store NaN if no valid bins

        # Store the results (consistent with stop period function output)
        angular_velocity_per_sec_before_stop.append(avg_velocity_per_second)

    return angular_velocity_per_sec_before_stop   

def PVA_velocity_right_before_stopPeriod_with_strength_threshold(
    PVA_array,
    PVA_strength_array,
    volume_time,
    stop_index_array,
    duration,
    strength_threshold
):
    """
    Calculates PVA angular velocity in 1-second bins before each stop period,
    excluding weak-strength frames, using enhanced NaN handling logic.

    Parameters:
    - PVA_array (np.array): Array of PVA angles.
    - PVA_strength_array (np.array): Array of PVA strengths corresponding to the angles.
    - volume_time (float): Time per frame.
    - stop_index_array (np.array): Nx2 array where:
        - stop_index_array[:,0] is stop start index.
        - stop_index_array[:,1] is stop duration.
    - duration (int): Seconds before stop to include in calculation.
    - strength_threshold (float): Minimum PVA strength to include a frame.

    Returns:
    - angular_velocity_per_sec_before_stop: List of arrays (one per stop chunk)
      with average angular velocity per 1s bin (deg/s).
    """
    frames_before_stop = int(np.ceil(duration / volume_time))
    angular_velocity_per_sec_before_stop = []
    frames_per_second = int(1 / volume_time)

    for i in range(len(stop_index_array)):
        stop_start_index = stop_index_array[i, 0]

        if stop_start_index - frames_before_stop < 1:
            angular_velocity_per_sec_before_stop.append(np.full((1,), np.nan))
            continue

        PVA_segment = PVA_array[(stop_start_index - frames_before_stop):stop_start_index].copy()
        strength_segment = PVA_strength_array[(stop_start_index - frames_before_stop):stop_start_index].copy()

        PVA_segment[strength_segment < strength_threshold] = np.nan

        angle_drift_framewise_non_abs = []

        for j in range(len(PVA_segment) - 1):
            a = PVA_segment[j]
            b = PVA_segment[j + 1]

            if np.isnan(a):
                if j > 0:
                    a_prev = PVA_segment[j - 1]
                    if not np.isnan(a_prev) and not np.isnan(b):
                        diff = b - a_prev
                        if diff > 180:
                            diff -= 360
                        elif diff < -180:
                            diff += 360
                        angle_drift_framewise_non_abs.append(diff / volume_time)
                        continue
                angle_drift_framewise_non_abs.append(np.nan)
            elif np.isnan(b):
                angle_drift_framewise_non_abs.append(np.nan)
            else:
                diff = b - a
                if diff > 180:
                    diff -= 360
                elif diff < -180:
                    diff += 360
                angle_drift_framewise_non_abs.append(diff / volume_time)

        angle_drift_framewise_non_abs = np.array(angle_drift_framewise_non_abs)
        usable_len = (len(angle_drift_framewise_non_abs) // frames_per_second) * frames_per_second

        if usable_len > 0:
            reshaped = angle_drift_framewise_non_abs[:usable_len].reshape(-1, frames_per_second)
            avg_per_sec = np.nanmean(reshaped, axis=1)
        else:
            avg_per_sec = np.full((1,), np.nan)

        angular_velocity_per_sec_before_stop.append(avg_per_sec)

    return angular_velocity_per_sec_before_stop
def behavior_velocity_right_before_stopPeriod(velocity_array, volume_time, stop_index_array, duration):
    """
    Calculates behavior angular velocity in 1-second bins before each stop period.

    Parameters:
    - velocity_array (numpy array): Angular velocity (deg/s) per frame.
    - volume_time (float): Time per frame (in seconds).
    - stop_index_array (numpy array): An Nx2 array where:
        - stop_index_array[:,0] is the index of stop start (restart index).
        - stop_index_array[:,1] is the duration of the stop period in frames.
    - duration (int): Time window (in seconds) before each stop period to analyze.

    Returns:    
    - velocity_per_sec_before_stop: List of NumPy arrays containing 
      average angular velocity (deg/s) in 1s bins before each stop.
    """

    frames_before_stop = int(np.ceil(duration / volume_time))
    velocity_per_sec_before_stop = []

    for i in range(len(stop_index_array)):
        stop_start_index = stop_index_array[i, 0]

        # Check if enough frames exist before the stop
        if stop_start_index - frames_before_stop < 0:
            velocity_per_sec_before_stop.append(np.full((1,), np.nan))
            continue

        # Slice the velocity data directly
        velocity_before_stop = velocity_array[(stop_start_index - frames_before_stop):stop_start_index]

        # Bin into 1s intervals
        frames_per_second = int(1 / volume_time)
        valid_length = (len(velocity_before_stop) // frames_per_second) * frames_per_second

        if valid_length > 0:
            reshaped = velocity_before_stop[:valid_length].reshape(-1, frames_per_second)
            avg_velocity_per_second = np.mean(reshaped, axis=1)
        else:
            avg_velocity_per_second = np.full((1,), np.nan)

        velocity_per_sec_before_stop.append(avg_velocity_per_second)

    return velocity_per_sec_before_stop
 
def PVA_position_right_before_stopPeriod(PVA_array, volume_time, stop_index_array, duration):
    """
    Calculates average PVA position (in radians) in 1-second bins before each stop period.

    Parameters:
    - PVA_array (numpy array): Circular PVA position data (radians from -π to π).
    - volume_time (float): Time per frame (in seconds).
    - stop_index_array (numpy array): An Nx2 array where:
        - stop_index_array[:,0] is the index of stop start (restart index).
        - stop_index_array[:,1] is the duration of the stop period in frames.
    - duration (int): Time window (in seconds) before each stop period to analyze.

    Returns:    
    - pva_per_sec_before_stop: List of NumPy arrays containing 
      average PVA position (radians) in 1s bins before each stop.
    """

    frames_before_stop = int(np.ceil(duration / volume_time))
    pva_per_sec_before_stop = []

    for i in range(len(stop_index_array)):
        stop_start_index = stop_index_array[i, 0]

        if stop_start_index - frames_before_stop < 0:
            pva_per_sec_before_stop.append(np.full((1,), np.nan))
            continue

        PVA_before_stop = PVA_array[(stop_start_index - frames_before_stop):stop_start_index]

        frames_per_second = int(1 / volume_time)
        valid_length = (len(PVA_before_stop) // frames_per_second) * frames_per_second

        if valid_length > 0:
            reshaped = PVA_before_stop[:valid_length].reshape(-1, frames_per_second)
            # Use circular mean for each 1-second bin
            circ_avg_per_second = np.array([
                circmean(bin, high=180, low=-180) for bin in reshaped
            ])
        else:
            circ_avg_per_second = np.full((1,), np.nan)

        pva_per_sec_before_stop.append(circ_avg_per_second)

    return pva_per_sec_before_stop



def second_wise_PVA_position_and_change_rate_at_stop_strength_threshold(
    stop_index_array,
    PVA_array,
    strength_array,
    volume_time,
    strength_threshold
):
    PVA_angle_drift_per_sec_2_second_wise_threshold = []
    PVA_angle_drift_per_sec_2_second_wise_threshold_absolute = []
    PVA_position_second_wise_threshold = []

    time_unit_per_cell = 1
    data_per_second = int(time_unit_per_cell / volume_time)

    for current_index in range(len(stop_index_array)):
        restart_index_current = stop_index_array[current_index, 0]
        stop_points_current = stop_index_array[current_index, 0] - stop_index_array[current_index, 1] + 1

        PVA_segment = PVA_array[stop_points_current:restart_index_current + 1].copy()
        strength_segment = strength_array[stop_points_current:restart_index_current + 1].copy()
        PVA_segment[strength_segment < strength_threshold] = np.nan

        angle_drift_framewise_abs = []
        angle_drift_framewise_non_abs = []

        for i in range(len(PVA_segment) - 1):
            a = PVA_segment[i]
            b = PVA_segment[i + 1]
            if np.isnan(a):
                if i > 0:
                    a_prev = PVA_segment[i - 1]
                    if not np.isnan(a_prev) and not np.isnan(b):
                        diff = b - a_prev
                        if diff > 180:
                            diff -= 360
                        elif diff < -180:
                            diff += 360
                        angle_drift_framewise_non_abs.append(diff / volume_time)
                        angle_drift_framewise_abs.append(np.abs(diff) / volume_time)
                        continue
                angle_drift_framewise_non_abs.append(np.nan)
                angle_drift_framewise_abs.append(np.nan)
            elif np.isnan(b):
                angle_drift_framewise_non_abs.append(np.nan)
                angle_drift_framewise_abs.append(np.nan)
            else:
                diff = b - a
                if diff > 180:
                    diff -= 360
                elif diff < -180:
                    diff += 360
                angle_drift_framewise_non_abs.append(diff / volume_time)
                angle_drift_framewise_abs.append(np.abs(diff) / volume_time)

        angle_drift_framewise_non_abs = np.array(angle_drift_framewise_non_abs)
        angle_drift_framewise_abs = np.array(angle_drift_framewise_abs)

        usable_length = (len(angle_drift_framewise_non_abs) // data_per_second) * data_per_second
        data_non_abs = angle_drift_framewise_non_abs[:usable_length].reshape(-1, data_per_second)
        data_abs = angle_drift_framewise_abs[:usable_length].reshape(-1, data_per_second)

        average_per_second_non_abs = np.nanmean(data_non_abs, axis=1)
        average_per_second_abs = np.nanmean(data_abs, axis=1)

        PVA_angle_drift_per_sec_2_second_wise_threshold.append(average_per_second_non_abs)
        PVA_angle_drift_per_sec_2_second_wise_threshold_absolute.append(average_per_second_abs)

        usable_len_signal = (len(PVA_segment) // data_per_second) * data_per_second
        PVA_segment_binned = PVA_segment[:usable_len_signal].reshape(-1, data_per_second)
        circ_means = np.array([
            circmean(row[~np.isnan(row)], high=180, low=-180) if np.any(~np.isnan(row)) else np.nan
            for row in PVA_segment_binned
        ])
        PVA_position_second_wise_threshold.append(circ_means)

    return (
        PVA_angle_drift_per_sec_2_second_wise_threshold,
        PVA_position_second_wise_threshold,
        PVA_angle_drift_per_sec_2_second_wise_threshold_absolute
    )

    
def PVA_during_stopPeriod(stop_index_array,PVA_array, volume_time):
    #Initiate output dataframe and 
    df_PVA_stop = pd.DataFrame()
    stop_points = []
    stop_points_average_5_frame_after = []
    restart_points = []
    restart_points_100ms_before = []
    restart_points_500ms_before = []
    restart_points_2s_before = []
    stop_after_3s = []
    stop_after_5s = []
    stop_after_10s = []
    stop_after_20s = []
    stop_after_10s_average_5_frame_before = []
    stop_after_20s_average_5_frame_before = []
    stop_after_35s = []
    stop_after_60s = []
    middle_points = []
    PVA_angle_drift_per_sec = []
    PVA_angle_drift_per_sec_second_wise = []
    PVA_angle_drift_per_sec_2 = []
    PVA_angle_drift_per_sec_2_second_wise = []
    PVA_angle_drift_per_sec_2_second_wise_absolute = []
    PVA_position_second_wise = []
    frame_100ms_before = int(np.ceil(0.1/volume_time))
    frame_500ms_before = int(np.ceil(0.5/volume_time))
    frame_2s_before = int(np.ceil(2/volume_time))
    frame_3s_after_stop = int(np.ceil(3/volume_time))
    frame_5s_after_stop = int(np.ceil(5/volume_time))
    frame_10s_after_stop = int(np.ceil(10/volume_time))
    frame_20s_after_stop = int(np.ceil(20/volume_time))
    frame_35s_after_stop = int(np.ceil(35/volume_time))
    frame_60s_after_stop = int(np.ceil(60/volume_time))
    duration_stop = []
    for current_index in range(len(stop_index_array)):
        restart_index_current = stop_index_array[current_index,0]
        stop_points_current = stop_index_array[current_index,0]-stop_index_array[current_index,1]+1
     
        duration_stop.append(stop_index_array[current_index,1]*volume_time)
        restart_points.append(PVA_array[restart_index_current])
        stop_points.append(PVA_array[stop_points_current])
        stop_points_average_5_frame_after.append(circmean_degrees(PVA_array[stop_points_current:stop_points_current+5]))
        middle_points.append(PVA_array[stop_index_array[current_index,0]-int(np.floor(stop_index_array[current_index,1]/2))])
        index_of_stop = stop_index_array[current_index,0]-stop_index_array[current_index,1]+1
        restart_points_100ms_before.append(PVA_array[stop_index_array[current_index,0]-frame_100ms_before])
        restart_points_500ms_before.append(PVA_array[stop_index_array[current_index,0]-frame_500ms_before+1])
        restart_points_2s_before.append(PVA_array[stop_index_array[current_index,0]-frame_2s_before+1])
        if index_of_stop + frame_3s_after_stop - 1 <= stop_index_array[current_index,0]:
            stop_after_3s.append(PVA_array[index_of_stop + frame_3s_after_stop - 1])
        else:
            stop_after_3s.append(np.NaN)
        if index_of_stop + frame_5s_after_stop - 1 <= stop_index_array[current_index,0]:
            stop_after_5s.append(PVA_array[index_of_stop + frame_5s_after_stop - 1])
        else:
            stop_after_5s.append(np.NaN)
        if index_of_stop + frame_10s_after_stop - 1 <= stop_index_array[current_index,0]:
            stop_after_10s.append(PVA_array[index_of_stop + frame_10s_after_stop - 1])
            stop_after_10s_average_5_frame_before.append(circmean_degrees(PVA_array[(index_of_stop + frame_10s_after_stop-4):(index_of_stop + frame_10s_after_stop)]))
        else:
            stop_after_10s.append(np.NaN)
            stop_after_10s_average_5_frame_before.append(np.NaN)
        if index_of_stop + frame_20s_after_stop - 1 <= stop_index_array[current_index,0]:
            stop_after_20s.append(PVA_array[index_of_stop + frame_20s_after_stop - 1])
            stop_after_20s_average_5_frame_before.append(circmean_degrees(PVA_array[(index_of_stop + frame_20s_after_stop-4):(index_of_stop + frame_20s_after_stop)]))
        else:
            stop_after_20s.append(np.NaN)
            stop_after_20s_average_5_frame_before.append(np.NaN)
        if index_of_stop + frame_35s_after_stop - 1 <= stop_index_array[current_index,0]:
            stop_after_35s.append(PVA_array[index_of_stop + frame_35s_after_stop - 1])
        else:
            stop_after_35s.append(np.NaN)
        if index_of_stop + frame_60s_after_stop - 1 <= stop_index_array[current_index,0]:
            stop_after_60s.append(PVA_array[index_of_stop + frame_60s_after_stop - 1])
        else:
            stop_after_60s.append(np.NaN)
        
        
        angle_drift_current_frame_wise = []
        angle_drift_current_frame_wise_non_absolute = []
        
        #time unit for cell of calculation
        time_unit_per_cell = 1
        #Calculate PVA angle drift frame by frame and get angle drift/s (This is actucal dwell location difference
        for current_PVA_index in range(restart_index_current-stop_points_current):
            current_difference = PVA_array[stop_points_current+current_PVA_index+1] -PVA_array[stop_points_current]
            if np.abs(current_difference) > 180:
                if current_difference  < 0:
                    current_difference= current_difference  + 360
                else:
                    current_difference = current_difference - 360
            angle_drift_current_frame_wise.append(np.abs(current_difference))
            angle_drift_current_frame_wise_non_absolute.append(current_difference)
        angle_drift_current_frame_wise = np.array(angle_drift_current_frame_wise)
        angle_drift_current_frame_wise_non_absolute = np.array(angle_drift_current_frame_wise_non_absolute)
        # Calculate the number of data points per second
        data_per_second = int(time_unit_per_cell / volume_time)  # 1 second / 0.2 seconds per data point

        # Reshape the data array into a 2D array with each row representing one second
        data_reshaped = angle_drift_current_frame_wise[:len(angle_drift_current_frame_wise) // data_per_second * data_per_second].reshape(-1, data_per_second)
        data_reshaped_non_absolute = angle_drift_current_frame_wise_non_absolute[:len(angle_drift_current_frame_wise_non_absolute) // data_per_second* data_per_second].reshape(-1, data_per_second)

        #Calculate the average value per second
        average_per_second = np.mean(data_reshaped, axis=1)
        average_per_second_non_absolute = np.mean(data_reshaped_non_absolute, axis=1)

        # Calculate the single averaged value for per second
        PVA_angle_drift_per_sec_current = np.mean(average_per_second)
        PVA_angle_drift_per_sec.append(PVA_angle_drift_per_sec_current)
        # Calculate the data that only averaged to one second window 
        PVA_angle_drift_per_sec_second_wise.append(average_per_second_non_absolute)
        
        
        # ADDED: compute circular average of PVA positions per second
        pva_segment = PVA_array[stop_points_current:restart_index_current + 1]
        if len(pva_segment) >= data_per_second:
            pva_reshaped = pva_segment[:len(pva_segment) // data_per_second * data_per_second].reshape(-1, data_per_second)
            circular_means = np.array([circmean(row, high=180, low=-180) for row in pva_reshaped])
        else:
            circular_means = np.full((1,), np.nan)
        PVA_position_second_wise.append(circular_means)
        
        
        
        angle_drift_current_frame_wise_2= []
        angle_drift_current_frame_wise_2_non_absolute = []
        #Calculate PVA angle drift frame by frame and get angle drift/s
        for current_PVA_index in range(restart_index_current-stop_points_current):
            current_difference = PVA_array[stop_points_current+current_PVA_index+1] -PVA_array[stop_points_current+current_PVA_index]
            if np.abs(current_difference) > 180:
                if current_difference  < 0:
                    current_difference= current_difference  + 360
                else:
                    current_difference = current_difference - 360
            angle_drift_current_frame_wise_2.append(np.abs(current_difference))
            angle_drift_current_frame_wise_2_non_absolute.append(current_difference)
        angle_drift_current_frame_wise_2 = np.array(angle_drift_current_frame_wise_2)
        angle_drift_current_frame_wise_2_non_absolute = np.array(angle_drift_current_frame_wise_2_non_absolute/volume_time)
        # Calculate the number of data points per second
        data_per_second_2 = int(time_unit_per_cell / volume_time)  # 1 second / 0.2 seconds per data point

        # Reshape the data array into a 2D array with each row representing one second
        data_reshaped_2 = angle_drift_current_frame_wise_2[:len(angle_drift_current_frame_wise_2) // data_per_second_2 * data_per_second_2].reshape(-1, data_per_second_2)
        data_reshaped_2_non_absolute = angle_drift_current_frame_wise_2_non_absolute[:len(angle_drift_current_frame_wise_2_non_absolute) // data_per_second_2 * data_per_second_2].reshape(-1, data_per_second_2)

        #Calculate the average value per second
        average_per_second_2 = np.mean(data_reshaped_2, axis=1)
        average_per_second_2_non_absolute = np.mean(data_reshaped_2_non_absolute, axis=1)

        # Calculate the single averaged value for one trial
        PVA_angle_drift_per_sec_current_2 = np.mean(average_per_second_2)
        PVA_angle_drift_per_sec_2.append(PVA_angle_drift_per_sec_current_2)
        # Calculate the data that only averaged to one second window 
        PVA_angle_drift_per_sec_2_second_wise.append(average_per_second_2_non_absolute)
        PVA_angle_drift_per_sec_2_second_wise_absolute.append(average_per_second_2)
        
        
    
    df_PVA_stop['Period_duration'] = duration_stop
    df_PVA_stop['PVA_before_stop'] = stop_points
    df_PVA_stop['PVA_before_stop_average_5_frames_after'] = stop_points_average_5_frame_after
    df_PVA_stop['PVA_at_restart'] = restart_points
    df_PVA_stop['PVA_100ms_before_restart'] = restart_points_100ms_before
    df_PVA_stop['PVA_500ms_before_restart'] = restart_points_500ms_before
    df_PVA_stop['PVA_in_middle'] = middle_points
    df_PVA_stop['PVA_2s_before_restart'] = restart_points_2s_before
    df_PVA_stop['PVA_3s_after_stop'] = stop_after_3s
    df_PVA_stop['PVA_5s_after_stop'] = stop_after_5s
    df_PVA_stop['PVA_10s_after_stop'] = stop_after_10s
    df_PVA_stop['PVA_10s_after_stop_average_5_frames_before'] = stop_after_10s_average_5_frame_before
    df_PVA_stop['PVA_20s_after_stop'] = stop_after_20s
    df_PVA_stop['PVA_20s_after_stop_average_5_frames_before'] = stop_after_20s_average_5_frame_before
    df_PVA_stop['PVA_35s_after_stop'] = stop_after_35s
    df_PVA_stop['PVA_60s_after_stop'] = stop_after_60s
    df_PVA_stop['PVA_angle_drift_per_second']= PVA_angle_drift_per_sec
    df_PVA_stop['PVA_angle_drift_per_second_2']= PVA_angle_drift_per_sec_2
    return df_PVA_stop, PVA_angle_drift_per_sec_2_second_wise,  PVA_angle_drift_per_sec_second_wise,PVA_angle_drift_per_sec_2_second_wise_absolute,PVA_position_second_wise





def forwrad_speed_during_stopPeriod(stop_index_array,forward_speed_array, volume_time):
    #Initiate output dataframe and 
    df_forward_speed_array_stop = pd.DataFrame()
    stop_points = []
    restart_points = []
    restart_points_100ms_before = []
    restart_points_500ms_before = []
    restart_points_2s_before = []
    stop_after_3s = []
    stop_after_5s = []
    stop_after_10s = []
    stop_after_20s = []
    stop_after_35s = []
    middle_points = []
    frame_100ms_before = int(np.ceil(0.1/volume_time))
    frame_500ms_before = int(np.ceil(0.5/volume_time))
    frame_2s_before = int(np.ceil(2/volume_time))
    frame_3s_after_stop = int(np.ceil(3/volume_time))
    frame_5s_after_stop = int(np.ceil(5/volume_time))
    frame_10s_after_stop = int(np.ceil(10/volume_time))
    frame_20s_after_stop = int(np.ceil(20/volume_time))
    frame_35s_after_stop = int(np.ceil(35/volume_time))
    
    for current_index in range(len(stop_index_array)):
        restart_points.append(forward_speed_array[stop_index_array[current_index,0]])
        stop_points.append(forward_speed_array[stop_index_array[current_index,0]-stop_index_array[current_index,1]+1])
        middle_points.append(forward_speed_array[stop_index_array[current_index,0]-int(np.floor(stop_index_array[current_index,1]/2))])
        index_of_stop = stop_index_array[current_index,0]-stop_index_array[current_index,1]+1
        restart_points_100ms_before.append(forward_speed_array[stop_index_array[current_index,0]-frame_100ms_before])
        restart_points_500ms_before.append(forward_speed_array[stop_index_array[current_index,0]-frame_500ms_before+1])
        restart_points_2s_before.append(forward_speed_array[stop_index_array[current_index,0]-frame_2s_before+1])
        if index_of_stop + frame_3s_after_stop - 1 <= stop_index_array[current_index,0]:
            stop_after_3s.append(forward_speed_array[index_of_stop + frame_3s_after_stop - 1])
        else:
            stop_after_3s.append(np.NaN)
        if index_of_stop + frame_5s_after_stop - 1 <= stop_index_array[current_index,0]:
            stop_after_5s.append(forward_speed_array[index_of_stop + frame_5s_after_stop - 1])
        else:
            stop_after_5s.append(np.NaN)
        if index_of_stop + frame_10s_after_stop - 1 <= stop_index_array[current_index,0]:
            stop_after_10s.append(forward_speed_array[index_of_stop + frame_10s_after_stop - 1])
        else:
            stop_after_10s.append(np.NaN)
        if index_of_stop + frame_20s_after_stop - 1 <= stop_index_array[current_index,0]:
            stop_after_20s.append(forward_speed_array[index_of_stop + frame_20s_after_stop - 1])
        else:
            stop_after_20s.append(np.NaN)
        if index_of_stop + frame_35s_after_stop - 1 <= stop_index_array[current_index,0]:
            stop_after_35s.append(forward_speed_array[index_of_stop + frame_35s_after_stop - 1])
        else:
            stop_after_35s.append(np.NaN)
    
    
    df_forward_speed_array_stop ['forward_speed_before_stop'] = stop_points
    df_forward_speed_array_stop ['forward_speed_at_restart'] = restart_points
    df_forward_speed_array_stop ['forward_speed_100ms_before_restart'] = restart_points_100ms_before
    df_forward_speed_array_stop ['forward_speed_500ms_before_restart'] = restart_points_500ms_before
    df_forward_speed_array_stop ['forward_speed_in_middle'] = middle_points
    df_forward_speed_array_stop ['forward_speed_2s_before_restart'] = restart_points_2s_before
    df_forward_speed_array_stop ['forward_speed_3s_after_stop'] = stop_after_3s
    df_forward_speed_array_stop ['forward_speed_5s_after_stop'] = stop_after_5s
    df_forward_speed_array_stop ['forward_speed_10s_after_stop'] = stop_after_10s
    df_forward_speed_array_stop ['forward_speed_20s_after_stop'] = stop_after_20s
    df_forward_speed_array_stop ['forward_speed_35s_after_stop'] = stop_after_35s
    
    return df_forward_speed_array_stop 




def angular_speed_during_stopPeriod(stop_index_array,angular_speed_array, angular_velocity_array,volume_time):
    #Initiate output dataframe and 
    df_angular_speed_array_stop = pd.DataFrame()
    stop_points = []
    restart_points = []
    restart_points_100ms_before = []
    restart_points_500ms_before = []
    restart_points_2s_before = []
    stop_after_3s = []
    stop_after_5s = []
    stop_after_10s = []
    stop_after_20s = []
    stop_after_35s = []
    middle_points = []
    frame_100ms_before = int(np.ceil(0.1/volume_time))
    frame_500ms_before = int(np.ceil(0.5/volume_time))
    frame_2s_before = int(np.ceil(2/volume_time))
    frame_3s_after_stop = int(np.ceil(3/volume_time))
    frame_5s_after_stop = int(np.ceil(5/volume_time))
    frame_10s_after_stop = int(np.ceil(10/volume_time))
    frame_20s_after_stop = int(np.ceil(20/volume_time))
    frame_35s_after_stop = int(np.ceil(35/volume_time))
    angular_velocity_binned = []
    
    for current_index in range(len(stop_index_array)):
        restart_idx = stop_index_array[current_index, 0]
        stop_duration = stop_index_array[current_index, 1]
        stop_start_idx = restart_idx - stop_duration + 1

        restart_points.append(angular_speed_array[restart_idx])
        stop_points.append(angular_speed_array[stop_start_idx])
        middle_idx = restart_idx - int(np.floor(stop_duration / 2))
        middle_points.append(angular_speed_array[middle_idx])
        restart_points_100ms_before.append(angular_speed_array[restart_idx - frame_100ms_before])
        restart_points_500ms_before.append(angular_speed_array[restart_idx - frame_500ms_before + 1])
        restart_points_2s_before.append(angular_speed_array[restart_idx - frame_2s_before + 1])

        if stop_start_idx + frame_3s_after_stop - 1 <= restart_idx:
            stop_after_3s.append(angular_speed_array[stop_start_idx + frame_3s_after_stop - 1])
        else:
            stop_after_3s.append(np.NaN)
        if stop_start_idx + frame_5s_after_stop - 1 <= restart_idx:
            stop_after_5s.append(angular_speed_array[stop_start_idx + frame_5s_after_stop - 1])
        else:
            stop_after_5s.append(np.NaN)
        if stop_start_idx + frame_10s_after_stop - 1 <= restart_idx:
            stop_after_10s.append(angular_speed_array[stop_start_idx + frame_10s_after_stop - 1])
        else:
            stop_after_10s.append(np.NaN)
        if stop_start_idx + frame_20s_after_stop - 1 <= restart_idx:
            stop_after_20s.append(angular_speed_array[stop_start_idx + frame_20s_after_stop - 1])
        else:
            stop_after_20s.append(np.NaN)
        if stop_start_idx + frame_35s_after_stop - 1 <= restart_idx:
            stop_after_35s.append(angular_speed_array[stop_start_idx + frame_35s_after_stop - 1])
        else:
            stop_after_35s.append(np.NaN)

        # Get second-wise binned angular velocity during stop
        velocity_segment = angular_velocity_array[stop_start_idx:restart_idx]
        frames_per_second = int(1 / volume_time)
        valid_len = (len(velocity_segment) // frames_per_second) * frames_per_second

        if valid_len > 0:
            reshaped_velocity = velocity_segment[:valid_len].reshape(-1, frames_per_second)
            avg_velocity_bins = np.mean(reshaped_velocity, axis=1)
        else:
            avg_velocity_bins = np.full((1,), np.nan)

        angular_velocity_binned.append(avg_velocity_bins)
    
    
    df_angular_speed_array_stop['angular_speed_before_stop'] = stop_points
    df_angular_speed_array_stop['angular_speed_at_restart'] = restart_points
    df_angular_speed_array_stop['angular_speed_100ms_before_restart'] = restart_points_100ms_before
    df_angular_speed_array_stop['angular_speed_500ms_before_restart'] = restart_points_500ms_before
    df_angular_speed_array_stop['angular_speed_in_middle'] = middle_points
    df_angular_speed_array_stop['angular_speed_2s_before_restart'] = restart_points_2s_before
    df_angular_speed_array_stop['angular_speed_3s_after_stop'] = stop_after_3s
    df_angular_speed_array_stop['angular_speed_5s_after_stop'] = stop_after_5s
    df_angular_speed_array_stop['angular_speed_10s_after_stop'] = stop_after_10s
    df_angular_speed_array_stop['angular_speed_20s_after_stop'] = stop_after_20s
    df_angular_speed_array_stop['angular_speed_35s_after_stop'] = stop_after_35s
    
    return df_angular_speed_array_stop,angular_velocity_binned



def bump_amplitude_during_stopPeriod(stop_index_array,bump_amplitude_array, volume_time):
    #Initiate output dataframe and 
    df_bump_amplitude_array_stop = pd.DataFrame()
    stop_points = []
    restart_points = []
    restart_points_100ms_before = []
    restart_points_500ms_before = []
    restart_points_2s_before = []
    stop_after_3s = []
    stop_after_5s = []
    stop_after_10s = []
    stop_after_20s = []
    stop_after_35s = []
    middle_points = []
    frame_100ms_before = int(np.ceil(0.1/volume_time))
    frame_500ms_before = int(np.ceil(0.5/volume_time))
    frame_2s_before = int(np.ceil(2/volume_time))
    frame_3s_after_stop = int(np.ceil(3/volume_time))
    frame_5s_after_stop = int(np.ceil(5/volume_time))
    frame_10s_after_stop = int(np.ceil(10/volume_time))
    frame_20s_after_stop = int(np.ceil(20/volume_time))
    frame_35s_after_stop = int(np.ceil(35/volume_time))
    
    for current_index in range(len(stop_index_array)):
        restart_points.append(bump_amplitude_array[stop_index_array[current_index,0]])
        stop_points.append(bump_amplitude_array[stop_index_array[current_index,0]-stop_index_array[current_index,1]+1])
        middle_points.append(bump_amplitude_array[stop_index_array[current_index,0]-int(np.floor(stop_index_array[current_index,1]/2))])
        index_of_stop = stop_index_array[current_index,0]-stop_index_array[current_index,1]+1
        restart_points_100ms_before.append(bump_amplitude_array[stop_index_array[current_index,0]-frame_100ms_before])
        restart_points_500ms_before.append(bump_amplitude_array[stop_index_array[current_index,0]-frame_500ms_before+1])
        restart_points_2s_before.append(bump_amplitude_array[stop_index_array[current_index,0]-frame_2s_before+1])
        if index_of_stop + frame_3s_after_stop - 1 <= stop_index_array[current_index,0]:
            stop_after_3s.append(bump_amplitude_array[index_of_stop + frame_3s_after_stop - 1])
        else:
            stop_after_3s.append(np.NaN)
        if index_of_stop + frame_5s_after_stop - 1 <= stop_index_array[current_index,0]:
            stop_after_5s.append(bump_amplitude_array[index_of_stop + frame_5s_after_stop - 1])
        else:
            stop_after_5s.append(np.NaN)
        if index_of_stop + frame_10s_after_stop - 1 <= stop_index_array[current_index,0]:
            stop_after_10s.append(bump_amplitude_array[index_of_stop + frame_10s_after_stop - 1])
        else:
            stop_after_10s.append(np.NaN)
        if index_of_stop + frame_20s_after_stop - 1 <= stop_index_array[current_index,0]:
            stop_after_20s.append(bump_amplitude_array[index_of_stop + frame_20s_after_stop - 1])
        else:
            stop_after_20s.append(np.NaN)
        if index_of_stop + frame_35s_after_stop - 1 <= stop_index_array[current_index,0]:
            stop_after_35s.append(bump_amplitude_array[index_of_stop + frame_35s_after_stop - 1])
        else:
            stop_after_35s.append(np.NaN)
    
    
    df_bump_amplitude_array_stop ['bump_amplitude_before_stop'] = stop_points
    df_bump_amplitude_array_stop ['bump_amplitude_at_restart'] = restart_points
    df_bump_amplitude_array_stop ['bump_amplitude_100ms_before_restart'] = restart_points_100ms_before
    df_bump_amplitude_array_stop ['bump_amplitude_500ms_before_restart'] = restart_points_500ms_before
    df_bump_amplitude_array_stop ['bump_amplitude_in_middle'] = middle_points
    df_bump_amplitude_array_stop ['bump_amplitude_2s_before_restart'] = restart_points_2s_before
    df_bump_amplitude_array_stop ['bump_amplitude_3s_after_stop'] = stop_after_3s
    df_bump_amplitude_array_stop ['bump_amplitude_5s_after_stop'] = stop_after_5s
    df_bump_amplitude_array_stop ['bump_amplitude_10s_after_stop'] = stop_after_10s
    df_bump_amplitude_array_stop ['bump_amplitude_20s_after_stop'] = stop_after_20s
    df_bump_amplitude_array_stop ['bump_amplitude_35s_after_stop'] = stop_after_35s
    
    return df_bump_amplitude_array_stop



def PVA_strength_during_stopPeriod(stop_index_array,PVA_strength_array, volume_time):
    #Initiate output dataframe and 
    df_PVA_strength_stop = pd.DataFrame()
    stop_points = []
    restart_points = []
    restart_points_100ms_before = []
    restart_points_500ms_before = []
    restart_points_2s_before = []
    stop_after_3s = []
    stop_after_5s = []
    stop_after_10s = []
    stop_after_20s = []
    stop_after_35s = []
    middle_points = []
    frame_100ms_before = int(np.ceil(0.1/volume_time))
    frame_500ms_before = int(np.ceil(0.5/volume_time))
    frame_2s_before = int(np.ceil(2/volume_time))
    frame_3s_after_stop = int(np.ceil(3/volume_time))
    frame_5s_after_stop = int(np.ceil(5/volume_time))
    frame_10s_after_stop = int(np.ceil(10/volume_time))
    frame_20s_after_stop = int(np.ceil(20/volume_time))
    frame_35s_after_stop = int(np.ceil(35/volume_time))
    duration_stop = []
    PVA_strength_per_sec = []
    PVA_strength_per_sec_second_wise = []
    for current_index in range(len(stop_index_array)):
        duration_stop.append(stop_index_array[current_index,1]*volume_time)
        restart_points.append(PVA_strength_array[stop_index_array[current_index,0]])
        stop_points.append(PVA_strength_array[stop_index_array[current_index,0]-stop_index_array[current_index,1]+1])
        middle_points.append(PVA_strength_array[stop_index_array[current_index,0]-int(np.floor(stop_index_array[current_index,1]/2))])
        index_of_stop = stop_index_array[current_index,0]-stop_index_array[current_index,1]+1
        restart_points_100ms_before.append(PVA_strength_array[stop_index_array[current_index,0]-frame_100ms_before])
        restart_points_500ms_before.append(PVA_strength_array[stop_index_array[current_index,0]-frame_500ms_before+1])
        restart_points_2s_before.append(PVA_strength_array[stop_index_array[current_index,0]-frame_2s_before+1])
        if index_of_stop + frame_3s_after_stop - 1 <= stop_index_array[current_index,0]:
            stop_after_3s.append(PVA_strength_array[index_of_stop + frame_3s_after_stop - 1])
        else:
            stop_after_3s.append(np.NaN)
        if index_of_stop + frame_5s_after_stop - 1 <= stop_index_array[current_index,0]:
            stop_after_5s.append(PVA_strength_array[index_of_stop + frame_5s_after_stop - 1])
        else:
            stop_after_5s.append(np.NaN)
        if index_of_stop + frame_10s_after_stop - 1 <= stop_index_array[current_index,0]:
            stop_after_10s.append(PVA_strength_array[index_of_stop + frame_10s_after_stop - 1])
        else:
            stop_after_10s.append(np.NaN)
        if index_of_stop + frame_20s_after_stop - 1 <= stop_index_array[current_index,0]:
            stop_after_20s.append(PVA_strength_array[index_of_stop + frame_20s_after_stop - 1])
        else:
            stop_after_20s.append(np.NaN)
        if index_of_stop + frame_35s_after_stop - 1 <= stop_index_array[current_index,0]:
            stop_after_35s.append(PVA_strength_array[index_of_stop + frame_35s_after_stop - 1])
        else:
            stop_after_35s.append(np.NaN)
    
        
        #Calculate PVA strength in a frame/second wise fashion
        restart_index_current = stop_index_array[current_index,0]
        stop_points_current = stop_index_array[current_index,0]-stop_index_array[current_index,1]+1
        
        current_stop_strength_array = PVA_strength_array[stop_points_current:restart_index_current] 
        #time unit for cell of calculation
        time_unit_per_cell = 1
   
        # Calculate the number of data points per second
        data_per_second = int(time_unit_per_cell / volume_time)  # 1 second / 0.2 seconds per data point

        # Reshape the data array into a 2D array with each row representing one second
        data_reshaped_non_absolute = current_stop_strength_array[:len(current_stop_strength_array) //  data_per_second* data_per_second].reshape(-1, data_per_second)

        #Calculate the average value per second
        average_per_second_non_absolute = np.mean(data_reshaped_non_absolute, axis=1)

        # Calculate the single averaged value for per second
        PVA_strength_sec_current = np.mean(average_per_second_non_absolute)
        PVA_strength_per_sec.append(PVA_strength_sec_current )
        # Calculate the data that only averaged to one second window 
        PVA_strength_per_sec_second_wise.append(average_per_second_non_absolute)
    
    
    
    df_PVA_strength_stop['PVA_strength_before_stop'] = stop_points
    df_PVA_strength_stop['PVA_strength_at_restart'] = restart_points
    df_PVA_strength_stop['PVA_strength_100ms_before_restart'] = restart_points_100ms_before
    df_PVA_strength_stop['PVA_strength_500ms_before_restart'] = restart_points_500ms_before
    df_PVA_strength_stop['PVA_strength_in_middle'] = middle_points
    df_PVA_strength_stop['PVA_strength_2s_before_restart'] = restart_points_2s_before
    df_PVA_strength_stop['PVA_strength_3s_after_stop'] = stop_after_3s
    df_PVA_strength_stop['PVA_strength_5s_after_stop'] = stop_after_5s
    df_PVA_strength_stop['PVA_strength_10s_after_stop'] = stop_after_10s
    df_PVA_strength_stop['PVA_strength_20s_after_stop'] = stop_after_20s
    df_PVA_strength_stop['PVA_strength_35s_after_stop'] = stop_after_35s
    return df_PVA_strength_stop ,PVA_strength_per_sec_second_wise




def extract_signal_during_stop(stop_index_array, signal_array):
    """
    Extract signal traces during each stop period from the given signal_array.
    
    Parameters:
        stop_index_array (np.ndarray): shape (n_stops, 2), columns are [restart_idx, stop_duration]
        signal_array (np.ndarray): shape (n_ROIs, n_frames) or (n_frames,), raw data
        
    Returns:
        List of np.ndarrays: each element is a slice of signal_array[:, stop_start:restart+1] 
                             (or signal_array[stop_start:restart+1] for 1D input)
    """
    signal_during_stop = []

    is_1d = signal_array.ndim == 1

    for current_index in range(len(stop_index_array)):
        restart_idx = stop_index_array[current_index, 0]
        stop_duration = stop_index_array[current_index, 1]
        stop_start_idx = restart_idx - stop_duration + 1

        if is_1d:
            segment = signal_array[stop_start_idx:restart_idx + 1]
        else:
            segment = signal_array[:, stop_start_idx:restart_idx + 1]

        signal_during_stop.append(segment)

    return signal_during_stop







def calculate_angle_difference_between_two_time_point(pd_start_point, pd_end_point):
    angle_difference = np.zeros(len(pd_start_point))
    for current_point in range(len(pd_start_point)):
        angle_difference[current_point] = pd_end_point[current_point] - pd_start_point[current_point]
        if np.abs(angle_difference[current_point] ) > 180:
            if angle_difference[current_point]  < 0:
                angle_difference[current_point] = angle_difference[current_point]  + 360
            else:
                angle_difference[current_point] = angle_difference[current_point]  - 360
    return angle_difference



def find_qualified_stopping_period (volume_time,stopping_array,PVAinAngle,minimum_frame_length,stable_PVA_threshold):
    #Find qualified stopping index (must meet the minimimun length of demand and must have 1s of active period before)
    persistence_stop_index_and_length_qualified_index = []
    #Only considering PVA stable if 10s later the PVA is still threshold degree within compared to PVA at stop (1 for stable, 0 for not stable)
    for current_index in range(len(stopping_array)):
        start_index = stopping_array[current_index,0]-stopping_array[current_index,1]+1
        #At least 1s after the trial starts
        if start_index * volume_time -1 > 0:
            #Stopping_period must above minimum set duration
            if start_index + minimum_frame_length - 1 <= stopping_array[current_index,0]:
                # At least 1s of active period before the stopping period 
                if current_index == 0:
                    persistence_stop_index_and_length_qualified_index.append(current_index)
                elif (start_index-[stopping_array[current_index-1,0]])*volume_time>1:
                    persistence_stop_index_and_length_qualified_index.append(current_index)
                        
    
    persistence_stop_index_and_length_qualified = np.zeros((len(persistence_stop_index_and_length_qualified_index),2))
    
    for current_index in range(len(persistence_stop_index_and_length_qualified_index)):
        persistence_stop_index_and_length_qualified[current_index,0] = stopping_array[persistence_stop_index_and_length_qualified_index[current_index],0]
        persistence_stop_index_and_length_qualified[current_index,1] = stopping_array[persistence_stop_index_and_length_qualified_index[current_index],1]
    persistence_stop_index_and_length_qualified = persistence_stop_index_and_length_qualified.astype(int) 
    
    
    stable_PVA_index = np.zeros(len(persistence_stop_index_and_length_qualified))
    
    for current_index in range(len(persistence_stop_index_and_length_qualified)):
        #Get PVA angle ar stop
        current_stop_PVA = PVAinAngle[persistence_stop_index_and_length_qualified[current_index,0]-persistence_stop_index_and_length_qualified[current_index,1]+1]
        current_difference_during_stop= PVAinAngle[persistence_stop_index_and_length_qualified[current_index,0]] - current_stop_PVA
        if np.abs(current_difference_during_stop) > 180:
            if current_difference_during_stop < 0:
                current_difference_during_stop =  current_difference_during_stop + 360
            else:
                current_difference_during_stop =  current_difference_during_stop - 360
        
        if np.abs(current_difference_during_stop) <= stable_PVA_threshold:
            stable_PVA_index[current_index] = 1
        else:
            stable_PVA_index[current_index] = 0
    
    
    return persistence_stop_index_and_length_qualified,stable_PVA_index
    

def stopping_period_signal_decay(volume_time, bump_amplitude_stopping_duration,bump_amplitude_stopping_bin_size, signal_array,qualified_stop_array,active_period_before_len):
    #Count the fact that adding 1s of active period before each stopping period
    bump_amplitude_stopping_bin_number =int((bump_amplitude_stopping_duration+active_period_before_len)/bump_amplitude_stopping_bin_size)
    bins_amplitude = []
    for low in range  (0, int(0+100*bump_amplitude_stopping_bin_size*bump_amplitude_stopping_bin_number),int(100*bump_amplitude_stopping_bin_size)):
        bins_amplitude.append((low, int(low+bump_amplitude_stopping_bin_size*100)))
        
    bump_amplitude_stopping_current = np.zeros((len(qualified_stop_array),bump_amplitude_stopping_bin_number))
    
    
    #Binning all the qualified stopping period 
    for current_index in range(len(qualified_stop_array)):
        start_index = int([qualified_stop_array[current_index,0]-qualified_stop_array[current_index,1]+active_period_before_len][0]-np.ceil(1/volume_time))
        temp_amplitude = 0
        temp_count = 0
        bin_index = 0
        for current_binduration_index in range(int(np.ceil((bump_amplitude_stopping_duration+active_period_before_len)/volume_time))):
            if current_binduration_index == int(np.ceil((bump_amplitude_stopping_duration+active_period_before_len)/volume_time)) - 1:
                if current_binduration_index * volume_time * 100 >= bins_amplitude[bin_index][0] and current_binduration_index * volume_time * 100 < bins_amplitude[bin_index][1]:
                    temp_amplitude = temp_amplitude + signal_array[start_index+current_binduration_index]
                    temp_count = temp_count + 1
                    bump_amplitude_stopping_current[current_index, bin_index] = temp_amplitude/temp_count
                else:
                    bump_amplitude_stopping_current[current_index, bin_index] = temp_amplitude/temp_count
                    temp_amplitude = signal_array[start_index+current_binduration_index]
                    temp_count = 1
                    bin_index = bin_index + 1
                    bump_amplitude_stopping_current[current_index, bin_index] = temp_amplitude/temp_count
            elif current_binduration_index * volume_time * 100 >= bins_amplitude[bin_index][0] and current_binduration_index * volume_time * 100 < bins_amplitude[bin_index][1]:
                temp_amplitude = temp_amplitude + signal_array[start_index+current_binduration_index]
                temp_count = temp_count + 1
            else:
                bump_amplitude_stopping_current[current_index, bin_index] = temp_amplitude/temp_count
                temp_amplitude = signal_array[start_index+current_binduration_index]
                temp_count = 1
                bin_index = bin_index + 1
    return bump_amplitude_stopping_current



def plot_stopping_period_signal(signal_df, stopping_bin_size_s,decay_start_from_exact_stop,decay_length_s,active_period_length_s,run_spline_fit,celltype, trial_condition):
    
    sample_per_second = int(1/stopping_bin_size_s)
    actual_stop_start_index = 0 + active_period_length_s*sample_per_second
    end_of_decay_index = actual_stop_start_index+ decay_length_s*sample_per_second-1    
    if decay_start_from_exact_stop == 1:
        average_signal = round((signal_df.mean(axis=1)[actual_stop_start_index]-signal_df.mean(axis=1)[end_of_decay_index])/np.abs(signal_df.mean(axis=1)[actual_stop_start_index])*100,2)
    else:
        average_signal = round((signal_df.mean(axis=1)[0]-signal_df.mean(axis=1)[end_of_decay_index])/signal_df.mean(axis=1)[0]*100,2)
        
    
    
 
    time_array_decay = np.arange(len(signal_df.mean(axis=1)))/sample_per_second
    
    
    
    plt.figure(figsize=(15,8))
    for i in range(len(signal_df.columns)):
        plt.plot(time_array_decay,signal_df[i],linewidth = 0.7)
    plt.plot(time_array_decay,signal_df.mean(axis=1), linewidth=8, color = 'navy')
    
    if run_spline_fit == True:
        spl = UnivariateSpline(time_array_decay,signal_df.mean(axis=1),k=2, s=3)
        plt.plot(time_array_decay,spl(time_array_decay),linewidth=2, color ='red')
        cutpoint =np.absolute(np.gradient(spl(time_array_decay))-0).argmin()
        plt.axvline(x= cutpoint*0.2 , color = 'b',linestyle ='--')
        mean = signal_df.mean(axis=1)
        tau = np.around(np.absolute(mean[0:cutpoint]-mean[0]*0.632).argmin()*0.2, decimals=2)
        stable_mean = np.around(np.mean(mean[cutpoint:]),decimals=2)
        plt.title(f"Bump amplitude over {decay_length_s}s of stop {celltype, trial_condition}, \n Average decay by {average_signal}%, decay period tau = {tau}s, stable period avaerage = {stable_mean} ",fontsize =20)
    else:
        plt.title(f"Bump amplitude over {decay_length_s}s of stop {celltype, trial_condition}, \n Average decay by {average_signal}%",fontsize =20)
        
    
    plt.xticks(ticks=plt.xticks()[0][0:], labels=np.array(plt.xticks()[0][0:]-active_period_length_s, dtype=np.int64), fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlim(0,decay_length_s+0.8)
    plt.axvline(x= active_period_length_s, color = 'r',linestyle ='--')
    plt.xlabel('Time(s)', fontsize=20)
    plt.ylabel('Bump Amplitude', fontsize=20)
    plt.show()


    
    
#Get average signal decay from each single fly   
def calculate_stop_signal_in_separate_fly(df_bump_decay_all_trial,flylist,trial_list,stability_list, separate_stableY_N):
    mean_signal_pooled = pd.DataFrame()
    count = 0
    for fly_index in range(len(flylist)):
        same_fly_trial_index = np.where(np.array(trial_list) == flylist[fly_index])[0]
        if  separate_stableY_N == 1:
            same_fly_trial_index = [idx for idx in same_fly_trial_index if stability_list[idx]==1]
        if (len(same_fly_trial_index)) > 0:
            current_fly_df = df_bump_decay_all_trial.iloc[:,same_fly_trial_index] 
            if count == 0:
                mean_signal_pooled = current_fly_df.mean(axis=1)
            else:
                mean_signal_current = current_fly_df.mean(axis=1)
                mean_signal_pooled = pd.concat([mean_signal_pooled,mean_signal_current], axis=1)
            count = count + 1
    mean_signal_pooled.columns = range(len(mean_signal_pooled.columns))
    return mean_signal_pooled


def run_ANOVA_for_comparing_across_genotypes(dataframe_input, subcolumn):
    df_input = dataframe_input[subcolumn]
    columns_to_test = df_input.columns
    all_values = [df_input [column].dropna().values for column in df_input]
    anova_result = f_oneway(*all_values)


    # Calculate the adjusted degrees of freedom
    df_between = len(columns_to_test) - 1
    df_within = len(np.concatenate(all_values)) - len(columns_to_test)



    # Print the ANOVA result
    print("One-way ANOVA Result:")
    print("F-statistic:", anova_result.statistic)
    print("p-value:", anova_result.pvalue)
    print("Adjusted df between:", df_between)
    print("Adjusted df within:", df_within)
    
    
    if anova_result.pvalue < 0.05:
        # Create a list of labels corresponding to each group
        group_labels = np.concatenate([[column] * len(values) for column, values in zip(columns_to_test, all_values)])
        # Perform Tukey-Kramer post hoc test
        tukey_results = pairwise_tukeyhsd(np.concatenate(all_values), group_labels)

        # Print the results
        print(tukey_results)

        
        
def get_PVA_angle_and_forwared_speed_during_stop(PVA_angle_array, speed_array,persistence_stop_index_and_length_array):
    
    difference_during_stop = []
    current_difference_during_stop = []
    current_forward_speed_during_stop = []
    forward_speed_during_stop =[]
    
    for current_index in range(len(persistence_stop_index_and_length_array)):
        current_stop_PVA = PVA_angle_array[persistence_stop_index_and_length_array[current_index,0]-persistence_stop_index_and_length_array[current_index,1]+1]
        current_stop_start_index = persistence_stop_index_and_length_array[current_index,0]-persistence_stop_index_and_length_array[current_index,1]+1
        current_difference_during_stop = PVA_angle_array[current_stop_start_index:persistence_stop_index_and_length_array[current_index,0]+1] - current_stop_PVA
        current_forward_speed_during_stop = speed_array[current_stop_start_index:persistence_stop_index_and_length_array[current_index,0]+1]
        #Modify the value so angle difference will never be different more than 180 degrees
        for j in range(len(current_difference_during_stop)):
            if np.abs(current_difference_during_stop[j]) > 180:
                if current_difference_during_stop[j] < 0:
                    current_difference_during_stop[j] =  current_difference_during_stop[j] + 360
                else:
                    current_difference_during_stop[j] =  current_difference_during_stop[j] - 360
        difference_during_stop.append(current_difference_during_stop)
        forward_speed_during_stop.append(current_forward_speed_during_stop)
    
    return difference_during_stop, forward_speed_during_stop
        

    
    
    
    
    
# Sliding window function for circular variance
def sliding_window_circular_variance(offset_array, behavior_state, strength_array, window_size, time_per_frame,step_size_frames):
    variances = []
    num_frames = len(offset_array)
    
    # Convert window size and step size from seconds to frames 
    window_size_frames =  int(np.ceil(window_size/time_per_frame))
    
    
    # Calculate the mean and standard deviation of the strength array
    mean_strength = np.mean(strength_array)
    std_strength = np.std(strength_array)
    
    # Define the threshold for low strength (one SD below the mean)
    strength_threshold = mean_strength - std_strength
    
    
    # Slide over the data
    for start in range(0, num_frames - window_size_frames + 1, step_size_frames):
        end = start + window_size_frames
        
        # Get the circular data in the current window
        offset_window = offset_array[start:end]
        
        # Get the behavior state in the current window
        behavior_window = behavior_state[start:end]
        
        # Get the strength data in the current window
        strength_window = strength_array[start:end]
        
        # Calculate the proportion of walking frames (assuming 1 = walking, 0 = stopping)
        walking_ratio = np.sum(behavior_window == 1) / len(behavior_window)
        
        # Calculate the average strength in the window
        avg_strength = np.mean(strength_window)
        
        # Drop window if majority of frames are in stopping period
        if walking_ratio < 0.7 or avg_strength < strength_threshold:
            continue  # Skip this window
        
        # Calculate circular variance in the current window
        variance = circvar(offset_window,high=np.pi, low= -np.pi)
        variances.append(variance)
    
    return variances    
    
    

    
    

    
    
    
    
    
def plot_fly_data_histograms_fly_wise(data_drift, data_strength, plot_individual=False, different_color_for_fly=False, label=None, color=None):
    # Get a colormap for multiple datasets
    cmap = plt.get_cmap('tab10')  # Using 'tab10' colormap for different colors

    # Create a figure only if one doesn't exist (for multiple datasets on the same plot)
    if not plt.fignum_exists(1):
        plt.figure(figsize=(10, 6))  # Create the figure if it doesn't exist
    
    # Initialize an empty array to store all data if plotting in a single color
    all_flattened_data = []

    # Collect all strength data to compute the threshold
    all_strength_data = []
    for fly, trials in data_strength['output_second_wise_PVA_strength'].items():
        for trial in trials:
            for time_bin in trial:
                all_strength_data.extend(time_bin)

    # Calculate the overall mean and standard deviation of the strength data
    mean_strength = np.mean(all_strength_data)
    std_strength = np.std(all_strength_data)
    
    # Threshold for qualifying data (1 SD below mean)
    strength_threshold = mean_strength - std_strength

    # Loop over each fly's data and strength data to filter and plot
    for idx, (key_drift, trials_drift) in enumerate(data_drift['output_second_wise_absolute_bump_drift'].items()):
        trials_strength = data_strength['output_second_wise_PVA_strength'][key_drift]

        # Flatten the data for each fly, filtering out data below the strength threshold
        try:
            filtered_drift_data = []
            for trial_drift, trial_strength in zip(trials_drift, trials_strength):
                # Filter each time bin in the trial
                for drift_bin, strength_bin in zip(trial_drift, trial_strength):
                    if isinstance(drift_bin, np.ndarray):
                        filtered_drift_bin = drift_bin[strength_bin >= strength_threshold]
                        filtered_drift_data.extend(filtered_drift_bin)

            # Select a color based on the index of the dataset
            current_color = cmap(idx / len(data_drift['output_second_wise_absolute_bump_drift'])) if color is None else color

            if plot_individual:
                # Plot each fly's data individually
                plt.figure(figsize=(10, 6))
                n, bins, patches = plt.hist(filtered_drift_data, bins=30, alpha=0.75, color=current_color, edgecolor='black')
                plt.axvline(np.mean(filtered_drift_data), color='r', linestyle='dashed', linewidth=2, label=f'Mean: {np.mean(filtered_drift_data):.2f}')
                plt.title(f"Histogram of Values for {key_drift}")
                plt.xlabel("Value")
                plt.ylabel("Frequency")
                plt.legend()
                plt.show()
            elif different_color_for_fly:
                # Plot histograms with different colors for each fly
                n, bins, patches = plt.hist(filtered_drift_data, bins=30, alpha=0.75, color=current_color, edgecolor='black', label=key_drift)
            else:
                # Accumulate all data into one array for single-color plotting
                all_flattened_data.extend(filtered_drift_data)

        except ValueError:
            print(f"No data available for {key_drift}. Skipping...")

    # Plot a single histogram for all flies if different_color_for_fly is False
    if not plot_individual and not different_color_for_fly:
        # Plot a single histogram with a single color
        n, bins, patches = plt.hist(all_flattened_data, bins=30, alpha=0.75, color='blue' if color is None else color, edgecolor='black', label=label)
        # Add a line for the mean
        mean_value = np.mean(all_flattened_data)
        plt.axvline(mean_value, color='red', linestyle='dashed', linewidth=2, label=f"Mean: {mean_value:.2f}")
    
    # Add labels and title if this is the final plot
    if not plot_individual:
        plt.title("Histogram of Bump Drift Rate Across All Flies (Filtered by Strength)", fontsize=16, fontweight='bold')
        plt.xlabel("Drift Rate (°/s)", fontsize=14)
        plt.ylabel("Counts (# of 1s-bin)", fontsize=14)
        plt.legend()    
    

# Global variable to store max y limit across datasets
global_max_y = 0    
    
    
def plot_fly_data_density_fly_wise(data_drift, data_strength, plot_individual=False, different_color_for_fly=False, label=None, color=None, show_bins=True, fill_under_curve=False, reset_y_lim=False):
    global global_max_y
    filtered_data_by_fly = {}

    if reset_y_lim:
        global_max_y = 0

    cmap = plt.get_cmap('tab10')

    if not plt.fignum_exists(1):
        plt.figure(figsize=(10, 6))

    all_flattened_data = []
    all_strength_data = [value for trials in data_strength['output_second_wise_PVA_strength'].values() for trial in trials for time_bin in trial for value in time_bin]
    mean_strength = np.mean(all_strength_data)
    std_strength = np.std(all_strength_data)
    strength_threshold = mean_strength - std_strength

    bin_edges = np.linspace(0, 60, 30)

    for idx, (key_drift, trials_drift) in enumerate(data_drift['output_second_wise_absolute_bump_drift'].items()):
        trials_strength = data_strength['output_second_wise_PVA_strength'][key_drift]

        filtered_drift_data = [
            drift_bin[strength_bin >= strength_threshold]
            for trial_drift, trial_strength in zip(trials_drift, trials_strength)
            for drift_bin, strength_bin in zip(trial_drift, trial_strength) if isinstance(drift_bin, np.ndarray)
        ]
        
        filtered_drift_data = np.concatenate(filtered_drift_data) if filtered_drift_data else np.array([])

        # Only proceed if the filtered data has at least two elements
        if filtered_drift_data.size > 1:
            filtered_data_by_fly[key_drift] = filtered_drift_data
            current_color = cmap(idx / len(data_drift['output_second_wise_absolute_bump_drift'])) if color is None else color
            kde = gaussian_kde(filtered_drift_data)
            x_vals = np.linspace(min(filtered_drift_data) - 1, max(filtered_drift_data) + 1, 1000)
            density_vals = kde(x_vals)

            if plot_individual:
                plt.figure(figsize=(10, 6))

                if show_bins:
                    hist_vals, _, _ = plt.hist(filtered_drift_data, bins=bin_edges, density=True, alpha=0.3, color=current_color, edgecolor='black')
                    global_max_y = max(global_max_y, np.max(hist_vals))

                plt.plot(x_vals, density_vals, color=current_color, linewidth=2, label=f"{key_drift} Mean: {np.mean(filtered_drift_data):.2f}")
                global_max_y = max(global_max_y, np.max(density_vals))

                if fill_under_curve:
                    plt.fill_between(x_vals, density_vals, color=current_color, alpha=0.3)

                plt.ylim(0, global_max_y * 1.1)
                plt.show()
            elif different_color_for_fly:
                plt.plot(x_vals, density_vals, color=current_color, linewidth=2, label=key_drift)

                if show_bins:
                    hist_vals, _, _ = plt.hist(filtered_drift_data, bins=bin_edges, density=True, alpha=0.3, color=current_color, edgecolor='black')
                    global_max_y = max(global_max_y, np.max(hist_vals))
                
                if fill_under_curve:
                    plt.fill_between(x_vals, density_vals, color=current_color, alpha=0.3)

                global_max_y = max(global_max_y, np.max(density_vals))
            else:
                all_flattened_data.extend(filtered_drift_data)

    if not plot_individual and not different_color_for_fly:
        kde = gaussian_kde(all_flattened_data)
        x_vals = np.linspace(min(all_flattened_data) - 1, max(all_flattened_data) + 1, 1000)
        density_vals = kde(x_vals)
        plt.plot(x_vals, density_vals, color='blue' if color is None else color, linewidth=2, label=label)
        
        if show_bins:
            hist_vals, _, _ = plt.hist(all_flattened_data, bins=bin_edges, density=True, alpha=0.3, color='blue' if color is None else color, edgecolor='black')
            global_max_y = max(global_max_y, np.max(hist_vals))

        if fill_under_curve:
            plt.fill_between(x_vals, density_vals, color='blue' if color is None else color, alpha=0.3)

        plt.axvline(np.mean(all_flattened_data), color='red', linestyle='dashed', linewidth=2, label=f"Mean: {np.mean(all_flattened_data):.2f}")
        global_max_y = max(global_max_y, np.max(density_vals))

    plt.ylim(0, global_max_y * 1.1)

    if not plot_individual:
        plt.title("Density Plot of Bump Drift Rate Across All Flies (Filtered by Strength)", fontsize=16, fontweight='bold')
        plt.xlabel("Drift Rate (°/s)", fontsize=25)
        plt.ylabel("Density", fontsize=25)
        plt.xlim(0, 60)
        plt.legend(fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

    # Return the filtered data for each fly
    return filtered_data_by_fly
  
    
def permutation_test_median(data1, data2, num_permutations=10000, significance_level=0.05, effect_size_threshold=0.0):
    # Combine data and calculate the observed difference in medians adjusted by the effect size threshold
    combined_data = np.concatenate([data1, data2])
    observed_diff = abs(np.median(data1) - np.median(data2)) - abs(effect_size_threshold)  # Adjust with absolute effect size

    # Calculate permutation differences
    perm_diffs = []
    for _ in range(num_permutations):
        np.random.shuffle(combined_data)
        perm_data1 = combined_data[:len(data1)]
        perm_data2 = combined_data[len(data1):]
        perm_diff = abs(np.median(perm_data1) - np.median(perm_data2))  # Use absolute difference
        perm_diffs.append(perm_diff)

    # Calculate p-value, considering only permutations that meet the adjusted observed difference
    extreme_count = sum(perm_diff >= observed_diff for perm_diff in perm_diffs)
    p_value = extreme_count / num_permutations

    print(f"Observed Difference in Medians (Adjusted by Effect Size Threshold): {observed_diff}")
    print(f"Permutation Test (Median): p-value = {p_value} (Significance level = {significance_level})")
    
    if p_value < significance_level:
        print("Result: Significant difference at chosen significance level")
    else:
        print("Result: No significant difference")

    # Return adjusted observed difference and p-value for external use
    return observed_diff, p_value




def plot_stop_drift_rate_heatmaps(data_all, data_key):
    """
    Processes and plots heatmaps for each experiment separately.

    Parameters:
    - data_all (dict): Dictionary containing multiple datasets.
    - data_key (str): Key name in `data_all` (e.g., "output_second_wise_PVA_strength").
    """

    # Extract the relevant data using the provided key
    if data_key not in data_all:
        raise KeyError(f"'{data_key}' not found in the provided dictionary.")

    data_dict = data_all[data_key]
    print(f"Processing dataset: {data_key}")

    for experiment, trials in data_dict.items():
        print(f"Processing experiment: {experiment}")

        combined_data = []

        for trial in trials:
            # Collect trial data if it's not empty
            trial_data = [np.array(array) for array in trial if len(array) > 0]

            if len(trial_data) > 0:
                # Find the maximum length of the arrays for padding
                max_length = max(len(arr) for arr in trial_data)

                # Pad arrays to the same length with NaN
                padded_data = np.array([
                    np.pad(arr, (0, max_length - len(arr)), 'constant', constant_values=np.nan)
                    for arr in trial_data
                ])

                # Append to combined_data
                combined_data.extend(padded_data)

        if len(combined_data) == 0:
            print(f"No valid data found for experiment: {experiment}, skipping...")
            continue

        # Convert to DataFrame
        df = pd.DataFrame(combined_data)

        # Plot heatmap for this experiment
        plt.figure(figsize=(30, 15))
        sns.heatmap(df, cmap="coolwarm", annot=False, center=0)
        plt.title(f'Combined Heatmap for {experiment} ({data_key})', fontsize=20)
        plt.xlabel('Second', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.ylabel('Stop Trial', fontsize=20)
        plt.show()

    print(f"Done plotting heatmaps for {data_key}.")








def run_fixed_heading_period_analysis_across_trial(directory,dual_imaging,genotype,trial_condition,Bump_amplitude_stopping_duration,ROI_type):
    
    
    # Part 1: Create and add each DataFrame to the dictionary
    count = 0
    output_pooled_dictionary = {}
    output_pooled_dictionary['output_df_pooled'] = pd.DataFrame()
    output_pooled_dictionary['output_bump_amplitude_V3_pooled'] = pd.DataFrame()
    output_pooled_dictionary['output_bump_amplitude_V4_pooled'] = pd.DataFrame()
    output_pooled_dictionary['output_bump_width_pooled'] = pd.DataFrame()
    output_pooled_dictionary['output_PVA_strength_pooled'] = pd.DataFrame()
    output_pooled_dictionary['output_Angular_Speed_pooled'] = pd.DataFrame()
    output_pooled_dictionary['strong_PVA_chunk_pooled'] = []
    output_pooled_dictionary['strong_PVA_chunk_behavior_state'] = []
    output_pooled_dictionary['PVA_at_strong_signal'] = []
    output_pooled_dictionary['mean_PVA_strength_per_trial_pooled'] = []
    output_pooled_dictionary['output_stable_PVA_index_pooled'] = []
    output_pooled_dictionary['circular_variance'] = []
    output_pooled_dictionary['circular_variance_slide_window_during_walking'] = []
    output_pooled_dictionary['Average_bump_shape'] = pd.DataFrame()
    output_pooled_dictionary['Average_bump_shape_strong_PVA'] = pd.DataFrame()
    output_pooled_dictionary['Bump_shape_at_stop'] = pd.DataFrame()
    output_pooled_dictionary['Bump_shape_at_stop_more_than_10s_trial_only'] = pd.DataFrame()
    output_pooled_dictionary['Bump_shape_at_5s_after_stop'] = pd.DataFrame()
    output_pooled_dictionary['Bump_shape_at_10s_after_stop'] = pd.DataFrame()
    output_pooled_dictionary['Bump_shape_at_20s_after_stop'] = pd.DataFrame()
    output_pooled_dictionary['max_Raw_F_at_stop'] = []
    output_pooled_dictionary['max_Raw_F_at_10s_after_stop'] = []
    output_pooled_dictionary['output_flytrial'] = []
    output_pooled_dictionary['output_flytrial_for_qualified_stop'] = []
    output_pooled_dictionary['output_second_wise_bump_drift'] = {}
    output_pooled_dictionary['output_second_wise_bump_drift_at_stop_thresholded'] = {}
    output_pooled_dictionary['output_second_wise_bump_drift_at_stop_thresholded_absolute'] = {}
    output_pooled_dictionary['output_second_wise_bump_drift_right_before_stop'] = {}
    output_pooled_dictionary['output_second_wise_bump_drift_right_before_stop_thresholded'] = {}
    output_pooled_dictionary['output_second_wise_angular_velocity_right_before_stop'] = {}
    output_pooled_dictionary['output_second_wise_angular_velocity_during_stop'] = {}
    output_pooled_dictionary['output_second_wise_PVA_position_right_before_stop'] = {}
    output_pooled_dictionary['output_second_wise_PVA_position_during_stop'] = {}
    output_pooled_dictionary['output_second_wise_PVA_position_during_stop_thresholded'] = {}
    output_pooled_dictionary['output_second_wise_absolute_bump_drift'] = {}
    output_pooled_dictionary['output_second_wise_bump_dwell_difference'] = {}
    output_pooled_dictionary['output_second_wise_PVA_strength'] = {}
    output_pooled_dictionary['output_PVA_heading_offset'] = {}
    output_pooled_dictionary['PVA_at_strong_signal'] = {}
    output_pooled_dictionary['dff_8_roi_at_stop'] = {}
    output_pooled_dictionary['PVA_strength_frame_wise_at_stop'] = {}
    output_pooled_dictionary['PVA_Angle_frame_wise_at_stop'] = {}
    
    if dual_imaging == 1:
        output_pooled_dictionary['output_df_pooled_red'] = pd.DataFrame()
        output_pooled_dictionary['output_bump_amplitude_V3_pooled_red'] = pd.DataFrame()
        output_pooled_dictionary['output_bump_amplitude_V4_pooled_red'] = pd.DataFrame()
        output_pooled_dictionary['output_bump_width_pooled_red'] = pd.DataFrame()
        output_pooled_dictionary['output_PVA_strength_pooled_red'] = pd.DataFrame()
        output_pooled_dictionary['output_stable_PVA_index_pooled_red'] = []
        output_pooled_dictionary['strong_PVA_chunk_pooled_red'] = []
        output_pooled_dictionary['mean_PVA_strength_per_trial_pooled_red'] = []
        output_pooled_dictionary['circular_variance_red']
        output_pooled_dictionary['output_lag_s'] = []
  
    
    
    
    
    
    
    #Part 2:import data
    # import data
    for single_df in os.listdir(directory):
        current_file = pd.read_csv(sep.join([directory, single_df]))
        #Get time info
        volume_cycle = len(current_file )
        volume_time = current_file['Time_Stamp'][1]
        volume_rate = 1/volume_time
        time_array_imaging = np.arange(volume_cycle)/volume_rate
        #Get necessary parameters
        PVA_Unwrapped_Radian = current_file['Unwrapped_Radian'].values
        PVA_Angle = current_file['PVA_Angle'].values
        PVA_Radian = current_file['PVA_Radian'].values
        Angular_velocity = current_file['Angular_Velocity'].values
        integrated_x = current_file['Integrated_x'].values
        PVA_strength = current_file['PVA_strength'].values
        Bump_amplitude = current_file['Bump_amplitude'].values
        Unwrapped_heading = current_file['Unwrapped_Bar_Position/Heading'].values
        Wrapped_heading = current_file['Bar_Position/Heading'].values
    
        if dual_imaging == 1:
            PVA_Angle_red = current_file['PVA_Angle_red'].values
            PVA_Radian_red = current_file['PVA_Radian_red'].values
            PVA_strength_red = current_file['PVA_strength_red'].values
            filtered_PVA_radian = low_pass_filter_TQ(moving_wrapped_plot_by_offset(PVA_Radian,math.degrees(np.pi), ifRadian =True),0.5,volume_rate)
            filtered_PVA_radian_red = low_pass_filter_TQ(PVA_Radian_red,0.5,volume_rate)
            correlation = correlate(filtered_PVA_radian, filtered_PVA_radian_red, mode='full')
            lag_between_channels = np.argmax(correlation) - len(filtered_PVA_radian_red) + 1
            lag_between_channels_s = lag_between_channels*volume_time
            output_pooled_dictionary['output_lag_s'].append(lag_between_channels_s)
    
    
        dff_normalized_8_roi = np.array([current_file[f'dFF_Roi_{i}'] for i in range(1, 9)]).T
        columns = ['dFF_Roi_5', 'dFF_Roi_6', 'dFF_Roi_7', 'dFF_Roi_8', 'dFF_Roi_1', 'dFF_Roi_2', 'dFF_Roi_3', 'dFF_Roi_4']
        data = [current_file[col].to_numpy() for col in columns]
        dff_normalized_8_roi_shifted = np.column_stack(data)
        dff_normalized_8_roi_shifted = dff_normalized_8_roi_shifted.transpose()
        
        # array for 16 ROIs (specific for bump width analysis)
        #raw_f_16_roi = np.array([current_file[f'Raw_F_{i}'] for i in range(1, 17)]).T
        #For E-PG/P-EG,P-ENb PB type ROI
        if ROI_type == 1: 
            columns = ['Raw_F_5', 'Raw_F_14', 'Raw_F_6', 'Raw_F_15', 'Raw_F_7', 'Raw_F_16','Raw_F_8','Raw_F_9','Raw_F_1','Raw_F_10','Raw_F_2','Raw_F_11','Raw_F_3','Raw_F_12','Raw_F_4', 'Raw_F_13']
        #For delta7 type ROI
        elif ROI_type == 2:
            columns = ['Raw_F_5', 'Raw_F_13', 'Raw_F_6', 'Raw_F_14', 'Raw_F_7', 'Raw_F_15', 'Raw_F_8', 'Raw_F_16','Raw_F_1','Raw_F_9','Raw_F_2','Raw_F_10','Raw_F_3','Raw_F_11','Raw_F_4','Raw_F_12']
        # For EB ROI: 
        else:
            columns = ['Raw_F_1', 'Raw_F_2', 'Raw_F_3', 'Raw_F_4', 'Raw_F_5', 'Raw_F_6', 'Raw_F_7', 'Raw_F_8','Raw_F_9','Raw_F_10','Raw_F_11','Raw_F_12','Raw_F_13','Raw_F_14','Raw_F_15','Raw_F_16']
        raw_data_16_ROI = [current_file[col].to_numpy() for col in columns]
        raw_data_16_ROI = np.array(raw_data_16_ROI).T
        dff_16_roi = get_dff_array(raw_F_array = raw_data_16_ROI, ROI_num = 16, F_zero_cutoff = 0.05, if_plot =0)
        dff_normalized_16_roi = normalizing_dff_array(dff_16_roi,ROI_num= 16, normalize_cutoff= 0.95,if_plot =0)
        dff_normalized_16_roi_shifted = np.column_stack(dff_normalized_16_roi)
        #dff_normalized_16_roi_shifted = dff_normalized_16_roi_shifted.transpose()
        
        raw_F_max = np.max(raw_data_16_ROI,axis =1)
        
        
        
        
        if dual_imaging == 1:
            dff_normalized_8_roi_red = np.array([current_file[f'dFF_Roi_{i}_red'] for i in range(1, 9)]).T
            columns_red = ['dFF_Roi_5_red', 'dFF_Roi_6_red', 'dFF_Roi_7_red', 'dFF_Roi_8_red', 'dFF_Roi_1_red', 'dFF_Roi_2_red', 'dFF_Roi_3_red', 'dFF_Roi_4_red']
            data_red = [current_file[col].to_numpy() for col in columns_red]
            dff_normalized_8_roi_shifted_red = np.column_stack(data_red)
            dff_normalized_8_roi_shifted_red = dff_normalized_8_roi_shifted_red.transpose()
    
        integrated_x_unwrapped =  fictrack_signal_decoding(integrated_x,time_array_imaging, 10, already_radian = True)
        Forward_velocity = np.gradient(integrated_x_unwrapped)/volume_time
        Forward_speed_radian = np.abs(Forward_velocity)
        Forward_speed_degrees =Forward_speed_radian * 180/np.pi
        Angular_speed_degrees =  np.abs(Angular_velocity) * 180/np.pi
        #Get persistence period
        persistence_stop_index_and_length = find_stop_period_on_heading(head_velocity_array = Angular_velocity,degree_of_tolerance =5,shortest_stopFrame=int(np.ceil(3/volume_time)))
        
        #Create a index array that indicates whether the current frame is stop/active stop index = 0, active index =1
        behavior_state_frame_index = np.ones(len(PVA_Radian))
        for current_index in range(len(persistence_stop_index_and_length)):
            start_index = persistence_stop_index_and_length[current_index,0]-persistence_stop_index_and_length[current_index,1]+1
            end_index = persistence_stop_index_and_length[current_index,0]
            currrent_stop_duration = end_index - start_index +1
            behavior_state_frame_index[start_index:end_index+1] = [0] * currrent_stop_duration
    
        
      
    
        #Average bump shape peak centered
        df_dff_in_ROI_normalized_shifted_peak_centered = np.zeros((len(dff_normalized_16_roi_shifted),       len(dff_normalized_16_roi_shifted[0])))

        shifted_by_all = []

        for i in range(len(dff_normalized_16_roi_shifted[0])):
            original_order = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
            current_peak = np.argmax(dff_normalized_16_roi_shifted[:,i])
            shift_by = current_peak - 8
            shifted_order = original_order[shift_by % len(original_order):] + original_order[:shift_by % len(original_order)]
            df_dff_in_ROI_normalized_shifted_peak_centered[:,i] = dff_normalized_16_roi_shifted[shifted_order,i]
            shifted_by_all.append(shift_by)
    
        average_bump_shape = np.mean(df_dff_in_ROI_normalized_shifted_peak_centered, axis=1)
        
        
        
        #Same thing for 8 ROIs
        df_dff_in_ROI_normalized_shifted_peak_centered_8_ROI = np.zeros((len(dff_normalized_8_roi_shifted),       len(dff_normalized_8_roi_shifted[0])))
        shifted_by_all_8_ROI = []
        
        for i in range(len(dff_normalized_8_roi_shifted[0])):
            original_order = [0,1,2,3,4,5,6,7]
            current_peak = np.argmax(dff_normalized_8_roi_shifted[:,i])
            shift_by = current_peak - 3
            shifted_order = original_order[shift_by % len(original_order):] + original_order[:shift_by % len(original_order)]
            df_dff_in_ROI_normalized_shifted_peak_centered_8_ROI[:,i] = dff_normalized_8_roi_shifted[shifted_order,i]
            shifted_by_all_8_ROI.append(shift_by)
    
        average_bump_shape_8_ROI = np.mean(df_dff_in_ROI_normalized_shifted_peak_centered_8_ROI, axis=1)
        average_bump_shape_8_ROI =pd.DataFrame(average_bump_shape_8_ROI)
    
    
    
    
        #Create a index array that indicates whether the current frame is stop/active stop index = 0, active index =1
        behavior_state_frame_index = np.ones(len(PVA_Radian))
        for current_index in range(len(persistence_stop_index_and_length)):
            start_index = persistence_stop_index_and_length[current_index,0]-persistence_stop_index_and_length[current_index,1]+1
            end_index = persistence_stop_index_and_length[current_index,0]
            currrent_stop_duration = end_index - start_index +1
            behavior_state_frame_index[start_index:end_index+1] = [0] * currrent_stop_duration

    
        
    
    
        ##Part3: get signal during the stopping period
        
        #3.0 Get frame_wise signal of dff during the stop
        dff_8_roi_at_stop_current = extract_signal_during_stop(persistence_stop_index_and_length, dff_normalized_8_roi.T)
        PVA_strength_at_stop_frame_wise_current = extract_signal_during_stop(persistence_stop_index_and_length, PVA_strength)
        PVA_Angle_at_stop_frame_wise_current = extract_signal_during_stop(persistence_stop_index_and_length, PVA_Angle)
        
    
        # 3.1:Calculate annd plot difference throughout the stop period
        difference_during_stop, forward_speed_during_stop = get_PVA_angle_and_forwared_speed_during_stop(PVA_Angle, Forward_speed_degrees,persistence_stop_index_and_length)
    
        if dual_imaging == 1:
            difference_during_stop_red, forward_speed_during_stop = get_PVA_angle_and_forwared_speed_during_stop(PVA_Angle_red, Forward_speed_degrees,persistence_stop_index_and_length)
        
    
        # 3.2: calculate bump amplitude by getting the normalized dff at current PVA angle 
        Bump_amplitude_V3, Bump_amplitude_V3_opposite = calcualteBumpAmplitude_V3(dff_normalized_8_roi,PVA_Radian)
        Bump_amplitude_V4 = calcualteBumpAmplitude_V4(Bump_amplitude_V3,dff_normalized_8_roi,PVA_Radian,persistence_stop_index_and_length)    
        if dual_imaging == 1:
            Bump_amplitude_V3_red, Bump_amplitude_V3_opposite_red = calcualteBumpAmplitude_V3(dff_normalized_8_roi_red,PVA_Radian_red)
            Bump_amplitude_V4_red =  calcualteBumpAmplitude_V4(Bump_amplitude_V3_red,dff_normalized_8_roi_red,PVA_Radian_red,persistence_stop_index_and_length)    
    
        # 3.3: Average PVA_strength per trial
        output_pooled_dictionary['mean_PVA_strength_per_trial_pooled'].append(np.mean(PVA_strength))
        if dual_imaging == 1:
            output_pooled_dictionary['mean_PVA_strength_per_trial_pooled_red'].append(np.mean(PVA_strength_red))
    
        # 3.4: Get strong PVA chunk or signal chunk and Bump shape associated
        strength_threshold = 0.6
        strength_threshold_2_sd = np.mean(PVA_strength) + np.std(PVA_strength)
        min_window_s = 2
        current_strong_PVA_chunk = strong_PVA_duration(PVA_strength, strength_threshold ,volume_time, min_window_s)
        current_strong_PVA_index = strong_PVA_index(PVA_strength, strength_threshold_2_sd ,volume_time, 0.5)
        
        
        average_dff_normalized = np.mean(dff_normalized_8_roi,axis = 1)
        signal_threshold_1_sd = np.mean(average_dff_normalized) + np.std(average_dff_normalized)
        signal_threshold_1_sd_weak = np.mean(average_dff_normalized) - np.std(average_dff_normalized)
        current_strong_signal_index = strong_signal_index(average_dff_normalized, signal_threshold_1_sd ,volume_time, 0.5)
        current_weak_signal_index = weak_signal_index(average_dff_normalized, signal_threshold_1_sd_weak  ,volume_time, 0.5)
        
        current_strong_PVA_behavior_state = get_behavior_state_of_strong_PVA(current_strong_PVA_index,behavior_state_frame_index)
        
        current_strong_signal_bump_shape = get_bump_shape_at_strong_signal(df_dff_in_ROI_normalized_shifted_peak_centered_8_ROI,current_strong_signal_index,8)
        
        current_weak_signal_bump_shape = get_bump_shape_at_strong_signal(df_dff_in_ROI_normalized_shifted_peak_centered_8_ROI,current_weak_signal_index,8)
        
        current_strong_signal_PVA = get_PVA_at_strong_signal(PVA_Angle, current_strong_signal_index)
       
        
        # Check if 'strong_PVA_chunk_behavior_state' exists and is not empty
        if len(output_pooled_dictionary['strong_PVA_chunk_behavior_state']) == 0:
            output_pooled_dictionary['strong_PVA_chunk_behavior_state'] = current_strong_PVA_behavior_state
        else:
            output_pooled_dictionary['strong_PVA_chunk_behavior_state'] = np.concatenate(
        (output_pooled_dictionary['strong_PVA_chunk_behavior_state'], current_strong_PVA_behavior_state))
            
        
        output_pooled_dictionary['strong_PVA_chunk_pooled'] = output_pooled_dictionary['strong_PVA_chunk_pooled'] + current_strong_PVA_chunk
    
        if dual_imaging == 1:
            current_strong_PVA_chunk_red = strong_PVA_duration(PVA_strength_red, strength_threhold ,volume_time, min_window_s)
            output_pooled_dictionary['strong_PVA_chunk_pooled_red'] = output_pooled_dictionary['strong_PVA_chunk_pooled_red'] + current_strong_PVA_chunk_red
        
        
        # 3.5: Get Bump_width information
        #raw_data_16_ROI_z = zscore(raw_data_16_ROI, axis=1)
        #dff_16_roi_z = zscore(dff_16_roi, axis=1)
        Bump_width = calculateBumpWidth_v1(dff_16_roi, 16)
        
        if dual_imaging == 1:
            Bump_width_red = calculateBumpWidth_v1(dff_normalized_8_roi_red, 8)
        
        
        # 3.6: Get radian offset information
        radian_offset_current = offset_calculation(Wrapped_heading, PVA_Radian, IfRadian = True)
        radian_offset_no_stopping_period_current = []
        for i in range(len (radian_offset_current)):
            if np.abs(Angular_velocity[i]) >= 0.26:
                radian_offset_no_stopping_period_current.append(radian_offset_current[i])
        circular_va = circvar(radian_offset_no_stopping_period_current,high=np.pi, low= -np.pi)
        output_pooled_dictionary['circular_variance'].append(circular_va)
        if dual_imaging == 1:
            radian_offset_red_current = offset_calculation(Wrapped_heading, PVA_Radian_red, IfRadian = True)
            circular_va_red = circvar(radian_offset_red_current,high=np.pi, low= -np.pi)
            output_pooled_dictionary['circular_variance_red'].append(circular_va_red)
            
            
            
        # 3.7: Get Circular Variance in a sliding window fashion 
        #5s_ window
        window_for_cir_var = 5
        sliding_win_cir_var_array = sliding_window_circular_variance(radian_offset_current,behavior_state_frame_index,PVA_strength, window_for_cir_var,volume_time,step_size_frames=5)
        output_pooled_dictionary['circular_variance_slide_window_during_walking'].append(sliding_win_cir_var_array)
        
        
            
            
        
        
        # 3.8 Get Bump dynamics during the stopping period
        bump_amplitude_stopping_duration = Bump_amplitude_stopping_duration
        minimum_frame_length = int(np.ceil(bump_amplitude_stopping_duration/volume_time))
        #200ms bin size
        bump_amplitude_stopping_bin_size = 0.2
    
        #PVA_strength_gaussian_smooth = gaussian_filter1d(PVA_strength, sigma =3)
        #Bump_width_gaussian_smooth = gaussian_filter1d(Bump_width, sigma =3)
                
        stable_PVA_threshold = 67.5
    
        #Get qualified stopping period for decay analysis 
        qualified_stopping_index_and_length, qualified_stable_PVA_index = find_qualified_stopping_period (volume_time =volume_time,stopping_array=persistence_stop_index_and_length,PVAinAngle=PVA_Angle,minimum_frame_length=int(np.ceil(bump_amplitude_stopping_duration/volume_time)),stable_PVA_threshold=stable_PVA_threshold)
        
        if dual_imaging == 1:
            qualified_stopping_index_and_length_red, qualified_stable_PVA_index_red = find_qualified_stopping_period (volume_time =volume_time,stopping_array=persistence_stop_index_and_length,PVAinAngle=PVA_Angle_red,minimum_frame_length=int(np.ceil(bump_amplitude_stopping_duration/volume_time)),stable_PVA_threshold=stable_PVA_threshold)
        
        
        
        active_period_before_len =1
    
        stopping_period_PVA_strength_current = stopping_period_signal_decay(volume_time, bump_amplitude_stopping_duration,bump_amplitude_stopping_bin_size = 0.2, signal_array =PVA_strength, qualified_stop_array=qualified_stopping_index_and_length,active_period_before_len=active_period_before_len)
        bump_amplitude_v3_stopping_current = stopping_period_signal_decay(volume_time, bump_amplitude_stopping_duration,bump_amplitude_stopping_bin_size = 0.2, signal_array =Bump_amplitude_V3, qualified_stop_array=qualified_stopping_index_and_length,active_period_before_len=active_period_before_len)
        bump_amplitude_v4_stopping_current = stopping_period_signal_decay(volume_time, bump_amplitude_stopping_duration,bump_amplitude_stopping_bin_size = 0.2, signal_array =Bump_amplitude_V4, qualified_stop_array=qualified_stopping_index_and_length,active_period_before_len=active_period_before_len)
        bump_width_stopping_current = stopping_period_signal_decay(volume_time, bump_amplitude_stopping_duration,bump_amplitude_stopping_bin_size = 0.2, signal_array =Bump_width, qualified_stop_array=qualified_stopping_index_and_length,active_period_before_len=active_period_before_len)
        Angular_speed_current = stopping_period_signal_decay(volume_time, bump_amplitude_stopping_duration,bump_amplitude_stopping_bin_size = 0.2, signal_array = Angular_speed_degrees, qualified_stop_array=qualified_stopping_index_and_length,active_period_before_len=active_period_before_len)
        
    
        stopping_period_bump_amp_v3_current = pd.DataFrame(bump_amplitude_v3_stopping_current.transpose())
        stopping_period_bump_amp_v4_current = pd.DataFrame(bump_amplitude_v4_stopping_current.transpose())
        stopping_period_bump_width_current = pd.DataFrame(bump_width_stopping_current.transpose())
        stopping_period_PVA_strength_current = pd.DataFrame(stopping_period_PVA_strength_current.transpose())
        Angular_speed_current = pd.DataFrame(Angular_speed_current.transpose())
        
        
        if dual_imaging == 1:
          
            stopping_period_PVA_strength_current_red = stopping_period_signal_decay(volume_time, bump_amplitude_stopping_duration,bump_amplitude_stopping_bin_size = 0.2, signal_array =PVA_strength_red, qualified_stop_array=qualified_stopping_index_and_length_red,active_period_before_len=active_period_before_len)
            bump_amplitude_v3_stopping_current_red = stopping_period_signal_decay(volume_time, bump_amplitude_stopping_duration,bump_amplitude_stopping_bin_size = 0.2, signal_array =Bump_amplitude_V3_red, qualified_stop_array=qualified_stopping_index_and_length_red,active_period_before_len=active_period_before_len)
            bump_amplitude_v4_stopping_current_red = stopping_period_signal_decay(volume_time, bump_amplitude_stopping_duration,bump_amplitude_stopping_bin_size = 0.2, signal_array =Bump_amplitude_V4_red, qualified_stop_array=qualified_stopping_index_and_length_red,active_period_before_len=active_period_before_len)
            bump_width_stopping_current_red = stopping_period_signal_decay(volume_time, bump_amplitude_stopping_duration,bump_amplitude_stopping_bin_size = 0.2, signal_array =Bump_width_red, qualified_stop_array=qualified_stopping_index_and_length_red,active_period_before_len=active_period_before_len)
            
    
            stopping_period_bump_amp_v3_current_red = pd.DataFrame(bump_amplitude_v3_stopping_current_red.transpose())
            stopping_period_bump_amp_v4_current_red = pd.DataFrame(bump_amplitude_v4_stopping_current_red.transpose())
            stopping_period_bump_width_current_red = pd.DataFrame(bump_width_stopping_current_red.transpose())
            stopping_period_PVA_strength_current_red = pd.DataFrame(stopping_period_PVA_strength_current_red.transpose())
    
        
        
        
        
        
        
        
        
        #3.9 Data storage
        
        
        #Store the flyinformation if wants to fly-by-fly analysis later on
        single_trial_info = single_df.split("-")
        flytrial = [single_trial_info[0], single_trial_info[1]]
        output_pooled_dictionary['output_flytrial'].append(single_trial_info[0])
        #just in case some trials do not have stopping period
        for i in range(stopping_period_bump_amp_v4_current.shape[1]):
            output_pooled_dictionary['output_flytrial_for_qualified_stop'].append(single_trial_info[0])
        
        
        
        #Store bump dynamic data during the stop
        if count == 0:
            output_pooled_dictionary['output_PVA_strength_pooled'] =  stopping_period_PVA_strength_current
            output_pooled_dictionary['output_bump_amplitude_V3_pooled'] = stopping_period_bump_amp_v3_current
            output_pooled_dictionary['output_bump_amplitude_V4_pooled'] = stopping_period_bump_amp_v4_current
            output_pooled_dictionary['output_bump_width_at_stop_pooled'] = stopping_period_bump_width_current
            output_pooled_dictionary['output_Angular_Speed_pooled'] = Angular_speed_current
            output_pooled_dictionary['Average_bump_shape'] = pd.DataFrame(average_bump_shape)
            output_pooled_dictionary['Average_bump_shape_strong_signal'] =  pd.DataFrame(current_strong_signal_bump_shape)
            output_pooled_dictionary['Average_bump_shape_weak_signal'] =  pd.DataFrame(current_weak_signal_bump_shape)
        else:
            output_pooled_dictionary['output_PVA_strength_pooled'] = pd.concat([ output_pooled_dictionary['output_PVA_strength_pooled'],stopping_period_PVA_strength_current],ignore_index=True, axis =1)
            output_pooled_dictionary['output_bump_amplitude_V3_pooled'] = pd.concat([output_pooled_dictionary['output_bump_amplitude_V3_pooled'],stopping_period_bump_amp_v3_current],ignore_index=True, axis =1)
            output_pooled_dictionary['output_bump_amplitude_V4_pooled'] = pd.concat([output_pooled_dictionary['output_bump_amplitude_V4_pooled'],stopping_period_bump_amp_v4_current],ignore_index=True, axis =1)
            output_pooled_dictionary['output_bump_width_at_stop_pooled'] = pd.concat([ output_pooled_dictionary['output_bump_width_at_stop_pooled'],stopping_period_bump_width_current],ignore_index=True, axis =1)        
            output_pooled_dictionary['output_Angular_Speed_pooled'] = pd.concat([output_pooled_dictionary['output_Angular_Speed_pooled'],Angular_speed_current],ignore_index=True, axis =1)      
            output_pooled_dictionary['Average_bump_shape'] = pd.concat([output_pooled_dictionary['Average_bump_shape'] ,pd.DataFrame(average_bump_shape)],ignore_index=True, axis =1) 
            output_pooled_dictionary['Average_bump_shape_strong_signal'] = pd.concat([output_pooled_dictionary['Average_bump_shape_strong_signal'] ,pd.DataFrame(current_strong_signal_bump_shape)],ignore_index=True, axis =1) 
            output_pooled_dictionary['Average_bump_shape_weak_signal'] = pd.concat([output_pooled_dictionary['Average_bump_shape_weak_signal'] ,pd.DataFrame(current_weak_signal_bump_shape)],ignore_index=True, axis =1) 
    
        for i in range(len(qualified_stable_PVA_index)):
            output_pooled_dictionary['output_stable_PVA_index_pooled'].append(int(qualified_stable_PVA_index[i]))
    
    
        if dual_imaging == 1:
            if count == 0:
                output_pooled_dictionary['output_PVA_strength_pooled_red'] =  stopping_period_PVA_strength_current_red
                output_pooled_dictionary['output_bump_amplitude_V3_pooled_red'] = stopping_period_bump_amp_v3_current_red
                output_pooled_dictionary['output_bump_amplitude_V4_pooled_red'] = stopping_period_bump_amp_v4_current_red
                output_pooled_dictionary['output_bump_width_pooled_red'] = stopping_period_bump_width_current_red
            else:
                output_pooled_dictionary['output_PVA_strength_pooled_red'] = pd.concat([ output_pooled_dictionary['output_PVA_strength_pooled_red'],stopping_period_PVA_strength_current_red],ignore_index=True, axis =1)
                output_pooled_dictionary['output_bump_amplitude_V3_pooled_red'] = pd.concat([output_pooled_dictionary['output_bump_amplitude_V3_pooled_red'],stopping_period_bump_amp_v3_current_red],ignore_index=True, axis =1)
                output_pooled_dictionary['output_bump_amplitude_V4_pooled_red'] = pd.concat([output_pooled_dictionary['output_bump_amplitude_V4_pooled_red'],stopping_period_bump_amp_v4_current_red],ignore_index=True, axis =1)
                output_pooled_dictionary['output_bump_width_pooled_red'] = pd.concat([ output_pooled_dictionary['output_bump_width_pooled_red'],stopping_period_bump_width_current_red],ignore_index=True, axis =1)          
    
            for i in range(len(qualified_stable_PVA_index_red)):
                output_pooled_dictionary['output_stable_PVA_index_pooled_red'].append(int(qualified_stable_PVA_index_red[i]))
        
        
        
        # Store data for different stop point comparison
        second_wise_bump_drift_rate_at_stop_strength_threshold,second_wise_bump_position_at_stop_strength_threshold,second_wise_bump_drift_rate_at_stop_strength_threshold_abs=second_wise_PVA_position_and_change_rate_at_stop_strength_threshold(
    persistence_stop_index_and_length, PVA_Angle, PVA_strength, volume_time, strength_threshold =0.1
)
        second_wise_bump_drift_before_stop_threashold =PVA_velocity_right_before_stopPeriod_with_strength_threshold(PVA_Angle,PVA_strength,volume_time,persistence_stop_index_and_length, duration=5,strength_threshold=0.1)
        second_wise_bump_drift_before_stop = PVA_velocity_right_before_stopPeriod(PVA_array=PVA_Angle, volume_time=volume_time, stop_index_array=persistence_stop_index_and_length, duration=5)
        second_wise_angular_velocity_before_stop = behavior_velocity_right_before_stopPeriod(velocity_array=Angular_velocity, volume_time=volume_time, stop_index_array=persistence_stop_index_and_length, duration=5)
        second_wise_PVA_position_before_stop = PVA_position_right_before_stopPeriod(PVA_array=PVA_Angle, volume_time=volume_time, stop_index_array=persistence_stop_index_and_length, duration=5)
        Angular_speed_at_stop, second_wise_angular_velocity_during_stop = angular_speed_during_stopPeriod(stop_index_array = persistence_stop_index_and_length,angular_speed_array=Angular_speed_degrees,angular_velocity_array=Angular_velocity,volume_time=volume_time)
        
        PVA_angle_at_stop,second_wise_bump_drift,second_wise_dwell_difference,second_wise_absolute_bump_drift,second_wise_PVA_position = PVA_during_stopPeriod(stop_index_array = persistence_stop_index_and_length, PVA_array =PVA_Angle, volume_time=volume_time)
        PVA_strength_at_stop,second_wise_PVA_strength =             PVA_strength_during_stopPeriod(stop_index_array=persistence_stop_index_and_length,PVA_strength_array=PVA_strength, volume_time=volume_time)
        bump_shape_at_stop_current, bump_shape_at_5s_after_stop_current, bump_shape_at_10s_after_stop_current, bump_shape_at_20s_after_stop_current,bump_shape_at_stop_more_than_10s_trial_only_current = Bump_shape_during_stopPeriod(persistence_stop_index_and_length,df_dff_in_ROI_normalized_shifted_peak_centered_8_ROI,volume_time,8)
        max_F_at_stop_current, max_F_at_stop_10s_later= max_raw_F_during_stopPeriod(persistence_stop_index_and_length,raw_F_max,volume_time)
        output_pooled_dictionary['max_Raw_F_at_stop'].append(max_F_at_stop_current)
        output_pooled_dictionary['max_Raw_F_at_10s_after_stop'].append(max_F_at_stop_10s_later)

            
        if count == 0:
            output_pooled_dictionary['Bump_shape_at_stop'] = pd.DataFrame(bump_shape_at_stop_current)
            output_pooled_dictionary['Bump_shape_at_5s_after_stop'] = pd.DataFrame(bump_shape_at_5s_after_stop_current)
            output_pooled_dictionary['Bump_shape_at_10s_after_stop'] = pd.DataFrame(bump_shape_at_10s_after_stop_current)
            output_pooled_dictionary['Bump_shape_at_20s_after_stop'] = pd.DataFrame(bump_shape_at_20s_after_stop_current)
            output_pooled_dictionary['Bump_shape_at_stop_more_than_10s_trial_only'] = pd.DataFrame(bump_shape_at_stop_more_than_10s_trial_only_current)
        else:
            output_pooled_dictionary['Bump_shape_at_stop'] = pd.concat([output_pooled_dictionary['Bump_shape_at_stop'] ,pd.DataFrame(bump_shape_at_stop_current)],ignore_index=True, axis =1) 
            output_pooled_dictionary['Bump_shape_at_5s_after_stop'] = pd.concat([output_pooled_dictionary['Bump_shape_at_5s_after_stop'] ,pd.DataFrame(bump_shape_at_5s_after_stop_current)],ignore_index=True, axis =1) 
            output_pooled_dictionary['Bump_shape_at_10s_after_stop'] = pd.concat([output_pooled_dictionary['Bump_shape_at_10s_after_stop'] ,pd.DataFrame(bump_shape_at_10s_after_stop_current)],ignore_index=True, axis =1) 
            output_pooled_dictionary['Bump_shape_at_20s_after_stop'] = pd.concat([output_pooled_dictionary['Bump_shape_at_20s_after_stop'] ,pd.DataFrame(bump_shape_at_20s_after_stop_current)],ignore_index=True, axis =1) 
            output_pooled_dictionary['Bump_shape_at_stop_more_than_10s_trial_only'] = pd.concat([output_pooled_dictionary['Bump_shape_at_stop_more_than_10s_trial_only'] ,pd.DataFrame(bump_shape_at_stop_more_than_10s_trial_only_current)],ignore_index=True, axis =1) 
        
        
       
    
        #Store second wise drift data
        # Assuming single_trial_info[0] is your current key
        key = single_trial_info[0]
        key_for_heading_PVA_offset = tuple(flytrial)
        
        # Check if the key already exists in the dictionary
        if key not in output_pooled_dictionary['output_second_wise_bump_drift']:
            output_pooled_dictionary['output_second_wise_bump_drift'][key] = []
            output_pooled_dictionary['output_second_wise_bump_drift_at_stop_thresholded'][key] = []
            output_pooled_dictionary['output_second_wise_bump_drift_at_stop_thresholded_absolute'][key] = []
            output_pooled_dictionary['output_second_wise_bump_drift_right_before_stop'][key] = []
            output_pooled_dictionary['output_second_wise_bump_drift_right_before_stop_thresholded'][key] = []
            output_pooled_dictionary['output_second_wise_angular_velocity_right_before_stop'][key] = []
            output_pooled_dictionary['output_second_wise_angular_velocity_during_stop'][key] = []
            output_pooled_dictionary['output_second_wise_PVA_position_right_before_stop'][key] = []
            output_pooled_dictionary['output_second_wise_PVA_position_during_stop'][key] = []
            output_pooled_dictionary['output_second_wise_PVA_position_during_stop_thresholded'][key] = []
            output_pooled_dictionary['output_second_wise_absolute_bump_drift'][key] = []
            output_pooled_dictionary['output_second_wise_bump_dwell_difference'][key] = []            
            output_pooled_dictionary['output_second_wise_PVA_strength'][key]  = []
            output_pooled_dictionary['PVA_at_strong_signal'][key]  = []
            output_pooled_dictionary['dff_8_roi_at_stop'][key]  = []
            output_pooled_dictionary['PVA_strength_frame_wise_at_stop'][key]  = []
            output_pooled_dictionary['PVA_Angle_frame_wise_at_stop'][key]  = []
        output_pooled_dictionary['output_second_wise_bump_drift'][single_trial_info[0]].append(second_wise_bump_drift)
        output_pooled_dictionary['output_second_wise_absolute_bump_drift'][single_trial_info[0]].append(second_wise_absolute_bump_drift)
        output_pooled_dictionary['output_second_wise_bump_drift_at_stop_thresholded'][single_trial_info[0]].append(second_wise_bump_drift_rate_at_stop_strength_threshold)
        output_pooled_dictionary['output_second_wise_bump_drift_at_stop_thresholded_absolute'][single_trial_info[0]].append(second_wise_bump_drift_rate_at_stop_strength_threshold_abs)
        output_pooled_dictionary['output_second_wise_bump_drift_right_before_stop'][single_trial_info[0]].append(second_wise_bump_drift_before_stop)
        output_pooled_dictionary['output_second_wise_bump_drift_right_before_stop_thresholded'][single_trial_info[0]].append(second_wise_bump_drift_before_stop_threashold)
        output_pooled_dictionary['output_second_wise_angular_velocity_right_before_stop'][single_trial_info[0]].append(second_wise_angular_velocity_before_stop)
        output_pooled_dictionary['output_second_wise_angular_velocity_during_stop'][single_trial_info[0]].append(second_wise_angular_velocity_during_stop)
        output_pooled_dictionary['output_second_wise_PVA_position_during_stop'][single_trial_info[0]].append(second_wise_PVA_position)
        output_pooled_dictionary['output_second_wise_PVA_position_during_stop_thresholded'][single_trial_info[0]].append(second_wise_bump_position_at_stop_strength_threshold)
        output_pooled_dictionary['output_second_wise_PVA_position_right_before_stop'][single_trial_info[0]].append(second_wise_PVA_position_before_stop)
        output_pooled_dictionary['output_second_wise_bump_dwell_difference'][single_trial_info[0]].append(second_wise_dwell_difference)
        output_pooled_dictionary['output_second_wise_PVA_strength'][single_trial_info[0]].append(second_wise_PVA_strength)
        output_pooled_dictionary['PVA_at_strong_signal'][single_trial_info[0]].append(current_strong_signal_PVA)
        output_pooled_dictionary['dff_8_roi_at_stop'][single_trial_info[0]].append(dff_8_roi_at_stop_current)
        output_pooled_dictionary['PVA_strength_frame_wise_at_stop'][single_trial_info[0]].append(PVA_strength_at_stop_frame_wise_current)
        output_pooled_dictionary['PVA_Angle_frame_wise_at_stop'][single_trial_info[0]].append(PVA_Angle_at_stop_frame_wise_current)
        
        
        
        PVA_angle_at_stop.insert(0,'FlyTrial',"-".join(flytrial))
        PVA_angle_at_stop.insert(1,'Genotype',genotype)
        PVA_angle_at_stop.insert(2,'TrialType',trial_condition)
        Forward_speed_at_stop = forwrad_speed_during_stopPeriod(stop_index_array=persistence_stop_index_and_length,forward_speed_array=Forward_speed_degrees, volume_time=volume_time)
        Bump_amplitude_at_stop = bump_amplitude_during_stopPeriod(stop_index_array=persistence_stop_index_and_length,bump_amplitude_array=Bump_amplitude_V4, volume_time=volume_time)
        output_df = pd.concat([PVA_angle_at_stop,Forward_speed_at_stop,Bump_amplitude_at_stop, PVA_strength_at_stop,Angular_speed_at_stop], axis=1)
    
        if count == 0:
            output_pooled_dictionary['output_df_pooled'] = output_df
        else:
            output_pooled_dictionary['output_df_pooled'] = pd.concat([output_pooled_dictionary['output_df_pooled'],output_df], ignore_index=True)
        
        
        
        if  key_for_heading_PVA_offset not in output_pooled_dictionary['output_PVA_heading_offset']:
            output_pooled_dictionary['output_PVA_heading_offset'][key_for_heading_PVA_offset] = []
        output_pooled_dictionary['output_PVA_heading_offset'][key_for_heading_PVA_offset].append(radian_offset_current)
        
        
        
        if dual_imaging == 1:
            PVA_angle_at_stop_red = PVA_during_stopPeriod(stop_index_array = persistence_stop_index_and_length, PVA_array =PVA_Angle_red, volume_time=volume_time)
            PVA_angle_at_stop_red.insert(0,'FlyTrial',"-".join(flytrial))
            PVA_angle_at_stop_red.insert(1,'Genotype',genotype)
            PVA_angle_at_stop_red.insert(2,'TrialType',trial_condition)
            Bump_amplitude_at_stop_red = bump_amplitude_during_stopPeriod(stop_index_array=persistence_stop_index_and_length,bump_amplitude_array=Bump_amplitude_V4_red, volume_time=volume_time)
            PVA_strength_at_stop_red =             PVA_strength_during_stopPeriod(stop_index_array=persistence_stop_index_and_length,PVA_strength_array=PVA_strength_red, volume_time=volume_time)
            output_df_red = pd.concat([PVA_angle_at_stop_red,Forward_speed_at_stop,Bump_amplitude_at_stop_red, PVA_strength_at_stop_red], axis=1)
            
            if count == 0:
                output_pooled_dictionary['output_df_pooled_red'] = output_df_red
            else:
                output_pooled_dictionary['output_df_pooled_red'] = pd.concat([ output_pooled_dictionary['output_df_pooled_red'],output_df_red], ignore_index=True)
            
        
       
        
    
        count = count + 1
        
        
        
        
    return output_pooled_dictionary