import numpy as np
import pandas as pd
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.interpolate import UnivariateSpline
from matplotlib import pyplot as plt
import os
from os.path import sep
import math
from scipy.signal import correlate
from fly2p_function_TQ.imaging_2p_preprocessing import low_pass_filter_TQ
from fly2p_function_TQ.imaging_2p_fictrac_imaging_alignment import moving_wrapped_plot_by_offset,fictrack_signal_decoding,offset_calculation 
from fly2p_function_TQ.imaging_2p_PVA_functions import calcualteBumpAmplitude, calcualteBumpAmplitude_V3, calculateBumpWidth_v1, PVA_radian_to_angle,calcualteBumpAmplitude_V4,PVAangleToRoi,strong_PVA_duration

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





def PVA_during_stopPeriod(stop_index_array,PVA_array, volume_time):
    #Initiate output dataframe and 
    df_PVA_stop = pd.DataFrame()
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
    stop_after_60s = []
    middle_points = []
    PVA_angle_drift_per_sec = []
    PVA_angle_drift_per_sec_second_wise = []
    PVA_angle_drift_per_sec_2 = []
    PVA_angle_drift_per_sec_2_second_wise = []
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
        else:
            stop_after_10s.append(np.NaN)
        if index_of_stop + frame_20s_after_stop - 1 <= stop_index_array[current_index,0]:
            stop_after_20s.append(PVA_array[index_of_stop + frame_20s_after_stop - 1])
        else:
            stop_after_20s.append(np.NaN)
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
        data_per_second = int(1 / volume_time)  # 1 second / 0.2 seconds per data point

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
        data_per_second_2 = int(1 / volume_time)  # 1 second / 0.2 seconds per data point

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
        
        
    
    df_PVA_stop['Period_duration'] = duration_stop
    df_PVA_stop['PVA_before_stop'] = stop_points
    df_PVA_stop['PVA_at_restart'] = restart_points
    df_PVA_stop['PVA_100ms_before_restart'] = restart_points_100ms_before
    df_PVA_stop['PVA_500ms_before_restart'] = restart_points_500ms_before
    df_PVA_stop['PVA_in_middle'] = middle_points
    df_PVA_stop['PVA_2s_before_restart'] = restart_points_2s_before
    df_PVA_stop['PVA_3s_after_stop'] = stop_after_3s
    df_PVA_stop['PVA_5s_after_stop'] = stop_after_5s
    df_PVA_stop['PVA_10s_after_stop'] = stop_after_10s
    df_PVA_stop['PVA_20s_after_stop'] = stop_after_20s
    df_PVA_stop['PVA_35s_after_stop'] = stop_after_35s
    df_PVA_stop['PVA_60s_after_stop'] = stop_after_60s
    df_PVA_stop['PVA_angle_drift_per_second']= PVA_angle_drift_per_sec
    df_PVA_stop['PVA_angle_drift_per_second_2']= PVA_angle_drift_per_sec_2
    return df_PVA_stop, PVA_angle_drift_per_sec_2_second_wise,  PVA_angle_drift_per_sec_second_wise





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
    return df_PVA_strength_stop

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
        average_signal = round((signal_df.mean(axis=1)[actual_stop_start_index]-signal_df.mean(axis=1)[end_of_decay_index])/signal_df.mean(axis=1)[0]*100,2)
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
        
        
        
def run_fixed_heading_period_analysis_across_trial(directory,dual_imaging,genotype,trial_condition,Bump_amplitude_stopping_duration):
    
    
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
    output_pooled_dictionary['mean_PVA_strength_per_trial_pooled'] = []
    output_pooled_dictionary['output_stable_PVA_index_pooled'] = []
    output_pooled_dictionary['circular_variance'] = []
    output_pooled_dictionary['Average_bump_shape'] = pd.DataFrame()
    output_pooled_dictionary['output_flytrial'] = []
    output_pooled_dictionary['output_flytrial_for_qualified_stop'] = []
    output_pooled_dictionary['output_second_wise_bump_drift'] = {}
    output_pooled_dictionary['output_second_wise_bump_dwell_difference'] = {}
    
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
        persistence_stop_index_and_length = find_stop_period_on_heading(head_velocity_array = Angular_velocity,degree_of_tolerance =15,shortest_stopFrame=int(np.ceil(3/volume_time)))
    
    
    
        #Average bump shape peak centered
        df_dff_in_ROI_normalized_shifted_peak_centered = np.zeros((len(dff_normalized_8_roi_shifted),       len(dff_normalized_8_roi_shifted[0])))

        shifted_by_all = []

        for i in range(len(dff_normalized_8_roi_shifted[0])):
            original_order = [0,1,2,3,4,5,6,7]
            current_peak = np.argmax(dff_normalized_8_roi_shifted[:,i])
            shift_by = current_peak - 3
            shifted_order = original_order[shift_by % len(original_order):] + original_order[:shift_by % len(original_order)]
            df_dff_in_ROI_normalized_shifted_peak_centered[:,i] = dff_normalized_8_roi_shifted[shifted_order,i]
            shifted_by_all.append(shift_by)
    
        average_bump_shape = np.mean(df_dff_in_ROI_normalized_shifted_peak_centered, axis=1)
        average_bump_shape =pd.DataFrame(average_bump_shape)
   
    
 

    
        
    
    
        ##Part3: get signal during the stopping period
    
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
    
        # 3.4: Get strong PVA chunk 
        strength_threhold = 0.6
        min_window_s = 2
        current_strong_PVA_chunk = strong_PVA_duration(PVA_strength, strength_threhold ,volume_time, min_window_s)
        output_pooled_dictionary['strong_PVA_chunk_pooled'] = output_pooled_dictionary['strong_PVA_chunk_pooled'] + current_strong_PVA_chunk
    
        if dual_imaging == 1:
            current_strong_PVA_chunk_red = strong_PVA_duration(PVA_strength_red, strength_threhold ,volume_time, min_window_s)
            output_pooled_dictionary['strong_PVA_chunk_pooled_red'] = output_pooled_dictionary['strong_PVA_chunk_pooled_red'] + current_strong_PVA_chunk_red
        
        
        # 3.5: Get Bump_width information
        Bump_width = calculateBumpWidth_v1(dff_normalized_8_roi, 8)
        
        if dual_imaging == 1:
            Bump_width_red = calculateBumpWidth_v1(dff_normalized_8_roi_red, 8)
        
        
        # 3.6: Get radian offset information
        radian_offset_current = offset_calculation(Wrapped_heading, PVA_Radian, IfRadian = True)
        from scipy.stats import circvar
        circular_va = circvar(radian_offset_current,high=np.pi, low= -np.pi)
        output_pooled_dictionary['circular_variance'].append(circular_va)
        if dual_imaging == 1:
            radian_offset_red_current = offset_calculation(Wrapped_heading, PVA_Radian_red, IfRadian = True)
            circular_va_red = circvar(radian_offset_red_current,high=np.pi, low= -np.pi)
            output_pooled_dictionary['circular_variance_red'].append(circular_va_red)
        
        
        # 3.7 Get Bump dynamics during the stopping period
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
    
        
        
        
        
        
        
        
        
        #3.8 Data storage
        
        
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
            output_pooled_dictionary['output_bump_width_pooled'] = stopping_period_bump_width_current
            output_pooled_dictionary['output_Angular_Speed_pooled'] = Angular_speed_current
            output_pooled_dictionary['Average_bump_shape'] = average_bump_shape
        else:
            output_pooled_dictionary['output_PVA_strength_pooled'] = pd.concat([ output_pooled_dictionary['output_PVA_strength_pooled'],stopping_period_PVA_strength_current],ignore_index=True, axis =1)
            output_pooled_dictionary['output_bump_amplitude_V3_pooled'] = pd.concat([output_pooled_dictionary['output_bump_amplitude_V3_pooled'],stopping_period_bump_amp_v3_current],ignore_index=True, axis =1)
            output_pooled_dictionary['output_bump_amplitude_V4_pooled'] = pd.concat([output_pooled_dictionary['output_bump_amplitude_V4_pooled'],stopping_period_bump_amp_v4_current],ignore_index=True, axis =1)
            output_pooled_dictionary['output_bump_width_pooled'] = pd.concat([ output_pooled_dictionary['output_bump_width_pooled'],stopping_period_bump_width_current],ignore_index=True, axis =1)        
            output_pooled_dictionary['output_Angular_Speed_pooled'] = pd.concat([output_pooled_dictionary['output_Angular_Speed_pooled'],Angular_speed_current],ignore_index=True, axis =1)      
            output_pooled_dictionary['Average_bump_shape'] = pd.concat([output_pooled_dictionary['Average_bump_shape'] ,average_bump_shape],ignore_index=True, axis =1) 
    
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
        PVA_angle_at_stop,second_wise_bump_drift,second_wise_dwell_difference = PVA_during_stopPeriod(stop_index_array = persistence_stop_index_and_length, PVA_array =PVA_Angle, volume_time=volume_time)
        
        
        
       
    
        #Store second wise drift data
        # Assuming single_trial_info[0] is your current key
        key = single_trial_info[0]
        # Check if the key already exists in the dictionary
        if key not in output_pooled_dictionary['output_second_wise_bump_drift']:
            output_pooled_dictionary['output_second_wise_bump_drift'][key] = []
            output_pooled_dictionary['output_second_wise_bump_dwell_difference'][single_trial_info[0]] = []            
            
        output_pooled_dictionary['output_second_wise_bump_drift'][single_trial_info[0]].append(second_wise_bump_drift)
        output_pooled_dictionary['output_second_wise_bump_dwell_difference'][single_trial_info[0]].append(second_wise_dwell_difference)
        
        
        
        PVA_angle_at_stop.insert(0,'FlyTrial',"-".join(flytrial))
        PVA_angle_at_stop.insert(1,'Genotype',genotype)
        PVA_angle_at_stop.insert(2,'TrialType',trial_condition)
        Forward_speed_at_stop = forwrad_speed_during_stopPeriod(stop_index_array=persistence_stop_index_and_length,forward_speed_array=Forward_speed_degrees, volume_time=volume_time)
        Bump_amplitude_at_stop = bump_amplitude_during_stopPeriod(stop_index_array=persistence_stop_index_and_length,bump_amplitude_array=Bump_amplitude_V4, volume_time=volume_time)
        PVA_strength_at_stop =             PVA_strength_during_stopPeriod(stop_index_array=persistence_stop_index_and_length,PVA_strength_array=PVA_strength, volume_time=volume_time)
        output_df = pd.concat([PVA_angle_at_stop,Forward_speed_at_stop,Bump_amplitude_at_stop, PVA_strength_at_stop], axis=1)
    
        if count == 0:
            output_pooled_dictionary['output_df_pooled'] = output_df
        else:
            output_pooled_dictionary['output_df_pooled'] = pd.concat([output_pooled_dictionary['output_df_pooled'],output_df], ignore_index=True)
        
        
        
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