import numpy as np
import pandas as pd
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.interpolate import UnivariateSpline
from matplotlib import pyplot as plt

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
        restart_points.append(PVA_array[stop_index_array[current_index,0]])
        stop_points.append(PVA_array[stop_index_array[current_index,0]-stop_index_array[current_index,1]+1])
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
    return df_PVA_stop 






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
    