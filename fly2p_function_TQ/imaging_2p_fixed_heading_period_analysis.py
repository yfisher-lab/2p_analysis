import numpy as np
import pandas as pd

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
    stop_after_35s = []
    middle_points = []
    frame_100ms_before = int(np.ceil(0.1/volume_time))
    frame_500ms_before = int(np.ceil(0.5/volume_time))
    frame_2s_before = int(np.ceil(2/volume_time))
    frame_3s_after_stop = int(np.ceil(3/volume_time))
    frame_5s_after_stop = int(np.ceil(5/volume_time))
    frame_10s_after_stop = int(np.ceil(10/volume_time))
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
    stop_after_35s = []
    middle_points = []
    frame_100ms_before = int(np.ceil(0.1/volume_time))
    frame_500ms_before = int(np.ceil(0.5/volume_time))
    frame_2s_before = int(np.ceil(2/volume_time))
    frame_3s_after_stop = int(np.ceil(3/volume_time))
    frame_5s_after_stop = int(np.ceil(5/volume_time))
    frame_10s_after_stop = int(np.ceil(10/volume_time))
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
    stop_after_35s = []
    middle_points = []
    frame_100ms_before = int(np.ceil(0.1/volume_time))
    frame_500ms_before = int(np.ceil(0.5/volume_time))
    frame_2s_before = int(np.ceil(2/volume_time))
    frame_3s_after_stop = int(np.ceil(3/volume_time))
    frame_5s_after_stop = int(np.ceil(5/volume_time))
    frame_10s_after_stop = int(np.ceil(10/volume_time))
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
    df_bump_amplitude_array_stop ['bump_amplitude_35s_after_stop'] = stop_after_35s
    
    return df_bump_amplitude_array_stop





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



def stopping_period_signal_decay(volume_time, bump_amplitude_stopping_duration,bump_amplitude_stopping_bin_size, minimum_frame_length, stopping_array, signal_array):
    bump_amplitude_stopping_bin_number =int(bump_amplitude_stopping_duration/bump_amplitude_stopping_bin_size)
    bins_amplitude = []
    for low in range  (0, int(0+100*bump_amplitude_stopping_bin_size*bump_amplitude_stopping_bin_number),int(100*bump_amplitude_stopping_bin_size)):
        bins_amplitude.append((low, int(low+bump_amplitude_stopping_bin_size*100)))
        
    
    #Find qualified stopping index (must meet the minimimun length of demand and must have 1s of active period before)
    persistence_stop_index_and_length_qualified_index = []
    for current_index in range(len(stopping_array)):
        start_index = [stopping_array[current_index,0]-stopping_array[current_index,1]+1][0]
        if start_index * volume_time -1 > 0:            
            if start_index + minimum_frame_length - 1 <= stopping_array[current_index,0]:
                persistence_stop_index_and_length_qualified_index.append(current_index)
    
    persistence_stop_index_and_length_qualified = np.zeros((len(persistence_stop_index_and_length_qualified_index),2))
    for current_index in range(len(persistence_stop_index_and_length_qualified_index)):
        persistence_stop_index_and_length_qualified[current_index,0] = stopping_array[persistence_stop_index_and_length_qualified_index[current_index],0]
        persistence_stop_index_and_length_qualified[current_index,1] = stopping_array[persistence_stop_index_and_length_qualified_index[current_index],1]
    persistence_stop_index_and_length_qualified = persistence_stop_index_and_length_qualified.astype(int)    
    bump_amplitude_stopping_current = np.zeros((len(persistence_stop_index_and_length_qualified),bump_amplitude_stopping_bin_number))
    
    
    
    #Binning all the qualified stoppinf period 
    for current_index in range(len(persistence_stop_index_and_length_qualified)):
        start_index = [persistence_stop_index_and_length_qualified[current_index,0]-persistence_stop_index_and_length_qualified[current_index,1]+1][0]
        temp_amplitude = 0
        temp_count = 0
        bin_index = 0
        for current_binduration_index in range(int(np.ceil(bump_amplitude_stopping_duration/volume_time))):
            if current_binduration_index == int(np.ceil(bump_amplitude_stopping_duration/volume_time)) - 1:
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