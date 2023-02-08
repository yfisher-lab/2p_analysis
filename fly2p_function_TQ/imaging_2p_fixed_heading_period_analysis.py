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
    middle_points = []
    frame_100ms_before = int(np.ceil(0.1/volume_time))
    frame_500ms_before = int(np.ceil(0.5/volume_time))
    frame_2s_before = int(np.ceil(2/volume_time))
    frame_3s_after_stop = int(np.ceil(3/volume_time))
    frame_5s_after_stop = int(np.ceil(5/volume_time))
    frame_10s_after_stop = int(np.ceil(10/volume_time))
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
    middle_points = []
    frame_100ms_before = int(np.ceil(0.1/volume_time))
    frame_500ms_before = int(np.ceil(0.5/volume_time))
    frame_2s_before = int(np.ceil(2/volume_time))
    frame_3s_after_stop = int(np.ceil(3/volume_time))
    frame_5s_after_stop = int(np.ceil(5/volume_time))
    frame_10s_after_stop = int(np.ceil(10/volume_time))
    
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
    
    
    df_forward_speed_array_stop ['forward_speed_before_stop'] = stop_points
    df_forward_speed_array_stop ['forward_speed_at_restart'] = restart_points
    df_forward_speed_array_stop ['forward_speed_100ms_before_restart'] = restart_points_100ms_before
    df_forward_speed_array_stop ['forward_speed_500ms_before_restart'] = restart_points_500ms_before
    df_forward_speed_array_stop ['forward_speed_in_middle'] = middle_points
    df_forward_speed_array_stop ['forward_speed_2s_before_restart'] = restart_points_2s_before
    df_forward_speed_array_stop ['forward_speed_3s_after_stop'] = stop_after_3s
    df_forward_speed_array_stop ['forward_speed_5s_after_stop'] = stop_after_5s
    df_forward_speed_array_stop ['forward_speed_10s_after_stop'] = stop_after_10s
    
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
    middle_points = []
    frame_100ms_before = int(np.ceil(0.1/volume_time))
    frame_500ms_before = int(np.ceil(0.5/volume_time))
    frame_2s_before = int(np.ceil(2/volume_time))
    frame_3s_after_stop = int(np.ceil(3/volume_time))
    frame_5s_after_stop = int(np.ceil(5/volume_time))
    frame_10s_after_stop = int(np.ceil(10/volume_time))
    
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
    
    
    df_bump_amplitude_array_stop ['bump_amplitude_before_stop'] = stop_points
    df_bump_amplitude_array_stop ['bump_amplitude_at_restart'] = restart_points
    df_bump_amplitude_array_stop ['bump_amplitude_100ms_before_restart'] = restart_points_100ms_before
    df_bump_amplitude_array_stop ['bump_amplitude_500ms_before_restart'] = restart_points_500ms_before
    df_bump_amplitude_array_stop ['bump_amplitude_in_middle'] = middle_points
    df_bump_amplitude_array_stop ['bump_amplitude_2s_before_restart'] = restart_points_2s_before
    df_bump_amplitude_array_stop ['bump_amplitude_3s_after_stop'] = stop_after_3s
    df_bump_amplitude_array_stop ['bump_amplitude_5s_after_stop'] = stop_after_5s
    df_bump_amplitude_array_stop ['bump_amplitude_10s_after_stop'] = stop_after_10s
    
    return df_bump_amplitude_array_stop