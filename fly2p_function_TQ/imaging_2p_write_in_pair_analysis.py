import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
from os.path import sep
from fly2p_function_TQ.imaging_2p_fictrac_imaging_alignment import fictrack_signal_decoding
from fly2p_function_TQ.imaging_2p_fixed_heading_period_analysis import find_stop_period_on_heading
from fly2p_function_TQ.imaging_2p_PVA_functions import calcualteBumpAmplitude, calcualteBumpAmplitude_V3
from scipy.stats import circvar, circmean
from scipy.ndimage import gaussian_filter1d
from scipy.stats import zscore

def run_write_in_pair_analysis_across_trial(directory,directory_stimulation_frame,target_string):
    # Part 1: Create and add each DataFrame to the dictionary
    count = 0
    output_pooled_dictionary = {}
    output_pooled_dictionary['output_PVA_radian_pooled'] = pd.DataFrame()
    output_pooled_dictionary['output_bar_PVA_offset_pooled'] = pd.DataFrame()
    output_pooled_dictionary['average_Bump_amplitude_after_1st_stim'] = pd.DataFrame()
    output_pooled_dictionary['average_PVA_after_1st_stim'] = pd.DataFrame()
    output_pooled_dictionary['average_PVA_after_2nd_stim'] = pd.DataFrame()
    output_pooled_dictionary['average_strength_after_1st_stim'] = pd.DataFrame()
    output_pooled_dictionary['average_strength_after_2nd_stim'] = pd.DataFrame()
    output_pooled_dictionary['200ms_bin_PVA_after_1st_stim'] = pd.DataFrame()
    output_pooled_dictionary['200ms_bin_PVA_after_2nd_stim'] = pd.DataFrame()
    output_pooled_dictionary['200ms_bin_strength_after_1st_stim'] = pd.DataFrame()
    output_pooled_dictionary['200ms_bin_strength_after_2nd_stim'] = pd.DataFrame()
    output_pooled_dictionary['stop_stamp_after_2nd_stim'] = pd.DataFrame()
    output_pooled_dictionary['stop_s_before_stimulation'] = []
    output_pooled_dictionary['Angular_speed'] = pd.DataFrame()
    output_pooled_dictionary['Angular_speed_entire_trial'] = pd.DataFrame()
    output_pooled_dictionary['Bump_speed_entire_trial'] = pd.DataFrame()
    output_pooled_dictionary['Bump_amplitude_at_stop'] = []
    output_pooled_dictionary['Bump_amplitude_at_turn'] = []
    output_pooled_dictionary['Forward_speed'] = pd.DataFrame()
    output_pooled_dictionary['output_flytrial'] = []
    output_pooled_dictionary['volume_time'] = []
    output_pooled_dictionary['Angular_speed_during_offset_return'] = pd.DataFrame()
    #Follow the standard of Ki, et al., 2017
    output_pooled_dictionary['bump_jump_flow_index'] = pd.DataFrame()
    output_pooled_dictionary['sec_bump_move'] = []
    output_pooled_dictionary['delta7_bump_amplitude_at_bump_move'] = []
    output_pooled_dictionary['Angular_speed_at_bump_move'] = []
    
    
    
    
    #Part 2:import data
    # import data
    for single_df in os.listdir(directory):
        if target_string in single_df:
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
            PVA_strength_z = zscore(PVA_strength)
            Bump_amplitude = current_file['Bump_amplitude'].values
            Unwrapped_heading = current_file['Unwrapped_Bar_Position/Heading'].values
            Wrapped_heading = current_file['Bar_Position/Heading'].values 
            
        
            output_pooled_dictionary['volume_time'] = volume_time
        
        
            integrated_x_unwrapped =  fictrack_signal_decoding(integrated_x,time_array_imaging, 10, already_radian = True)
            Forward_velocity = np.gradient(integrated_x_unwrapped)/volume_time
            Forward_speed_radian = np.abs(Forward_velocity)
            Forward_speed_degrees =Forward_speed_radian * 180/np.pi
            Angular_velocity_degrees =  Angular_velocity * 180/np.pi
            Angular_speed_degrees =  np.abs(Angular_velocity) * 180/np.pi
            PVA_unwrapped = fictrack_signal_decoding(PVA_Radian, time_array_imaging, 10, already_radian = True)
            Bump_speed_degrees =  (np.gradient(PVA_unwrapped)/volume_time) * 180/np.pi
            dff_normalized_8_roi = np.array([current_file[f'dFF_Roi_{i}'] for i in range(1, 9)]).T
            Bump_amplitude_V3, Bump_amplitude_V3_opposite = calcualteBumpAmplitude_V3(dff_normalized_8_roi,PVA_Radian)
            
            
            
            #Get persistence period
            persistence_stop_index_and_length = find_stop_period_on_heading(head_velocity_array = Angular_velocity,degree_of_tolerance =15,shortest_stopFrame=int(np.ceil(3/volume_time)))
            
            
            #Store the flyinformation if wants to fly-by-fly analysis later on and to align
            single_trial_info = single_df.split("-")
            flytrial = [single_trial_info[0], single_trial_info[1]]
            for i in range(10):
                concatenated_string = f"{single_trial_info[0]}-{single_trial_info[1]}-{i+1}"
                output_pooled_dictionary['output_flytrial'].append(concatenated_string )
            
            count = count + 1
            
            
            #Get stimulation start/end frame data 
            current_stimulation_start_frame = None
            current_stimulation_end_frame = None
            concatenated_string_2 = f"{single_trial_info[0]}-{single_trial_info[1]}"
            for single_jf in os.listdir(directory_stimulation_frame):  
                if concatenated_string_2  in single_jf:
                    file_path = os.path.join(directory_stimulation_frame, single_jf)
                    current_stimulation_frame = pd.read_csv(file_path)
                    break
           
            current_stimulation_start_frame = current_stimulation_frame['stimulation_start_frame'].values
            current_stimulation_end_frame = current_stimulation_frame['stimulation_end_frame'].values
            
            current_stimulation_end_frame_first_stim = current_stimulation_end_frame[np.arange(0, len(current_stimulation_end_frame), 2)]
            current_stimulation_end_frame_second_stim = current_stimulation_end_frame[np.arange(1, len(current_stimulation_end_frame), 2)]
            
            current_stimulation_start_frame_first_stim = current_stimulation_start_frame[np.arange(0, len(current_stimulation_end_frame), 2)]
            current_stimulation_start_frame_second_stim = current_stimulation_start_frame[np.arange(1, len(current_stimulation_end_frame), 2)]
            
            
            #Create a index array that indicates whether the current frame is stop/active stop index = 0, active index =1
            behavior_state_frame_index = np.ones(len(PVA_Radian))
            for current_index in range(len(persistence_stop_index_and_length)):
                start_index = persistence_stop_index_and_length[current_index,0]-persistence_stop_index_and_length[current_index,1]+1
                end_index = persistence_stop_index_and_length[current_index,0]
                currrent_stop_duration = end_index - start_index +1
                behavior_state_frame_index[start_index:end_index+1] = [0] * currrent_stop_duration
             
            
            #get average bump amplitude at stop or turning
            output_pooled_dictionary['Bump_amplitude_at_stop'].append(np.mean(Bump_amplitude_V3[behavior_state_frame_index==0]))
            output_pooled_dictionary['Bump_amplitude_at_turn'].append(np.mean(Bump_amplitude_V3[behavior_state_frame_index==1]))
              
                
                
                   
                
            #get Average PVA Angle, PVA strength data after the first/second stimulation
            average_PVA_after_2nd_stim_current = get_signal_after_second_stimulation(PVA_Radian,current_stimulation_end_frame_second_stim ,1)
            average_strength_after_2nd_stim_current = get_signal_after_second_stimulation(PVA_strength,current_stimulation_end_frame_second_stim ,0)
            average_PVA_after_1st_stim_current = get_signal_after_first_stimulation(PVA_Radian,current_stimulation_end_frame_first_stim , current_stimulation_start_frame_second_stim,1)
            average_bump_amplitude_after_1st_stim_current = get_signal_after_first_stimulation(Bump_amplitude_V3,current_stimulation_end_frame_first_stim , current_stimulation_start_frame_second_stim,1)
            average_strength_after_1st_stim_current = get_signal_after_first_stimulation(PVA_strength,current_stimulation_end_frame_first_stim ,current_stimulation_start_frame_second_stim, 0)
            
            
            #get 200ms bin windoe data for  PVA Angle, PVA strength data after the first/second stimulation
            bin_PVA_after_1st_stim_current = get_bin_signal_after_first_stimulation(PVA_Radian,current_stimulation_end_frame_first_stim , current_stimulation_start_frame_second_stim,volume_time,1)
            bin_strength_after_1st_stim_current = get_bin_signal_after_first_stimulation(PVA_strength,current_stimulation_end_frame_first_stim ,current_stimulation_start_frame_second_stim,volume_time,0)
            bin_PVA_after_2nd_stim_current = get_bin_signal_after_second_stimulation(PVA_Radian, current_stimulation_end_frame_second_stim, volume_time, 1, 1)
            bin_strength_after_2nd_stim_current = get_bin_signal_after_second_stimulation(PVA_strength, current_stimulation_end_frame_second_stim, volume_time, 1, 0)
            
            
            #get Stop stamp for 20 frames after the second stim stop index = 0, active index =1
            
            stop_stamp_after_2nd_stim_current = check_stop_stamp(behavior_state_frame_index,current_stimulation_start_frame_second_stim,20)
            
            
            
            
            
            average_PVA_after_2nd_stim_current=pd.DataFrame(average_PVA_after_2nd_stim_current)
            average_strength_after_2nd_stim_current=pd.DataFrame(average_strength_after_2nd_stim_current)
            average_PVA_after_1st_stim_current=pd.DataFrame(average_PVA_after_1st_stim_current)
            average_strength_after_1st_stim_current=pd.DataFrame(average_strength_after_1st_stim_current)
            stop_stamp_after_2nd_stim_current=pd.DataFrame(stop_stamp_after_2nd_stim_current)
            average_bump_amplitude_after_1st_stim_current =pd.DataFrame(average_bump_amplitude_after_1st_stim_current)
            
            
            
            bin_PVA_after_1st_stim_current=pd.DataFrame(bin_PVA_after_1st_stim_current)
            bin_strength_after_1st_stim_current=pd.DataFrame(bin_strength_after_1st_stim_current)
            bin_PVA_after_2nd_stim_current=pd.DataFrame(bin_PVA_after_2nd_stim_current)
            bin_strength_after_2nd_stim_current=pd.DataFrame(bin_strength_after_2nd_stim_current)
            
            
            if count == 0:
                output_pooled_dictionary['average_PVA_after_1st_stim'] = average_PVA_after_1st_stim_current
                output_pooled_dictionary['average_Bump_amplitude_after_1st_stim'] = average_bump_amplitude_after_1st_stim_current
                output_pooled_dictionary['average_PVA_after_2nd_stim'] = average_PVA_after_2nd_stim_current
                output_pooled_dictionary['average_strength_after_1st_stim'] = average_strength_after_1st_stim_current
                output_pooled_dictionary['average_strength_after_2nd_stim'] = average_strength_after_2nd_stim_current
                
                output_pooled_dictionary['stop_stamp_after_2nd_stim'] = stop_stamp_after_2nd_stim_current
                
                output_pooled_dictionary['200ms_bin_PVA_after_1st_stim'] =  bin_PVA_after_1st_stim_current
                output_pooled_dictionary['200ms_bin_strength_after_1st_stim'] = bin_strength_after_1st_stim_current
                output_pooled_dictionary['200ms_bin_PVA_after_2nd_stim'] =  bin_PVA_after_2nd_stim_current
                output_pooled_dictionary['200ms_bin_strength_after_2nd_stim'] = bin_strength_after_2nd_stim_current
            else:
                output_pooled_dictionary['average_Bump_amplitude_after_1st_stim'] = pd.concat([ output_pooled_dictionary['average_Bump_amplitude_after_1st_stim'],average_bump_amplitude_after_1st_stim_current], ignore_index=True)
                output_pooled_dictionary['average_PVA_after_1st_stim'] = pd.concat([ output_pooled_dictionary['average_PVA_after_1st_stim'],average_PVA_after_1st_stim_current], ignore_index=True)
                output_pooled_dictionary['average_PVA_after_2nd_stim'] = pd.concat([ output_pooled_dictionary['average_PVA_after_2nd_stim'],average_PVA_after_2nd_stim_current], ignore_index=True)
                output_pooled_dictionary['average_strength_after_1st_stim'] = pd.concat([ output_pooled_dictionary['average_strength_after_1st_stim'],average_strength_after_1st_stim_current], ignore_index=True)
                output_pooled_dictionary['average_strength_after_2nd_stim'] = pd.concat([ output_pooled_dictionary['average_strength_after_2nd_stim'],average_strength_after_2nd_stim_current], ignore_index=True)
                
                output_pooled_dictionary['stop_stamp_after_2nd_stim'] = pd.concat([ output_pooled_dictionary['stop_stamp_after_2nd_stim'],stop_stamp_after_2nd_stim_current], ignore_index=True)
                
                output_pooled_dictionary['200ms_bin_PVA_after_1st_stim']= pd.concat([ output_pooled_dictionary['200ms_bin_PVA_after_1st_stim'],bin_PVA_after_1st_stim_current], ignore_index=True)
                output_pooled_dictionary['200ms_bin_strength_after_1st_stim']= pd.concat([ output_pooled_dictionary['200ms_bin_strength_after_1st_stim'],bin_strength_after_1st_stim_current], ignore_index=True)
                output_pooled_dictionary['200ms_bin_PVA_after_2nd_stim']= pd.concat([ output_pooled_dictionary['200ms_bin_PVA_after_2nd_stim'],bin_PVA_after_2nd_stim_current], ignore_index=True)
                output_pooled_dictionary['200ms_bin_strength_after_2nd_stim']= pd.concat([ output_pooled_dictionary['200ms_bin_strength_after_2nd_stim'],bin_strength_after_2nd_stim_current], ignore_index=True)
            
               
    
            count=count + 1
    
    return output_pooled_dictionary




def check_stop_stamp (stop_index_array, stimulation_index_array ,frame_duration):
    stop_stamp = np.zeros(len(stimulation_index_array))
    for i in range(len(stimulation_index_array)):
        current_stim_index = stimulation_index_array[i]
        if 1 in stop_index_array[current_stim_index:current_stim_index+frame_duration]:
            stop_stamp[i] = 1
        else:
            stop_stamp[i] = 0
            
        
        
    return stop_stamp


def get_signal_after_second_stimulation(signal_array, stimulation_index_array, if_circular):
    signal_after_stim = np.zeros(len(stimulation_index_array))
    for i in range(len(stimulation_index_array)):
        current_stim_index = stimulation_index_array[i]
        if if_circular == 1:
            signal_after_stim[i] = circmean(signal_array[current_stim_index+1:current_stim_index+3], high=np.pi, low=-np.pi)
        else:
            signal_after_stim[i] = np.mean(signal_array[current_stim_index+1:current_stim_index+3])
        
    return signal_after_stim



def get_signal_after_first_stimulation(signal_array, first_stimulation_index_array, second_stimulation_index_array ,if_circular):
    
    signal_after_stim = np.zeros(len(second_stimulation_index_array))
    for i in range(len(first_stimulation_index_array)):
        current_first_stim_end_index = first_stimulation_index_array[i]
        current_second_stim_start_index = second_stimulation_index_array[i]
        if if_circular == 1:
            signal_after_stim[i] = circmean(signal_array[current_first_stim_end_index+1:current_second_stim_start_index], high=np.pi, low=-np.pi)
        else:
            signal_after_stim[i] = np.mean(signal_array[current_first_stim_end_index+1:current_second_stim_start_index])
        
    return signal_after_stim



def get_bin_signal_after_first_stimulation(signal_array, first_stim_idx_array, second_stim_idx_array, volume_time, if_circular):
    """
    Computes 3 binned means (200ms each) between first and second stimulations.

    Parameters:
    - signal_array: 1D array of signal values.
    - first_stim_idx_array: List/array of first stimulation end frame indices.
    - second_stim_idx_array: List/array of second stimulation start frame indices.
    - volume_time: Time (in seconds) each frame represents (e.g. 0.2 for 200ms).
    - if_circular: 1 if signal is circular (use circmean), 0 for regular mean.

    Returns:
    - A list of [bin1, bin2, bin3] values for each trial.
    """
    bin_means_list = []
    frames_per_bin = int(round(0.2 / volume_time))  # frames per 200ms bin
    n_bins = 3

    for i in range(len(first_stim_idx_array)):
        start_idx = first_stim_idx_array[i] + 1
        end_idx = second_stim_idx_array[i]
        current_trial_bins = []

        for b in range(n_bins):
            bin_start = start_idx + b * frames_per_bin
            bin_end = bin_start + frames_per_bin

            if bin_start >= end_idx:
                current_trial_bins.append(np.nan)
                continue

            bin_data = signal_array[bin_start:min(bin_end, end_idx)]

            # Clean NaNs before computing
            bin_data = bin_data[~np.isnan(bin_data)]

            if len(bin_data) > 0:
                if if_circular:
                    value = circmean(bin_data, high=np.pi, low=-np.pi)
                else:
                    value = np.mean(bin_data)
            else:
                value = np.nan

            current_trial_bins.append(value)

        bin_means_list.append(current_trial_bins)

    return bin_means_list





def get_bin_signal_after_second_stimulation(signal_array, stimulation_index_array, volume_time, total_duration, if_circular):
    """
    Computes binned means (default 200ms per bin) after each second stimulation.

    Parameters:
    - signal_array: 1D array of signal values.
    - stimulation_index_array: Array of second stimulation frame indices.
    - volume_time: Time (in seconds) each frame represents (e.g., 0.2 for 200ms).
    - total_duration: Total duration to analyze after stimulation (in seconds).
    - if_circular: 1 if signal is circular (use circmean), 0 for regular mean.

    Returns:
    - A (n_trials, n_bins) NumPy array of binned means.
    """
    frames_per_bin = int(round(0.2 / volume_time))
    n_bins = int(round(total_duration / 0.2))
    n_trials = len(stimulation_index_array)

    bin_means_list = []

    for i in range(n_trials):
        trial_bins = []
        stim_idx = stimulation_index_array[i]

        for b in range(n_bins):
            bin_start = stim_idx + 1 + b * frames_per_bin
            bin_end = bin_start + frames_per_bin
            bin_data = signal_array[bin_start:bin_end]

            # Clean NaNs
            bin_data = bin_data[~np.isnan(bin_data)]

            if len(bin_data) > 0:
                if if_circular:
                    value = circmean(bin_data, high=np.pi, low=-np.pi)
                else:
                    value = np.mean(bin_data)
            else:
                value = np.nan

            trial_bins.append(value)

        bin_means_list.append(trial_bins)

    return bin_means_list