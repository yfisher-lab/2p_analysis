import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib
import os
from os.path import sep
from fly2p_function_TQ.imaging_2p_fictrac_imaging_alignment import moving_wrapped_plot_by_offset,fictrack_signal_decoding,offset_calculation
from fly2p_function_TQ.imaging_2p_fixed_heading_period_analysis import find_stop_period_on_heading
from fly2p_function_TQ.imaging_2p_PVA_functions import calcualteBumpAmplitude, calcualteBumpAmplitude_V3
from scipy.stats import circvar
from scipy.stats import circmean
from scipy.ndimage import gaussian_filter1d
from scipy.stats import zscore


def run_bar_jumping_analysis_across_trial(directory,directory_jumping_frame,dual_imaging,target_string):
    # Part 1: Create and add each DataFrame to the dictionary
    count = 0
    output_pooled_dictionary = {}
    output_pooled_dictionary['output_PVA_radian_pooled'] = pd.DataFrame()
    output_pooled_dictionary['output_bar_PVA_offset_pooled'] = pd.DataFrame()
    output_pooled_dictionary['output_PVA_strength_pooled'] = pd.DataFrame()
    output_pooled_dictionary['output_PVA_strength_z'] = pd.DataFrame()
    output_pooled_dictionary['output_bar_jumping_stop_stamp_pooled'] = []
    output_pooled_dictionary['stop_s_before_jump'] = []
    output_pooled_dictionary['circular_variance'] = []
    output_pooled_dictionary['circular_mean_before_jump'] = pd.DataFrame()
    output_pooled_dictionary['offset_return_time'] = pd.DataFrame()
    output_pooled_dictionary['Angular_speed'] = pd.DataFrame()
    output_pooled_dictionary['Angular_speed_entire_trial'] = pd.DataFrame()
    output_pooled_dictionary['Bump_speed_entire_trial'] = pd.DataFrame()
    output_pooled_dictionary['Forward_speed'] = pd.DataFrame()
    output_pooled_dictionary['output_flytrial'] = []
    output_pooled_dictionary['volume_time'] = []
    output_pooled_dictionary['Angular_speed_during_offset_return'] = pd.DataFrame()
    #Follow the standard of Ki, et al., 2017
    output_pooled_dictionary['bump_jump_flow_index'] = pd.DataFrame()
    output_pooled_dictionary['sec_bump_move'] = []
    output_pooled_dictionary['delta7_bump_amplitude_at_bump_move'] = []
    output_pooled_dictionary['Angular_speed_at_bump_move'] = []

    
    if dual_imaging == 1:
        output_pooled_dictionary['output_PVA_radian_pooled_red'] = pd.DataFrame()
        output_pooled_dictionary['output_bar_PVA_red_offset_pooled'] = pd.DataFrame()
        output_pooled_dictionary['output_PVA_strength_pooled_red'] = pd.DataFrame()
        output_pooled_dictionary['output_greed_red_PVA_offset'] = pd.DataFrame()
        output_pooled_dictionary['circular_variance_red'] = []
        output_pooled_dictionary['PVA_strength_ratio'] = []
        output_pooled_dictionary['Angular_speed_entire_trial_red'] = pd.DataFrame()
        output_pooled_dictionary['EPG_bump_amplitude_at_bump_move'] = []
        output_pooled_dictionary['EPG_bump_amplitude_before_bar_jump'] = []
    
    
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
            Wrapped_Bar_jumped = -current_file['Bar_Position/Heading_jumping_bar'].values 
        
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
        
            if dual_imaging == 1:
                PVA_Angle_red = current_file['PVA_Angle_red'].values
                PVA_Radian_red = current_file['PVA_Radian_red'].values
                PVA_strength_red = current_file['PVA_strength_red'].values
                PVA_strength_red_z = zscore(PVA_strength_red)
                PVA_strength_ratio = PVA_strength/PVA_strength_red
                PVA_strength_ratio_log = np.log(PVA_strength_ratio)
                dff_normalized_8_roi_red = np.array([current_file[f'dFF_Roi_{i}_red'] for i in range(1, 9)]).T
                Bump_amplitude_V3_red, Bump_amplitude_V3_opposite_red = calcualteBumpAmplitude_V3(dff_normalized_8_roi_red,PVA_Radian_red)
              
        
            #Get persistence period
            persistence_stop_index_and_length = find_stop_period_on_heading(head_velocity_array = Angular_velocity,degree_of_tolerance =15,shortest_stopFrame=int(np.ceil(3/volume_time)))
        
            
            
            #Store the flyinformation if wants to fly-by-fly analysis later on and to align
            single_trial_info = single_df.split("-")
            flytrial = [single_trial_info[0], single_trial_info[1]]
            for i in range(10):
                concatenated_string = f"{single_trial_info[0]}-{single_trial_info[1]}-{i+1}"
                output_pooled_dictionary['output_flytrial'].append(concatenated_string )
        
        
            #Get jumping frame data 
            current_jumping_frame = None
            concatenated_string_2 = f"{single_trial_info[0]}-{single_trial_info[1]}"
            for single_jf in os.listdir(directory_jumping_frame):  
                if concatenated_string_2  in single_jf:
                    file_path = os.path.join(directory_jumping_frame, single_jf)
                    current_jumping_frame = pd.read_csv(file_path)
                    break
           
            current_jumping_frame = current_jumping_frame['jumping_frame'].values
 
            
            #Create a index array that indicates whether the current frame is stop/active stop index = 0, active index =1
            behavior_state_frame_index = np.ones(len(PVA_Radian))
            for current_index in range(len(persistence_stop_index_and_length)):
                start_index = persistence_stop_index_and_length[current_index,0]-persistence_stop_index_and_length[current_index,1]+1
                end_index = persistence_stop_index_and_length[current_index,0]
                currrent_stop_duration = end_index - start_index +1
                behavior_state_frame_index[start_index:end_index+1] = [0] * currrent_stop_duration
           
            
            #Calculte the durations(s) of stop before the jump
            stop_s_before_jump = duration_of_stop_before_bar_jump(current_jumping_frame,behavior_state_frame_index,volume_time)
                      
        
            #Calculate Offset between Bar and PVA or two PVA if doing dual imaging
            radian_offset_PVA_Bar = offset_calculation(Wrapped_Bar_jumped, PVA_Radian, IfRadian = True)
        
            if dual_imaging == 1:
                radian_offset_PVA_red_Bar = offset_calculation(Wrapped_Bar_jumped, PVA_Radian_red, IfRadian = True)
                radian_offset_PVA_red_green = offset_calculation(PVA_Radian, PVA_Radian_red, IfRadian = True)
        
            #Get circular mean before the jump 
            circular_mean_before_bar_jump = get_circular_mean_before_bar_jump(radian_offset_PVA_Bar, current_jumping_frame, 20,volume_time)
            
            
            #Get jump/flow/index after the jump (1 for flow,2 for jump)
            bump_flow_jump_index_current,bump_flow_jump_start_frame, sec_bump_move = get_bump_jump_flow_index_array(current_jumping_frame,volume_time,PVA_Radian)
            
            
            EPG_bump_amplitude_at_bump_move_current = get_bump_amplitude_at_bump_jump_flow(Bump_amplitude_V3_red ,bump_flow_jump_start_frame)
            
 
            delta7_bump_amplitude_at_bump_move_current = get_bump_amplitude_at_bump_jump_flow(Bump_amplitude_V3 ,bump_flow_jump_start_frame)
    
            Angular_speed_at_bump_move_current  = get_bump_amplitude_at_bump_jump_flow(Angular_speed_degrees,bump_flow_jump_start_frame)
            
            
            #Calculate the time that needs for bump to return to its previous offset
            time_for_offset_recover = get_time_offset_recover_after_bar_jump(radian_offset_PVA_Bar,current_jumping_frame,volume_time)
            
            
            
            
            
            
            #Get all the jump-trigged data for further analysis 
            jump_triggered_behavior_stamp =get_behavior_state_for_jump_triggered_signal(current_jumping_frame,behavior_state_frame_index, volume_time,3)
            jump_triggered_PVA = get_data_before_after_bar_jump(current_jumping_frame, PVA_Radian, volume_time,29)
            jump_triggered_Bar_data = get_data_before_after_bar_jump(current_jumping_frame, Wrapped_Bar_jumped, volume_time,29) 
            jump_triggered_Bar_PVA_offset = get_data_before_after_bar_jump(current_jumping_frame, radian_offset_PVA_Bar, volume_time,29) 
            jump_triggered_Angular_speed = get_data_before_after_bar_jump(current_jumping_frame, Angular_speed_degrees, volume_time,29) 
            jump_triggered_Forward_speed = get_data_before_after_bar_jump(current_jumping_frame, Forward_speed_degrees, volume_time,29) 
        
        
            if dual_imaging == 1:
                jump_triggered_PVA_red = get_data_before_after_bar_jump(current_jumping_frame, PVA_Radian_red, volume_time,29)
                jump_triggered_Bar_PVA_red_offset = get_data_before_after_bar_jump(current_jumping_frame, radian_offset_PVA_red_Bar, volume_time,29)
                jump_triggered_Bar_PVA_red_green_offset = get_data_before_after_bar_jump(current_jumping_frame, radian_offset_PVA_red_green, volume_time,29)
                strength_ratio_before_jump = get_PVA_strength_ratio_before_bar_jump (current_jumping_frame,PVA_strength_ratio_log,volume_time,0.5) 
                EPG_bump_amplitude_before_bar_jump_current = get_PVA_strength_ratio_before_bar_jump (current_jumping_frame,Bump_amplitude_V3_red,volume_time,0.5) 
                Angular_speed_during_offset_return = get_Angular_speed_during_offset_return(current_jumping_frame,Angular_speed_degrees,volume_time,time_for_offset_recover)
        
            
            
            
            jump_triggered_PVA_current = pd.DataFrame(jump_triggered_PVA)
            jump_triggered_Bar_PVA_offset_current = pd.DataFrame(jump_triggered_Bar_PVA_offset)
            jump_triggered_PVA_red_current= pd.DataFrame(jump_triggered_PVA_red)
            jump_triggered_Bar_PVA_red_offset_current = pd.DataFrame(jump_triggered_Bar_PVA_red_offset)
            jump_triggered_Bar_PVA_red_green_offset_current = pd.DataFrame(jump_triggered_Bar_PVA_red_green_offset)
            jump_triggered_behavior_stamp_current = pd.DataFrame(jump_triggered_behavior_stamp)
            jump_triggered_Angular_speed_current = pd.DataFrame(jump_triggered_Angular_speed)
            jump_triggered_Forward_speed_current = pd.DataFrame(jump_triggered_Forward_speed)
            stop_s_before_jump_current = pd.DataFrame(stop_s_before_jump)
            strength_ratio_before_jump_current =pd.DataFrame(strength_ratio_before_jump)
            sec_bump_move = pd.DataFrame(sec_bump_move)
            EPG_bump_amplitude_at_bump_move_current =pd.DataFrame(EPG_bump_amplitude_at_bump_move_current)
            EPG_bump_amplitude_before_bar_jump_current =pd.DataFrame(EPG_bump_amplitude_before_bar_jump_current) 
            delta7_bump_amplitude_at_bump_move_current =pd.DataFrame(delta7_bump_amplitude_at_bump_move_current)
            Angular_speed_at_bump_move_current  = pd.DataFrame(Angular_speed_at_bump_move_current)
            circular_mean_before_bar_jump_current = pd.DataFrame(circular_mean_before_bar_jump )
            time_for_offset_recover_current = pd.DataFrame(time_for_offset_recover)
            Angular_speed_during_offset_return_current = pd.DataFrame(Angular_speed_during_offset_return)
 
            #Store the data 
            if count == 0:
                output_pooled_dictionary['output_PVA_radian_pooled'] =  jump_triggered_PVA_current
                output_pooled_dictionary['output_bar_PVA_offset_pooled'] = jump_triggered_Bar_PVA_offset_current 
                output_pooled_dictionary['output_bar_jumping_stop_stamp_pooled'] = jump_triggered_behavior_stamp_current
                output_pooled_dictionary['Angular_speed'] = jump_triggered_Angular_speed_current
                output_pooled_dictionary['Forward_speed'] = jump_triggered_Forward_speed_current
                output_pooled_dictionary['stop_s_before_jump'] = stop_s_before_jump_current
                output_pooled_dictionary['circular_mean_before_jump']= circular_mean_before_bar_jump_current
                output_pooled_dictionary['offset_return_time']  = time_for_offset_recover_current
                output_pooled_dictionary['Angular_speed_during_offset_return']=Angular_speed_during_offset_return_current
                output_pooled_dictionary['Angular_speed_entire_trial'] = pd.DataFrame(Angular_velocity_degrees)
                output_pooled_dictionary['Bump_speed_entire_trial'] = pd.DataFrame(Bump_speed_degrees)
                output_pooled_dictionary['bump_jump_flow_index'] = pd.DataFrame(bump_flow_jump_index_current)
                output_pooled_dictionary['sec_bump_move'] = sec_bump_move
                output_pooled_dictionary['delta7_bump_amplitude_at_bump_move'] = delta7_bump_amplitude_at_bump_move_current
                output_pooled_dictionary['Angular_speed_at_bump_move'] = Angular_speed_at_bump_move_current
                if dual_imaging == 1:
                    output_pooled_dictionary['output_PVA_radian_pooled_red'] = jump_triggered_PVA_red_current
                    output_pooled_dictionary['output_bar_PVA_red_offset_pooled'] = jump_triggered_Bar_PVA_red_offset_current
                    output_pooled_dictionary['output_greed_red_PVA_offset'] = jump_triggered_Bar_PVA_red_green_offset_current
                    output_pooled_dictionary['PVA_strength_ratio'] = strength_ratio_before_jump_current
                    output_pooled_dictionary['EPG_bump_amplitude_at_bump_move'] = EPG_bump_amplitude_at_bump_move_current
                    output_pooled_dictionary['EPG_bump_amplitude_before_bar_jump'] = EPG_bump_amplitude_before_bar_jump_current
            else:
                output_pooled_dictionary['output_PVA_radian_pooled'] = pd.concat([ output_pooled_dictionary['output_PVA_radian_pooled'],jump_triggered_PVA_current], ignore_index=True)
                output_pooled_dictionary['output_bar_PVA_offset_pooled'] = pd.concat([ output_pooled_dictionary['output_bar_PVA_offset_pooled'],jump_triggered_Bar_PVA_offset_current], ignore_index=True)
                output_pooled_dictionary['output_bar_jumping_stop_stamp_pooled'] = pd.concat([ output_pooled_dictionary['output_bar_jumping_stop_stamp_pooled'],jump_triggered_behavior_stamp_current], ignore_index=True)
                output_pooled_dictionary['Angular_speed'] = pd.concat([ output_pooled_dictionary['Angular_speed'],jump_triggered_Angular_speed_current], ignore_index=True)
                output_pooled_dictionary['Forward_speed'] = pd.concat([ output_pooled_dictionary['Forward_speed'],jump_triggered_Forward_speed_current], ignore_index=True)
                output_pooled_dictionary['stop_s_before_jump'] = pd.concat([ output_pooled_dictionary['stop_s_before_jump'],stop_s_before_jump_current], ignore_index=True)
                output_pooled_dictionary['circular_mean_before_jump'] = pd.concat([ output_pooled_dictionary['circular_mean_before_jump'],circular_mean_before_bar_jump_current], ignore_index=True, axis =1)
                output_pooled_dictionary['offset_return_time']  = pd.concat([ output_pooled_dictionary['offset_return_time'],time_for_offset_recover_current], ignore_index=True)
                output_pooled_dictionary['Angular_speed_during_offset_return']  = pd.concat([ output_pooled_dictionary['Angular_speed_during_offset_return'],Angular_speed_during_offset_return_current], ignore_index=True)
                output_pooled_dictionary['Angular_speed_entire_trial']  = pd.concat([ output_pooled_dictionary['Angular_speed_entire_trial'],pd.DataFrame(Angular_velocity_degrees)], ignore_index=True)
                output_pooled_dictionary['Bump_speed_entire_trial']  = pd.concat([ output_pooled_dictionary['Bump_speed_entire_trial'],pd.DataFrame(Bump_speed_degrees)], ignore_index=True)
                output_pooled_dictionary['bump_jump_flow_index']  = pd.concat([ output_pooled_dictionary['bump_jump_flow_index'],pd.DataFrame(bump_flow_jump_index_current)], ignore_index=True)
                output_pooled_dictionary['sec_bump_move'] = pd.concat([ output_pooled_dictionary['sec_bump_move'],sec_bump_move], ignore_index=True)
                output_pooled_dictionary['delta7_bump_amplitude_at_bump_move'] = pd.concat([ output_pooled_dictionary['delta7_bump_amplitude_at_bump_move'],delta7_bump_amplitude_at_bump_move_current], ignore_index=True)
                output_pooled_dictionary['Angular_speed_at_bump_move'] =pd.concat([ output_pooled_dictionary['Angular_speed_at_bump_move'],Angular_speed_at_bump_move_current], ignore_index=True) 
                               
                
                if dual_imaging == 1:
                    output_pooled_dictionary['output_PVA_radian_pooled_red'] = pd.concat([ output_pooled_dictionary['output_PVA_radian_pooled_red'],jump_triggered_PVA_red_current], ignore_index=True)
                    output_pooled_dictionary['output_bar_PVA_red_offset_pooled'] = pd.concat([ output_pooled_dictionary['output_bar_PVA_red_offset_pooled'],jump_triggered_Bar_PVA_red_offset_current], ignore_index=True)
                    output_pooled_dictionary['output_greed_red_PVA_offset'] = pd.concat([ output_pooled_dictionary['output_greed_red_PVA_offset'],jump_triggered_Bar_PVA_red_green_offset_current], ignore_index=True)
                    output_pooled_dictionary['PVA_strength_ratio'] = pd.concat([ output_pooled_dictionary['PVA_strength_ratio'],strength_ratio_before_jump_current], ignore_index=True)
                    output_pooled_dictionary['EPG_bump_amplitude_at_bump_move'] = pd.concat([ output_pooled_dictionary['EPG_bump_amplitude_at_bump_move'],EPG_bump_amplitude_at_bump_move_current], ignore_index=True)
                    output_pooled_dictionary['EPG_bump_amplitude_before_bar_jump'] = pd.concat([ output_pooled_dictionary['EPG_bump_amplitude_before_bar_jump'],EPG_bump_amplitude_before_bar_jump_current], ignore_index=True)
            
            
            
            count = count + 1
          
    return output_pooled_dictionary




        

def compute_circular_vel(angles, time_intervals):
  
    
    # Initialize the speed array with NaNs or zeros, same length as angles
    speed = np.full_like(angles, np.nan, dtype=float)

    # Calculate the difference between consecutive angles
    diff = np.diff(angles)

    # Adjust differences for circular wrap-around
    adjusted_diff = (diff + 180) % 360 - 180

    # Compute speed by dividing the adjusted differences by the time intervals
    speed[1:] = adjusted_diff / time_intervals

    return speed



def get_data_before_after_bar_jump(jumping_index_array, signal_array, volume_time,signal_length_s):
    signal_length_half = int(np.ceil(signal_length_s/volume_time))
    total_length_for_triggered_signal = signal_length_half * 2 + 1
    jump_triggered_signal = np.zeros((len(jumping_index_array),total_length_for_triggered_signal))
    
    for current_index in range (len(jumping_index_array)):
        jump_index_current = jumping_index_array[current_index]
        start_index = jump_index_current - signal_length_half 
        end_index = jump_index_current + signal_length_half 
        
        jump_triggered_signal[current_index,:] = signal_array[start_index:end_index+1]
    
    
    return jump_triggered_signal


def get_behavior_state_for_jump_triggered_signal(jumping_index_array,behavior_state_array, volume_time,stop_criteria_s):
    signal_length_half = int(np.ceil(stop_criteria_s/volume_time))
    
    #since active_index = 1, if we find 1 in the junmping period before/after criteria s, then it is not stop during the jump(might need to be changed later on)
    
    behavior_stamp_for_jump_triggered_trial = np.zeros(len(jumping_index_array))
    
    for current_index in range (len(jumping_index_array)):
        jump_index_current = jumping_index_array[current_index]
        start_index = jump_index_current - signal_length_half 
        end_index = jump_index_current + signal_length_half 
        behavior_state_array_current = behavior_state_array[start_index:end_index+1]
        
        if np.any(behavior_state_array_current == 1):
            behavior_stamp_for_jump_triggered_trial[current_index] = 1 
        else:
            behavior_stamp_for_jump_triggered_trial[current_index] = 0 
    
    return behavior_stamp_for_jump_triggered_trial




def circular_difference(angle1, angle2):
    # Compute the raw difference
    diff = angle1 - angle2

    # Adjust for circular wrap-around
    adjusted_diff = (diff + np.pi) % (np.pi*2) - np.pi

    return adjusted_diff

def is_within_range(angle1, angle2, threshold):
    # Compute the circular difference
    diff = circular_difference(angle1, angle2)

    # Compute the absolute difference
    abs_diff = np.abs(diff)

    # Check if the absolute difference is within the threshold
    return abs_diff <= threshold



def get_bump_jump_flow_index_array(jumping_index_array,volume_time,PVA_array):
    jump_flow_index = np.zeros(len(jumping_index_array))
    jump_start_indices = np.zeros(len(jumping_index_array))
    bump_move_time = np.zeros(len(jumping_index_array))
    #Criteria from Kim et al.,2017
    frame_before_jump = int(np.ceil(1/volume_time))
    frame_after_jump = int(np.ceil(2.5/volume_time))
    frame_after_jump_for_bump_move = int(np.ceil(29.5/volume_time))
                
                
                
                
    for current_index in range (len(jumping_index_array)):
        jump_index_current = jumping_index_array[current_index]
         
        circular_mean_initial = circmean(PVA_array[jump_index_current-frame_before_jump:jump_index_current+1], high=np.pi, low=-np.pi)
        circular_mean_4_frame_jump = circmean(PVA_array[jump_index_current:jump_index_current+4], high=np.pi, low=-np.pi)
        #2.5s after jump
        circular_mean_after_jump = circmean(PVA_array[jump_index_current+1:jump_index_current+1+frame_after_jump], high=np.pi, low=-np.pi)
        
        
        
        # Step through the window after the event
        for step in range(1, frame_after_jump_for_bump_move + 1):
            jump_detected = False
            slide_detected = False
            window_mean = circmean(PVA_array[jump_index_current+1:jump_index_current+1+step], high=np.pi, low=-np.pi)

            # Detect a jump (using larger threshold)
            if not is_within_range(circular_mean_initial, window_mean, threshold = np.pi/3):
                jump_start_indices[current_index] = jump_index_current + step  # Store when jump starts
                bump_move_time[current_index] = step*volume_time
                jump_detected = True  # Stop checking for jump
                break
            
            # Detect a slide (using smaller threshold)
            if not is_within_range(circular_mean_initial, window_mean, threshold = np.pi/8):
                jump_start_indices[current_index] = jump_index_current + step  # Store when slide starts
                bump_move_time[current_index] = step*volume_time
                slide_detected = True  # Stop checking for slide
                break

        
        if not jump_detected and not slide_detected:
            jump_start_indices[current_index] = np.nan
            bump_move_time[current_index] = np.nan
            
            
            
            
        if is_within_range(circular_mean_initial,circular_mean_4_frame_jump,threshold = np.pi/4)== False:
            jump_flow_index[current_index] = 0
        elif is_within_range(circular_mean_initial,circular_mean_after_jump,threshold = np.pi/4) == True:
            jump_flow_index[current_index] = 1
        else:
            jump_flow_index[current_index] = 2
        
    
    return jump_flow_index,jump_start_indices,bump_move_time 


def get_bump_amplitude_at_bump_jump_flow(bump_amplitude_array,jump_start_array):
    amplitude_at_jump =  np.zeros(len(jump_start_array))
 
    
    for current_frame in  range(len(jump_start_array)):
        if np.isnan(jump_start_array[current_frame]):
            amplitude_at_jump[current_frame] = np.nan
        else:
            frame = int(jump_start_array[current_frame])
            amplitude_at_jump[current_frame] = np.mean(bump_amplitude_array[frame-2:frame+1])
    
    return amplitude_at_jump


def get_PVA_strength_ratio_before_bar_jump(jumping_index_array,PVA_strtength_ratio_array,volume_time,s_before_jump):
    strength_ratio_before_jump = np.zeros(len(jumping_index_array))
    frame_before_jump = int(np.ceil(s_before_jump/volume_time))
    
    
    for current_index in range (len(jumping_index_array)):
        jump_index_current = jumping_index_array[current_index]
        strength_ratio_before_jump[current_index] = np.mean(PVA_strtength_ratio_array[jump_index_current-frame_before_jump :jump_index_current])
        
    return strength_ratio_before_jump



def get_Angular_speed_during_offset_return(jumping_index_array,angular_speed_array,volume_time,time_for_offset_recover):
    mean_angular_after_jump = np.zeros(len(jumping_index_array))
    frame_after_jump = np.zeros(len(jumping_index_array), dtype=int)
    for current_index in range (len(jumping_index_array)):
        if np.isnan(time_for_offset_recover[current_index]) == True:
            frame_after_jump[current_index] = 0
        elif time_for_offset_recover[current_index] <= 0:
            frame_after_jump[current_index] = 0
        else:
            frame_after_jump[current_index] = int(np.ceil(time_for_offset_recover[current_index]/volume_time))
    
    
    for current_index in range (len(jumping_index_array)):
        jump_index_current = jumping_index_array[current_index]
        if frame_after_jump[current_index] == 0:
            mean_angular_after_jump [current_index] = np.NaN
        else:
            mean_angular_after_jump [current_index] = np.mean(angular_speed_array[jump_index_current:jump_index_current+ frame_after_jump[current_index]+1])
        
    return mean_angular_after_jump




def duration_of_stop_before_bar_jump(jumping_index_array,behavior_state_array,volume_time):
    
    stop_duration_before_jump = np.zeros(len(jumping_index_array))
    
    
    for current_index in range(len(jumping_index_array)):
        #Create behavior state arrar till the jumping frame
        current_behavior_state_array = behavior_state_array[:jumping_index_array[current_index]] 
        active_indices = np.where(current_behavior_state_array == 1)[0] 
        if len(active_indices) == 0:
            stop_duration_before_jump[current_index] =  jumping_index_array[current_index]*volume_time
        else:
            stop_duration_before_jump[current_index] =  (jumping_index_array[current_index] - active_indices[-1]) * volume_time
        
    return stop_duration_before_jump
  

def plot_the_aligned_bar_jump_trial(data,frame_before,whether_shift,shift_radian,stop_s_before_jump, strength_ratio,whether_color_code_by_stop_duration_before,stop_stamp,volume_time,s_after_jump):
    jump_triggeed__data = data
    
    # Define the middle (jumping frame frame and the range around it
    middle_frame = int((data.shape[1]-1)/2) + 1
    
    # Get the value of the frame just before the middle frame for each trial
    frame_before_middle = middle_frame - frame_before
    frame_before_middle_values = data.iloc[:, frame_before_middle]
    
    # Calculate the shift required for each trial to align the value at the frame       before the middle to the input rasdian values
    if whether_shift == 1:
        target_value = shift_radian
        shifts = target_value - frame_before_middle_values
        
    # Ensure the shifts remain within the range from -π to π
    shifts = np.mod(shifts + np.pi, 2 * np.pi) - np.pi
    
    # Align the trials by shifting them circularly
    if whether_shift == 0:
        aligned_data =  jump_triggeed__data
    else:
        aligned_data = np.zeros_like(jump_triggeed__data)
        for i in range(data.shape[0]):
            aligned_data[i, :] = np.mod(jump_triggeed__data.iloc[i, :] +  shifts[i] + np.pi, 2 * np.pi) - np.pi
    
    
    # Define the range to plot (20 frames before and after the middle frame)
    plot_start_frame = middle_frame - frame_before
    
    frame_number_after_middle = int(s_after_jump/volume_time)
    plot_end_frame = middle_frame + frame_number_after_middle + 1  # +1 to include the end frame
    
    if whether_color_code_by_stop_duration_before == 1:
        # Normalize the stop_duration values to range between 0 and 1 for colormap
        stop_duration = stop_s_before_jump.values
        norm = plt.Normalize(stop_duration.min(), stop_duration.max())
        # Create a monochromatic colormap
        colormap = matplotlib.colormaps['Blues']
        
        
        strength_ratio = strength_ratio.values
        norm_2 = plt.Normalize(strength_ratio.min(),strength_ratio.max())
        # Create a monochromatic colormap
        colormap_2 = matplotlib.colormaps['Reds']
        
        # Plot each aligned trial within the specified range
        plt.figure(figsize=(25, 10))
        for i in range(aligned_data.shape[0]):
            color = colormap(norm(stop_duration[i]))
            color_2 = colormap_2(norm_2(strength_ratio[i]))
            if int(stop_stamp[0][i]) == 0 :
                plt.plot(range(-frame_before, frame_number_after_middle+1), aligned_data[i, plot_start_frame:plot_end_frame], label=f'Trial {i+1}',linewidth=3, linestyle='-', marker='o', markersize=4, color = color)
            else:
                plt.plot(range(-frame_before, frame_number_after_middle+1), aligned_data[i, plot_start_frame:plot_end_frame], label=f'Trial {i+1}',linewidth=3, linestyle='-', marker='o', markersize=4, color = color_2)
    else:
        plt.figure(figsize=(25, 10))
        for i in range(aligned_data.shape[0]):
            if int(stop_stamp[0][i]) == 0 :
                plt.plot(range(-frame_before,  frame_number_after_middle+1), aligned_data[i, plot_start_frame:plot_end_frame], label=f'Trial {i+1}',linewidth=3, linestyle='-', marker='o', markersize=4, color = 'navy')
            else:
                plt.plot(range(-frame_before,  frame_number_after_middle+1), aligned_data[i, plot_start_frame:plot_end_frame], label=f'Trial {i+1}',linewidth=3, linestyle='-', marker='o', markersize=4, color = 'gray')
                
    plt.xlabel('Time (frames)',fontsize = 15)  # Change the x-axis label to represent frames
    plt.ylabel('PVA Offset (radians)',fontsize = 15)
    
    
    # Set custom tick labels for the x-axis with exactly 10 ticks
    #plt.xticks(ticks=plt.xticks()[0][0:], labels=np.array(plt.xticks()[0][0:]*volume_time, dtype=np.int64), fontsize =15)
    #plt.yticks(fontsize = 15)
    # Set custom tick labels for the x-axis
    frame_labels = [f'{(frame - middle_frame) * volume_time:.1f}s' for frame in range(plot_start_frame, plot_end_frame)]
    plt.xticks(range(-frame_before, frame_number_after_middle+1), frame_labels, fontsize = 15)
    plt.locator_params(axis = 'x', nbins=6)
    plt.yticks(fontsize = 15)
    
    # Add a vertical line at the middle frame
    plt.axvline(x=0, color='r', linestyle='--', label='Middle Frame')

    # Show the plot
    plt.tight_layout()
    plt.show()
    
    

def get_circular_mean_before_bar_jump(data,jumping_index_array,s_before,volume_time):
    circular_mean_before_bar_jump = np.zeros(len(jumping_index_array))
    total_frame_before_jump = int(s_before/volume_time)
    data_smoothed = smooth_circular_data(data, 5)
    for i in range(len(jumping_index_array)):
        current_jump_index = jumping_index_array[i]
        window_data = data_smoothed[(current_jump_index-total_frame_before_jump):current_jump_index]
        circular_mean_before_bar_jump[i] = circmean(window_data, high=np.pi, low=-np.pi)
                
    return circular_mean_before_bar_jump



def smooth_circular_data(data, sigma):
    # Convert the angles to complex exponential form
    complex_data = np.exp(1j * data)
    
    # Apply Gaussian smoothing in the complex plane
    smoothed_complex_data = gaussian_filter1d(complex_data, sigma=sigma, axis=-1)
    
    # Normalize to keep the magnitude as 1
    smoothed_complex_data /= np.abs(smoothed_complex_data)
    
    # Convert the smoothed complex data back to angles
    smoothed_data = np.angle(smoothed_complex_data)
    
    return smoothed_data


def sliding_circular_mean(data, window_size):
    half_window = window_size // 2
    circular_means = np.zeros(len(data))
    
    for i in range(len(data)):
        if i < half_window:
            # For the beginning of the array
            window_data = data[0:i + half_window + 1]
        elif i > len(data) - half_window - 1:
            # For the end of the array
            window_data = data[i - half_window:]
        else:
            # For the middle of the array
            window_data = data[i - half_window:i + half_window + 1]
        
        # Calculate the circular mean for the current window
        circular_means[i] = circmean(window_data, high=np.pi, low=-np.pi)
    
    return circular_means


def get_time_offset_recover_after_bar_jump(offset_data,jumping_index_array,volume_time):
    time_return = np.zeros(len(jumping_index_array))
    circular_mean = circmean(offset_data, high=np.pi, low=-np.pi)
    #Find range for acceptable offset return (+-45 degrees)
    angle_radians = 60 * (np.pi / 180)
    circular_mean_upper_bound = (circular_mean+angle_radians +np.pi) % (2 * np.pi) - np.pi 
    circular_mean_lower_bound = (circular_mean-angle_radians+np.pi) % (2 * np.pi) - np.pi 
    for i in range(len(jumping_index_array)):
        current_jump_index = jumping_index_array[i]
        #29s after jump for detecting bump return 
        window_data = offset_data[current_jump_index+1:current_jump_index+int(29/volume_time)]
        offset_return_index_array = is_within_circular_range(window_data,circular_mean_lower_bound,circular_mean_upper_bound)
        return_indices = np.where(offset_return_index_array == 1)[0]
        if return_indices.size > 0:
            time_return[i] = np.round(return_indices[0]*volume_time,decimals = 1)
        else:
            time_return[i] = np.NaN
       
    return time_return
    
    
def is_within_circular_range(data, range_start, range_end):
    """
    Check if each element in a circular data array falls within a specified range.
    
    Parameters:
    - data: numpy array of circular data ranging from -pi to pi.
    - range_start: start of the range.
    - range_end: end of the range.
    
    Returns:
    - A numpy array with 1 if the element falls within the range, otherwise 0.
    """
    # Normalize the range to be between -pi and pi
    range_start = np.mod(range_start + np.pi, 2 * np.pi) - np.pi
    range_end = np.mod(range_end + np.pi, 2 * np.pi) - np.pi
    
    
    # If the range does not wrap around
    if range_start <= range_end:
        mask = (data >= range_start) & (data <= range_end)
    else:
        # If the range wraps around
        mask = (data >= range_start) | (data <= range_end)
    
    # Convert boolean mask to 1s and 0s
    result = mask.astype(int)
    
    return result