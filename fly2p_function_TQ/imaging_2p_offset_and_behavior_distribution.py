import numpy as np
import pandas as pd
from scipy.stats import zscore
from scipy.stats import circmean
from scipy.stats import circvar
import seaborn as sns
from matplotlib import pyplot as plt
import os
from os.path import sep
from fly2p_function_TQ.imaging_2p_preprocessing import low_pass_filter_TQ, get_dff_array, normalizing_dff_array
from fly2p_function_TQ.imaging_2p_fictrac_imaging_alignment import moving_wrapped_plot_by_offset,fictrack_signal_decoding,offset_calculation 
from fly2p_function_TQ.imaging_2p_PVA_functions import calcualteBumpAmplitude, calcualteBumpAmplitude_V3, calculateBumpWidth_v1, PVA_radian_to_angle,PVAangleToRoi,strong_PVA_duration,strong_PVA_index,get_behavior_state_of_strong_PVA,get_bump_shape_at_strong_signal,get_PVA_at_strong_signal,strong_signal_index,weak_signal_index,PVA_radian_calcul,PVA_radian_calcul_norm,calcualteBumpAmplitude_V2_green
from fly2p_function_TQ.imaging_2p_fixed_heading_period_analysis import find_stop_period_on_heading





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





def plot_circular_variance_distribution_at_turning_slide_window(data_groups, group_names=None, bins=30):
    """
    Plots the circular variance distribution for up to two groups using completely separate calculations.

    Parameters:
    - data_groups (list of lists or np.ndarray): List of 1 or 2 datasets to be plotted.
    - group_names (list of str, optional): Names corresponding to each dataset.
    - bins (int): Number of bins for the histogram.
    """

    # Ensure only 1 or 2 groups are provided
    if len(data_groups) not in [1, 2]:
        raise ValueError("This function only supports 1 or 2 groups.")

    # Default group names if not provided
    if group_names is None:
        group_names = [f"Group {i+1}" for i in range(len(data_groups))]

    # Define bins (from 0 to 1)
    bin_edges = np.linspace(0, 1, bins + 1)

    # Define color map for groups
    colors = ['dodgerblue', 'darkorange'][:len(data_groups)]  # Blue for 1st group, Orange for 2nd

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Process each group **completely separately**
    if len(data_groups) == 1:
        # Process **only one group**
        data_1 = np.concatenate([np.asarray(d, dtype=float).flatten() for d in data_groups[0]])
        hist_1, _ = np.histogram(data_1, bins=bin_edges)  # Compute histogram separately
        hist_1 = hist_1 / hist_1.sum() if hist_1.sum() > 0 else hist_1  # Normalize separately
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Bin centers

        # Plot histogram
        ax.fill_between(bin_centers, hist_1, alpha=0.4, color=colors[0], label=group_names[0])
        
        # Compute mean
        mean_1 = np.mean(data_1)
        ax.axvline(mean_1, color=colors[0], linestyle='dashed', linewidth=2, label=f'{group_names[0]} Mean: {mean_1:.2f}')

        # Set y-limits
        max_y = max(hist_1) * 1.1 
        ax.set_ylim(0, max_y)

    elif len(data_groups) == 2:
        # Process **two groups separately**
        data_1 = np.concatenate([np.asarray(d, dtype=float).flatten() for d in data_groups[0]])
        data_2 = np.concatenate([np.asarray(d, dtype=float).flatten() for d in data_groups[1]])

        # Compute histograms separately
        hist_1, _ = np.histogram(data_1, bins=bin_edges)
        hist_2, _ = np.histogram(data_2, bins=bin_edges)

        # Normalize each histogram separately
        hist_1 = hist_1 / hist_1.sum() 
        hist_2 = hist_2 / hist_2.sum() 

        # Compute bin centers
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  

        # Plot histograms separately
        ax.fill_between(bin_centers, hist_1, alpha=0.4, color=colors[0], label=group_names[0])
        ax.fill_between(bin_centers, hist_2, alpha=0.4, color=colors[1], label=group_names[1])

        # Compute means separately
        mean_1 = np.mean(data_1)
        mean_2 = np.mean(data_2)

        # Plot means separately
        ax.axvline(mean_1, color=colors[0], linestyle='dashed', linewidth=2, label=f'{group_names[0]} Mean: {mean_1:.2f}')
        ax.axvline(mean_2, color=colors[1], linestyle='dashed', linewidth=2, label=f'{group_names[1]} Mean: {mean_2:.2f}')

        # Set y-limits based on max of both histograms
        max_y = max(max(hist_1), max(hist_2)) * 1.1 
        ax.set_ylim(0, max_y)

    # Formatting
    ax.set_xlim(0, 1)  # X-axis from 0 to 1
    ax.set_xticks(np.linspace(0, 1, 6))  
    ax.set_xlabel("Circular Variance", fontsize=14)
    ax.set_ylabel("Proportion", fontsize=14)
    ax.legend(fontsize=12, loc="upper right")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title("Circular Variance Distribution During Walking", fontsize=16, fontweight='bold')

    # Show plot
    plt.tight_layout()
    plt.show()


def align_radian_offset_to_zero(data_dict, key_name='output_PVA_heading_offset'):
    """
    Align the circular mean of radian offset data to zero for each key in the dictionary 
    and combine all the aligned data.

    Parameters:
        data_dict (dict): A dictionary containing radian offset arrays as values.
                          The structure should be {key: [array]}.
        key_name (str): The key in the dictionary containing the radian offset arrays.

    Returns:
        np.ndarray: A combined array of all aligned radian offset data.
    """
    all_aligned_data = []

    # Loop through each key in the dictionary
    for key, radian_offset_list in data_dict[key_name].items():
        # Extract the radian offset array
        radian_offset = radian_offset_list[0]  # Assuming the array is the first item in the list
        
        # Align the mean angle to zero
        mean_angle = circmean(radian_offset, high=np.pi, low=-np.pi)  # Compute circular mean
        radian_offset_aligned = (radian_offset - mean_angle + np.pi) % (2 * np.pi) - np.pi  # Shift to align mean at zero
        
        # Append the aligned data to the combined list
        all_aligned_data.append(radian_offset_aligned)

    # Concatenate all aligned data into a single array
    all_aligned_data_combined = np.concatenate(all_aligned_data)

    return all_aligned_data_combined    


def plot_aligned_radian_offset_distribution(aligned_radian_offset_1, aligned_radian_offset_2=None, 
                                            label_1="EPG_shi_cl", label_2="empty_control", 
                                            color_1="Navy", color_2="grey", bins=30):
    """
    Plots the distribution of aligned radian offsets with circular variance.
    
    Parameters:
    - aligned_radian_offset_1: First dataset (required)
    - aligned_radian_offset_2: Second dataset (optional)
    - label_1: Label for first dataset
    - label_2: Label for second dataset (if provided)
    - color_1: Color for first dataset
    - color_2: Color for second dataset (if provided)
    - bins: Number of bins for histogram
    """

    # Define bins from -π to π
    bin_edges = np.linspace(-np.pi, np.pi, bins + 1)

    # Compute histogram for first dataset
    hist_1, _ = np.histogram(aligned_radian_offset_1, bins=bin_edges)
    hist_1 = hist_1 / hist_1.sum()  # Normalize

    # Compute bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot first dataset
    ax.fill_between(bin_centers, hist_1, alpha=0.4, color=color_1, label=f"{label_1} (Circ. Var: {circvar(aligned_radian_offset_1, high=np.pi, low=-np.pi):.2f})")

    # If second dataset is provided, plot it
    if aligned_radian_offset_2 is not None:
        hist_2, _ = np.histogram(aligned_radian_offset_2, bins=bin_edges)
        hist_2 = hist_2 / hist_2.sum()  # Normalize
        ax.fill_between(bin_centers, hist_2, alpha=0.4, color=color_2, label=f"{label_2} (Circ. Var: {circvar(aligned_radian_offset_2, high=np.pi, low=-np.pi):.2f})")

    # Formatting
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(0, max(hist_1.max(), hist_2.max() if aligned_radian_offset_2 is not None else 0) * 1.1)  # Adjust y-limit
    ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
    ax.set_xlabel("Radian Offset", fontsize=14)
    ax.set_ylabel("Proportion", fontsize=14)
    ax.legend(fontsize=12, loc="upper right")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title("Comparison of Aligned Radian Offset Distributions", fontsize=16, fontweight='bold')

    # Show plot
    plt.tight_layout()
    plt.show()    
    
    
    
    
def get_PVA_heading_offset_and_turning_speed_distribution(directory,separate_bridge,normalize_PVA_strength,signal_way):
    
    # Part 1: Create and add each DataFrame to the dictionary
    count = 0
    output_pooled_dictionary = {}
    output_pooled_dictionary['fly_trial_info'] = []
    output_pooled_dictionary['angular_velocity_distribution'] = {}
    
    if separate_bridge == True:
        output_pooled_dictionary['output_PVA_heading_offset_left_bridge'] = {}
        output_pooled_dictionary['circular_variance_left_bridge'] = []
        output_pooled_dictionary['circular_variance_slide_window_during_walking_left_bridge'] = []  
        output_pooled_dictionary['output_PVA_heading_offset_right_bridge'] = {}
        output_pooled_dictionary['circular_variance_right_bridge'] = []
        output_pooled_dictionary['circular_variance_slide_window_during_walking_right_bridge'] = []  
    else:
        output_pooled_dictionary['output_PVA_heading_offset'] = {}
        output_pooled_dictionary['circular_variance'] = []
        output_pooled_dictionary['circular_variance_slide_window_during_walking'] = []
        output_pooled_dictionary['bump_amplitude_distribution'] = {}    


    
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
        Angular_velocity = current_file['Angular_Velocity'].values
        Unwrapped_heading = current_file['Unwrapped_Bar_Position/Heading'].values
        Wrapped_heading = current_file['Bar_Position/Heading'].values
        
        #Get persistence period
        persistence_stop_index_and_length = find_stop_period_on_heading(head_velocity_array = Angular_velocity,degree_of_tolerance =15,shortest_stopFrame=int(np.ceil(3/volume_time)))   
        #Create a index array that indicates whether the current frame is stop/active stop index = 0, active index =1
        behavior_state_frame_index = np.ones(len(Angular_velocity))
        for current_index in range(len(persistence_stop_index_and_length)):
            start_index = persistence_stop_index_and_length[current_index,0]-persistence_stop_index_and_length[current_index,1]+1
            end_index = persistence_stop_index_and_length[current_index,0]
            currrent_stop_duration = end_index - start_index +1
            behavior_state_frame_index[start_index:end_index+1] = [0] * currrent_stop_duration
            
        if separate_bridge == True:
            # ----- split bridges -----
            columns_left = [f'Raw_F_{i}' for i in range(1, 9)]
            columns_right = [f'Raw_F_{i}' for i in range(9, 17)]
            raw_data_left_ROI = np.column_stack([current_file[col].to_numpy() for col in columns_left])
            raw_data_right_ROI = np.column_stack([current_file[col].to_numpy() for col in columns_right])
            
            if signal_way == "z":
                signal_left_bridge = zscore(raw_data_left_ROI, axis=0)
                signal_right_bridge = zscore(raw_data_right_ROI, axis=0)            
            elif signal_way == "normalize":
                dF_F_array_left = get_dff_array(raw_F_array = raw_data_left_ROI,ROI_num =8,F_zero_cutoff =0.05,if_plot =0)
                dF_F_array_right = get_dff_array(raw_F_array = raw_data_right_ROI,ROI_num =8,F_zero_cutoff =0.05,if_plot =0)
                signal_left_bridge = normalizing_dff_array(dF_F_array_left,ROI_num= 8, normalize_cutoff= 0.95, if_plot = 0)
                signal_right_bridge = normalizing_dff_array(dF_F_array_right,ROI_num= 8, normalize_cutoff= 0.95, if_plot = 0)
            else:
                raise ValueError("signal_way must be 'z' or 'normalize'")
            
            if normalize_PVA_strength == True:
                PVA_radian_array_left, PVA_strength_left = PVA_radian_calcul_norm(signal_left_bridge,volume_cycle,8, norm="contrast")
                PVA_radian_array_right, PVA_strength_right = PVA_radian_calcul_norm(signal_right_bridge,volume_cycle,8, norm="contrast")
            else:
                PVA_radian_array_left, PVA_strength_left = PVA_radian_calcul(signal_left_bridge, volume_cycle,8)
                PVA_radian_array_right, PVA_strength_right = PVA_radian_calcul(signal_right_bridge, volume_cycle,8)
        
            #Part 3.1: Get radian offset information
            radian_offset_current_left = offset_calculation(Wrapped_heading, PVA_radian_array_left, IfRadian = True)
            radian_offset_current_right = offset_calculation(Wrapped_heading, PVA_radian_array_right, IfRadian = True)

            radian_offset_no_stopping_period_current_left = []
            radian_offset_no_stopping_period_current_right = []
            for i in range(len (radian_offset_current)):
                if np.abs(Angular_velocity[i]) >= 0.26:
                    radian_offset_no_stopping_period_current_left.append(radian_offset_current_left[i])
                    radian_offset_no_stopping_period_current_right.append(radian_offset_current_right[i])
            circular_va_left = circvar(radian_offset_no_stopping_period_current_left,high=np.pi, low= -np.pi)
            circular_va_right = circvar(radian_offset_no_stopping_period_current_right,high=np.pi, low= -np.pi)
            output_pooled_dictionary['circular_variance_left_bridge'].append(circular_va_left)
            output_pooled_dictionary['circular_variance_right_bridge'].append(circular_va_right)
        
        
            #3.2.Store the flyinformation if wants to fly-by-fly analysis later on
            single_trial_info = single_df.split("-")
            flytrial = [single_trial_info[0], single_trial_info[1]]
            output_pooled_dictionary['fly_trial_info'].append(single_trial_info[0])
            key_for_heading_PVA_offset = tuple(flytrial)
            if  key_for_heading_PVA_offset not in output_pooled_dictionary['output_PVA_heading_offset_left_bridge']:
                    output_pooled_dictionary['output_PVA_heading_offset_left_bridge'][key_for_heading_PVA_offset] = []
                    output_pooled_dictionary['output_PVA_heading_offset_right_bridge'][key_for_heading_PVA_offset] = []
            output_pooled_dictionary['output_PVA_heading_offset_left_bridge'][key_for_heading_PVA_offset].append(radian_offset_no_stopping_period_current_left)
            output_pooled_dictionary['output_PVA_heading_offset_right_bridge'][key_for_heading_PVA_offset].append(radian_offset_no_stopping_period_current_right)
            if  key_for_heading_PVA_offset not in output_pooled_dictionary['angular_velocity_distribution']:
                output_pooled_dictionary['angular_velocity_distribution'][key_for_heading_PVA_offset] = []
            output_pooled_dictionary['angular_velocity_distribution'][key_for_heading_PVA_offset].append(Angular_velocity)
        
        
            # 3.3: Get Circular Variance in a sliding window fashion 
            #5s_ window
            window_for_cir_var = 5
            sliding_win_cir_var_array_left = sliding_window_circular_variance(radian_offset_current_left,behavior_state_frame_index,PVA_strength_left, window_for_cir_var,volume_time,step_size_frames=5)
            sliding_win_cir_var_array_right = sliding_window_circular_variance(radian_offset_current_right,behavior_state_frame_index,PVA_strength_right, window_for_cir_var,volume_time,step_size_frames=5)
            output_pooled_dictionary['circular_variance_slide_window_during_walking_right_bridge'].append(sliding_win_cir_var_array_right)
            output_pooled_dictionary['circular_variance_slide_window_during_walking_left_bridge'].append(sliding_win_cir_var_array_left)
            
            count = count + 1
            
        # If combine the bridge
        else:
            #Different  way of calculating the bump signal (and PVA)
            if signal_way == "z":       
                signal = np.array([current_file[f'Z_score_Roi_{i}'] for i in range(1, 9)]).T   
            elif signal_way == "normalize":   
                signal = np.array([current_file[f'dFF_Roi_{i}'] for i in range(1, 9)]).T
            else:
                raise ValueError("signal_way must be 'z' or 'normalize'")

            
            if normalize_PVA_strength == True:
                PVA_radian_array, PVA_strength = PVA_radian_calcul_norm(signal,volume_cycle,8, norm="contrast") 
            else:
                PVA_radian_array, PVA_strength = PVA_radian_calcul(signal, volume_cycle,8)
                        
            bump_amplitude = calcualteBumpAmplitude(signal)
         
            #Part 3.1. Get radian offset information
            radian_offset_current = offset_calculation(Wrapped_heading, PVA_radian_array, IfRadian = True)
            radian_offset_no_stopping_period_current = []
            for i in range(len (radian_offset_current)):
                if np.abs(Angular_velocity[i]) >= 0.26:
                    radian_offset_no_stopping_period_current.append(radian_offset_current[i])
            circular_va = circvar(radian_offset_no_stopping_period_current,high=np.pi, low= -np.pi)
            output_pooled_dictionary['circular_variance'].append(circular_va)
                                                  
            
            
            # 3.2.Get Circular Variance in a sliding window fashion 
            #5s_ window
            window_for_cir_var = 5
            sliding_win_cir_var_array = sliding_window_circular_variance(radian_offset_current,behavior_state_frame_index,PVA_strength, window_for_cir_var,volume_time,step_size_frames=5)
            output_pooled_dictionary['circular_variance_slide_window_during_walking'].append(sliding_win_cir_var_array)


            #3.3.Store the flyinformation if wants to fly-by-fly analysis later on
            single_trial_info = single_df.split("-")
            flytrial = [single_trial_info[0], single_trial_info[1]]
            output_pooled_dictionary['fly_trial_info'].append(single_trial_info[0])
            key = single_trial_info[0]
            key_for_heading_PVA_offset = tuple(flytrial)
            if  key_for_heading_PVA_offset not in output_pooled_dictionary['output_PVA_heading_offset']:
                    output_pooled_dictionary['output_PVA_heading_offset'][key_for_heading_PVA_offset] = []
            output_pooled_dictionary['output_PVA_heading_offset'][key_for_heading_PVA_offset].append(radian_offset_no_stopping_period_current)
            if  key_for_heading_PVA_offset not in output_pooled_dictionary['angular_velocity_distribution']:
                    output_pooled_dictionary['angular_velocity_distribution'][key_for_heading_PVA_offset] = []
            output_pooled_dictionary['angular_velocity_distribution'][key_for_heading_PVA_offset].append(Angular_velocity)
            if  key_for_heading_PVA_offset not in output_pooled_dictionary['bump_amplitude_distribution']:
                    output_pooled_dictionary['bump_amplitude_distribution'][key_for_heading_PVA_offset] = []
            output_pooled_dictionary['bump_amplitude_distribution'][key_for_heading_PVA_offset].append(bump_amplitude)


            count = count + 1

            
            
           
        
        
        
     
    return output_pooled_dictionary
    