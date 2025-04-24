import numpy as np
import pandas as pd
import os
from os.path import sep
from scipy.signal import correlate,correlation_lags
from scipy.stats import zscore
from scipy.ndimage import gaussian_filter1d
from scipy.stats import binned_statistic
from matplotlib import pyplot as plt
from fly2p_function_TQ.imaging_2p_fixed_heading_period_analysis import find_stop_period_on_heading,stopping_period_signal_decay







def find_qualified_stopping_period_single_ROI (volume_time,stopping_array,minimum_frame_length):
    #Find qualified stopping index (must meet the minimimun length of demand and must have 1s of active period before)
    persistence_stop_index_and_length_qualified_index = []

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
    
    
    return persistence_stop_index_and_length_qualified

def run_fixed_heading_period_analysis_across_trial_single_ROI(directory,dual_imaging,genotype,trial_condition,Signal_stopping_duration):
    # Part 1: Create and add each DataFrame to the dictionary
    count = 0
    output_pooled_dictionary = {}
    output_pooled_dictionary['stopping_Angular_Speed_pooled'] = pd.DataFrame()
    output_pooled_dictionary['stopping_Forward_Speed_pooled'] = pd.DataFrame()
    output_pooled_dictionary['output_flytrial'] = []
    output_pooled_dictionary['output_flytrial_for_qualified_stop'] = []
    output_pooled_dictionary['stopping_period_z_bg_subtracted_pooled'] =  pd.DataFrame()
    output_pooled_dictionary['stopping_period_F_bg_subtracted_pooled'] = pd.DataFrame()
    output_pooled_dictionary['stopping_period_F_pooled'] = pd.DataFrame()
    output_pooled_dictionary['stopping_period_z_pooled'] = pd.DataFrame()
        
        
        
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
        Unwrapped_heading = current_file['Unwrapped_Bar_Position/Heading'].values
        Wrapped_heading = current_file['Bar_Position/Heading'].values
        Angular_velocity = current_file['Angular_Velocity'].values
        Angular_speed_degrees = np.abs(Angular_velocity) * 180/np.pi
        Forward_velocity = current_file['Forward_Velocity'].values
        Forward_speed_degrees = np.abs(Forward_velocity) * 180/np.pi
        F_background_subtracted = current_file['F_background_subtracted'].values
        Raw_F = current_file['Raw_F'].values
        zscore_signal_background_subtracted = zscore(F_background_subtracted)
        zscore_signal_raw_F = zscore(Raw_F)
        
        #Get persistence period
        persistence_stop_index_and_length = find_stop_period_on_heading(head_velocity_array = Angular_velocity,degree_of_tolerance =15,shortest_stopFrame=int(np.ceil(3/volume_time)))
        
        
        #Part3: get signal during the stopping period
        # 3.1 Get sifgnal dynamics during the stopping period
        signal_stopping_duration = Signal_stopping_duration
        minimum_frame_length = int(np.ceil(signal_stopping_duration/volume_time))
        
        #200ms bin size
        signal_stopping_bin_size = 0.2
        
        #Get qualified stopping period for decay analysis 
        qualified_stopping_index_and_length = find_qualified_stopping_period_single_ROI(volume_time =volume_time,stopping_array=persistence_stop_index_and_length,minimum_frame_length=int(np.ceil(signal_stopping_duration/volume_time)))
        
        active_period_before_len =1
        
        
        stopping_period_z_bg_subtracted_current = stopping_period_signal_decay(volume_time, signal_stopping_duration,signal_stopping_bin_size, zscore_signal_background_subtracted, qualified_stopping_index_and_length,active_period_before_len)
        stopping_period_F_bg_subtracted_current = stopping_period_signal_decay(volume_time, signal_stopping_duration,signal_stopping_bin_size, F_background_subtracted, qualified_stopping_index_and_length,active_period_before_len)
        stopping_period_F_current = stopping_period_signal_decay(volume_time, signal_stopping_duration,signal_stopping_bin_size, Raw_F, qualified_stopping_index_and_length,active_period_before_len)
        stopping_period_z_current = stopping_period_signal_decay(volume_time, signal_stopping_duration,signal_stopping_bin_size, zscore_signal_raw_F, qualified_stopping_index_and_length,active_period_before_len)
        stopping_period_Angular_speed_current = stopping_period_signal_decay(volume_time, signal_stopping_duration,signal_stopping_bin_size, Angular_speed_degrees, qualified_stopping_index_and_length,active_period_before_len)
        stopping_period_Forward_speed_current = stopping_period_signal_decay(volume_time, signal_stopping_duration,signal_stopping_bin_size, Forward_speed_degrees, qualified_stopping_index_and_length,active_period_before_len)
        
        
        stopping_period_z_bg_subtracted_current = pd.DataFrame(stopping_period_z_bg_subtracted_current.transpose())
        stopping_period_F_bg_subtracted_current = pd.DataFrame(stopping_period_F_bg_subtracted_current.transpose())
        stopping_period_F_current = pd.DataFrame(stopping_period_F_current.transpose())
        stopping_period_z_current = pd.DataFrame(stopping_period_z_current.transpose())
        stopping_period_Angular_speed_current = pd.DataFrame(stopping_period_Angular_speed_current.transpose())               
        stopping_period_Forward_speed_current = pd.DataFrame(stopping_period_Forward_speed_current.transpose())
        
        
        #3.9 Data storage
        
        
        #Store the flyinformation if wants to fly-by-fly analysis later on
        single_trial_info = single_df.split("-")
        flytrial = [single_trial_info[0], single_trial_info[1]]
        output_pooled_dictionary['output_flytrial'].append(single_trial_info[0])
        #just in case some trials do not have stopping period
        for i in range(stopping_period_F_current.shape[1]):
            output_pooled_dictionary['output_flytrial_for_qualified_stop'].append(single_trial_info[0])
        
        
        
        if count == 0:
            output_pooled_dictionary['stopping_period_z_bg_subtracted_pooled'] =  stopping_period_z_bg_subtracted_current
            output_pooled_dictionary['stopping_period_F_bg_subtracted_pooled'] = stopping_period_F_bg_subtracted_current
            output_pooled_dictionary['stopping_period_F_pooled'] = stopping_period_F_current
            output_pooled_dictionary['stopping_period_z_pooled'] = stopping_period_z_current
            output_pooled_dictionary['stopping_Angular_Speed_pooled'] = stopping_period_Angular_speed_current 
            output_pooled_dictionary['stopping_Forward_Speed_pooled'] = stopping_period_Forward_speed_current
        else:
            output_pooled_dictionary['stopping_period_z_bg_subtracted_pooled'] = pd.concat([ output_pooled_dictionary['stopping_period_z_bg_subtracted_pooled'],stopping_period_z_bg_subtracted_current],ignore_index=True, axis =1)
            output_pooled_dictionary['stopping_period_F_bg_subtracted_pooled'] = pd.concat([output_pooled_dictionary['stopping_period_F_bg_subtracted_pooled'],stopping_period_F_bg_subtracted_current],ignore_index=True, axis =1)
            output_pooled_dictionary['stopping_period_F_pooled'] = pd.concat([output_pooled_dictionary['stopping_period_F_pooled'],stopping_period_F_current],ignore_index=True, axis =1)
            output_pooled_dictionary['stopping_period_z_pooled'] = pd.concat([ output_pooled_dictionary['stopping_period_z_pooled'],stopping_period_z_current],ignore_index=True, axis =1)
            output_pooled_dictionary['stopping_Angular_Speed_pooled'] = pd.concat([ output_pooled_dictionary['stopping_Angular_Speed_pooled'],stopping_period_Angular_speed_current],ignore_index=True, axis =1)
            output_pooled_dictionary['stopping_Forward_Speed_pooled'] = pd.concat([ output_pooled_dictionary['stopping_Forward_Speed_pooled'],stopping_period_Forward_speed_current],ignore_index=True, axis =1)
            
            
            
            
        count = count + 1
        
        
        
        
    return output_pooled_dictionary





def process_signal_speed_correlation_single_ROI(pooled_directory, lag_duration):
    """
    Processes fly data from a directory of .csv files and organizes the results by fly ID.

    Parameters:
        pooled_directory (str): Path to the directory containing the .csv files.
        ROI_type (int): Type of ROI (not used directly in current implementation but kept for compatibility).
        lag_duration (float): Lag duration in seconds to shift signals.

    Returns:
        dict: A dictionary with data for each fly.
    """
    fly_data = {}

    for filename in os.listdir(pooled_directory):
        if filename.endswith('.csv'):
            # Extract the fly identifier (e.g., 'TQfly109-001.csv' -> 'TQfly109')
            fly_id = filename.split('-')[0]
            file_path = os.path.join(pooled_directory, filename)
            current_file = pd.read_csv(file_path)

            # Initialize dictionary for a new fly ID
            if fly_id not in fly_data:
                fly_data[fly_id] = {
                    'lagged_z_bg_subtracted_smooth': [],
                    'Forward_speed': [],
                    'Angular_speed': []
                }

            # Get time and signal parameters
            volume_cycle = len(current_file)
            volume_time = current_file['Time_Stamp'][1]
            volume_rate = 1 / volume_time

            # Calculate time array
            time_array_imaging = np.arange(volume_cycle) / volume_rate

            # Extract relevant columns and perform calculations
            Unwrapped_heading = current_file['Unwrapped_Bar_Position/Heading'].values
            Wrapped_heading = current_file['Bar_Position/Heading'].values
            Angular_velocity = current_file['Angular_Velocity'].values
            Angular_speed_degrees = np.abs(Angular_velocity) * 180 / np.pi
            Forward_velocity = current_file['Forward_Velocity'].values
            Forward_speed_degrees = np.abs(Forward_velocity) * 180 / np.pi
            F_background_subtracted = current_file['F_background_subtracted'].values
            Raw_F = current_file['Raw_F'].values

            zscore_signal_background_subtracted = zscore(gaussian_filter1d(F_background_subtracted, sigma=5))

            # Calculate cross-correlation (not directly used but retained for completeness)
            cross_corr = correlate(Angular_speed_degrees, zscore_signal_background_subtracted, mode='full')
            lags = correlation_lags(len(Angular_speed_degrees), len(zscore_signal_background_subtracted), mode="full")

            # Determine lag in frames
            lag = int(lag_duration / volume_time)

            # Apply the lag to adjust the signal
            lagged_z_smooth_for_speed_cor = zscore_signal_background_subtracted[-lag:]
            Angular_speed_degrees = Angular_speed_degrees[:len(lagged_z_smooth_for_speed_cor)]
            Forward_speed_degrees = Forward_speed_degrees[:len(lagged_z_smooth_for_speed_cor)]

            # Append or concatenate new session data to existing fly data
            if len(fly_data[fly_id]['lagged_z_bg_subtracted_smooth']) > 0:
                fly_data[fly_id]['lagged_z_bg_subtracted_smooth'] = np.concatenate((fly_data[fly_id]['lagged_z_bg_subtracted_smooth'], lagged_z_smooth_for_speed_cor))
                fly_data[fly_id]['Angular_speed'] = np.concatenate((fly_data[fly_id]['Angular_speed'], Angular_speed_degrees))
                fly_data[fly_id]['Forward_speed'] = np.concatenate((fly_data[fly_id]['Forward_speed'], Forward_speed_degrees))
            else:
                fly_data[fly_id]['lagged_z_bg_subtracted_smooth'] = lagged_z_smooth_for_speed_cor
                fly_data[fly_id]['Angular_speed'] = Angular_speed_degrees
                fly_data[fly_id]['Forward_speed'] = Forward_speed_degrees

    return fly_data



def correaltion_across_speed_range(flydata, bin_Width, bin_Edges_low, bin_Edges_up,x_column,y_column):
    # Define the bin edges with a bin width of 5
    bin_width = bin_Width
    bin_edges = np.arange(bin_Edges_low, bin_Edges_up + bin_width, bin_width)

    # Prepare a DataFrame to store results
    binned_results_df = pd.DataFrame(index=bin_edges[:-1] + 0.5 * bin_width)
    
    # Process each fly's data
    for fly_id, data in flydata.items():
        x_data = data[x_column]
        y_data = data[y_column]

        # Calculate binned statistics
        binned_stats = binned_statistic(x_data, y_data, statistic='mean', bins=bin_edges)
        binned_counts = binned_statistic(x_data, y_data, statistic='count', bins=bin_edges).statistic

        # Check the number of values in each bin and raise warnings if below threshold
        
        warning_threshold = 10
        for bin_idx, count in enumerate(binned_counts):
            if count < warning_threshold:
                print(f"Warning: For fly {fly_id}, bin {bin_edges[bin_idx]}-{bin_edges[bin_idx+1]} has only {int(count)} values, which is below the threshold of {warning_threshold}.")

        # Extract the mean bump width per bin
        mean_Bump_width_per_bin = binned_stats.statistic

        # Store results in the DataFrame under the fly's identifier
        binned_results_df[fly_id] = mean_Bump_width_per_bin

        # Plotting (optional, can be commented out if only the DataFrame is needed)
        valid_bins = ~np.isnan(mean_Bump_width_per_bin)
        bin_centers = binned_stats.bin_edges[:-1] + 0.5 * bin_width
        correlation_coefficient = np.corrcoef(bin_centers[valid_bins], mean_Bump_width_per_bin[valid_bins])[0, 1]

        plt.figure(figsize=(15, 5))
        plt.bar(bin_centers[valid_bins], mean_Bump_width_per_bin[valid_bins], width=bin_width, align='center', edgecolor='black')
        plt.xlim(0, np.max(bin_edges))
        plt.xlabel(f'{x_column} (°/sec)')
        plt.ylabel(f'{y_column} (units)')
        plt.title(f'{fly_id} - Binned Mean {y_column} vs. {x_column}\nCorrelation: {correlation_coefficient:.2f}')
        plt.show()
    
    return binned_results_df
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        