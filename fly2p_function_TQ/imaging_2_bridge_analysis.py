import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import zscore
import os
from os.path import sep
from scipy.ndimage import gaussian_filter1d
from sklearn.utils import resample
from scipy.stats import binned_statistic
from fly2p_function_TQ.imaging_2p_fictrac_imaging_alignment import moving_wrapped_plot_by_offset,fictrack_signal_decoding,offset_calculation 
from fly2p_function_TQ.imaging_2p_PVA_functions import calcualteBumpAmplitude, calcualteBumpAmplitude_V2_green,PVA_radian_calcul_norm, calcualteBumpAmplitude_V3,calculateBumpWidth_v1,PVA_radian_calcul,PVA_radian_to_angle,calcualteBumpAmplitude_V4,PVAangleToRoi,strong_PVA_duration,strong_PVA_index,get_behavior_state_of_strong_PVA,get_bump_shape_at_strong_signal,get_PVA_at_strong_signal,strong_signal_index,weak_signal_index,get_bump_shape_at_strong_signal_various_speed
from fly2p_function_TQ.imaging_2p_fixed_heading_period_analysis import find_stop_period_on_heading
from fly2p_function_TQ.imaging_2p_preprocessing import get_dff_array,normalizing_dff_array



def normalizing_dff_array_lyu(f_input, ROI_num, if_plot):
    dF_F_array_normalized_output = np.zeros((len(f_input), ROI_num))

    for current_ROI in range(ROI_num):
        # Get all values for this ROI
        roi_values = f_input[:, current_ROI]
        
        # Compute F0 and Fmax for this ROI
        F0 = roi_values[roi_values <= np.quantile(roi_values, 0.05)].mean()
        Fmax = roi_values[roi_values >= np.quantile(roi_values, 0.95)].mean()
        
        # Normalize
        dF_F_array_normalized_output[:, current_ROI] = (roi_values - F0) / (Fmax - F0)
    
    # Optional plotting
    if if_plot == 1:
        if ROI_num > 1:
            fig, axs = plt.subplots(ROI_num, 1, figsize=(13, 12))
            for i in range(ROI_num):
                axs[i].plot(dF_F_array_normalized_output[:, i])
            fig.supylabel('dF/F-Normalized', fontsize=20)
            plt.xlabel('Frame Number', fontsize=20)
            plt.show()
        else:
            plt.figure(figsize=(25, 7))
            plt.plot(dF_F_array_normalized_output)
            plt.ylabel('dF/F-Normalized', fontsize=20)
            plt.xlabel('Frame Number', fontsize=20)
            plt.show()
    
    return dF_F_array_normalized_output



def get_2_bridge_data_for_analysis(directory):
    # Part 1: Create a data dictionary
    fly_data = {}
    
        
    #Part 2:import data
    # import data
    for single_df in os.listdir(directory):
        fly_id = "-".join(single_df.split("-")[:2])
        file_path = os.path.join(directory, single_df)
        current_file = pd.read_csv(file_path)
        
        
         # Check if the fly_id already has an entry in the dictionary
        if fly_id not in fly_data:
            fly_data[fly_id] = {
                'z_score_data_left_bridge':  pd.DataFrame(),
                'z_score_data_right_bridge': pd.DataFrame(),
                'dff_data_left_bridge':  pd.DataFrame(),
                'dff_data_right_bridge': pd.DataFrame(),
                'dff_data_norm_left_bridge':  pd.DataFrame(),
                'dff_data_norm_right_bridge': pd.DataFrame(),
                'dff_data_norm_TQ_left_bridge':  pd.DataFrame(),
                'dff_data_norm_TQ_right_bridge': pd.DataFrame(),
                'Angular_velocity': [],
                'Angular_speed': [],
                'PVA_left_bridge':[],
                'PVA_right_bridge' :[],
                'PVA_strength_left_bridge' : [],
                'PVA_strength_right_bridge': [],
                'PVA_ROI_left_bridge' : [],
                'PVA_ROI_right_bridge': [],
                'FWHM_left_bridge':[],
                'FWHM_right_bridge':[],
                'FWHM_dff_left_bridge':[],
                'FWHM_dff_right_bridge':[]
            }
        
              
  
        #Get time info
        volume_cycle = len(current_file )
        volume_time = current_file['Time_Stamp'][1]
        volume_rate = 1/volume_time
        time_array_imaging = np.arange(volume_cycle)/volume_rate
        #Get necessary parameters
        Angular_velocity = current_file['Angular_Velocity'].values
        integrated_x = current_file['Integrated_x'].values
        Unwrapped_heading = current_file['Unwrapped_Bar_Position/Heading'].values
        Wrapped_heading = current_file['Bar_Position/Heading'].values
        integrated_x_unwrapped =  fictrack_signal_decoding(integrated_x,time_array_imaging, 10, already_radian = True)
        Forward_velocity = np.gradient(integrated_x_unwrapped)/volume_time
        Forward_speed_radian = np.abs(Forward_velocity)
        Forward_speed_degrees =Forward_speed_radian * 180/np.pi
        Angular_speed_degrees =  np.abs(Angular_velocity) * 180/np.pi
        #Get persistence period
        persistence_stop_index_and_length = find_stop_period_on_heading(head_velocity_array = Angular_velocity,degree_of_tolerance =15,shortest_stopFrame=int(np.ceil(3/volume_time)))
        
        #get raw F data
        columns_left = ['Raw_F_1', 'Raw_F_2', 'Raw_F_3', 'Raw_F_4', 'Raw_F_5', 'Raw_F_6', 'Raw_F_7', 'Raw_F_8']
        columns_right = ['Raw_F_9','Raw_F_10','Raw_F_11','Raw_F_12','Raw_F_13','Raw_F_14','Raw_F_15','Raw_F_16']
        raw_data_left_ROI = [current_file[col].to_numpy() for col in columns_left]
        raw_data_left_ROI = np.array(raw_data_left_ROI).T
        raw_data_right_ROI = [current_file[col].to_numpy() for col in columns_right]
        raw_data_right_ROI = np.array(raw_data_right_ROI).T
        
        dF_F_array_left = get_dff_array(raw_F_array = raw_data_left_ROI, ROI_num = 8, F_zero_cutoff = 0.05, if_plot = 0)
        dF_F_array_right = get_dff_array(raw_F_array = raw_data_right_ROI, ROI_num = 8, F_zero_cutoff = 0.05, if_plot = 0)
        dF_F_array_norm_left = normalizing_dff_array_lyu(f_input = raw_data_left_ROI, ROI_num = 8, if_plot = 0)
        dF_F_array_norm_right = normalizing_dff_array_lyu(f_input = raw_data_right_ROI, ROI_num = 8, if_plot = 0)
        dF_F_array_norm_TQ_left = normalizing_dff_array(df_f_input= dF_F_array_left,ROI_num=8, normalize_cutoff=0.95, if_plot=0)
        dF_F_array_norm_TQ_right = normalizing_dff_array(df_f_input=dF_F_array_right,ROI_num=8, normalize_cutoff=0.95, if_plot=0)
        z_score_data_left_bridge = zscore(raw_data_left_ROI)
        z_score_data_right_bridge = zscore(raw_data_right_ROI)
        
        PVA_left_bridge, PVA_left_bridge_strength = PVA_radian_calcul_norm(z_score_data_left_bridge, volume_cycle,8,norm="contrast")
        PVA_right_bridge, PVA_right_bridge_strength = PVA_radian_calcul_norm(z_score_data_right_bridge, volume_cycle,8,norm="contrast")
        PVA_angle_array_left_bridge = PVA_radian_to_angle(PVA_left_bridge)
        PVA_angle_array_right_bridge = PVA_radian_to_angle(PVA_right_bridge)
        PVA_ROI_assigned_left_bridge = PVAangleToRoi(PVA_angle_array_left_bridge)
        PVA_ROI_assigned_right_bridge = PVAangleToRoi(PVA_angle_array_right_bridge)
        
        FWHM_left_bridge = calculateBumpWidth_v1(z_score_data_left_bridge,8)
        FWHM_right_bridge = calculateBumpWidth_v1(z_score_data_right_bridge,8)
        FWHM_dff_left_bridge = calculateBumpWidth_v1(dF_F_array_left,8)
        FWHM_dff_right_bridge = calculateBumpWidth_v1(dF_F_array_right,8)
        
        
        
 
        if len(fly_data[fly_id]['z_score_data_left_bridge']) >  0 :
            fly_data[fly_id]['z_score_data_left_bridge'] =pd.concat([fly_data[fly_id]['z_score_data_left_bridge'],pd.DataFrame(z_score_data_left_bridge)],ignore_index=True, axis =1)
            fly_data[fly_id]['z_score_data_right_bridge'] =pd.concat([fly_data[fly_id]['z_score_data_right_bridge'],pd.DataFrame(z_score_data_right_bridge)],ignore_index=True, axis =1)
            fly_data[fly_id]['dff_data_left_bridge'] =pd.concat([fly_data[fly_id]['dff_data_left_bridge'],pd.DataFrame(dF_F_array_left)],ignore_index=True, axis =1)
            fly_data[fly_id]['dff_data_right_bridge'] =pd.concat([fly_data[fly_id]['dff_data_right_bridge'],pd.DataFrame(dF_F_array_right)],ignore_index=True, axis =1)
            fly_data[fly_id]['dff_data_norm_left_bridge'] =pd.concat([fly_data[fly_id]['dff_data_norm_left_bridge'],pd.DataFrame(dF_F_array_norm_left)],ignore_index=True, axis =1)
            fly_data[fly_id]['dff_data_norm_right_bridge'] =pd.concat([fly_data[fly_id]['dff_data_norm_right_bridge'],pd.DataFrame(dF_F_array_norm_right)],ignore_index=True, axis =1)
            fly_data[fly_id]['dff_data_norm_TQ_left_bridge'] =pd.concat([fly_data[fly_id]['dff_data_norm_TQ_left_bridge'],pd.DataFrame(dF_F_array_norm_TQ_left)],ignore_index=True, axis =1)
            fly_data[fly_id]['dff_data_norm_TQ_right_bridge'] =pd.concat([fly_data[fly_id]['dff_data_norm_TQ_right_bridge'],pd.DataFrame(dF_F_array_norm_TQ_right)],ignore_index=True, axis =1)
            fly_data[fly_id]['Angular_velocity'] = pd.concat([fly_data[fly_id]['Angular_velocity'],Angular_velocity])
            fly_data[fly_id]['Angular_speed'] = pd.concat([fly_data[fly_id]['Angular_speed'],Angular_speed_degrees])
            fly_data[fly_id]['PVA_left_bridge'] = pd.concat([fly_data[fly_id]['PVA_left_bridge'],PVA_angle_array_left_bridge])
            fly_data[fly_id]['PVA_right_bridge'] = pd.concat([fly_data[fly_id]['PVA_right_bridge'],PVA_angle_array_right_bridge])
            fly_data[fly_id]['PVA_strength_left_bridge'] = pd.concat([fly_data[fly_id]['PVA_strength_left_bridge'],PVA_left_bridge_strength ])
            fly_data[fly_id]['PVA_strength_right_bridge'] = pd.concat([fly_data[fly_id]['PVA_strength_right_bridge'],PVA_right_bridge_strength] )
            fly_data[fly_id]['PVA_ROI_left_bridge'] = pd.concat([fly_data[fly_id]['PVA_ROI_left_bridge'],PVA_ROI_assigned_left_bridge])
            fly_data[fly_id]['PVA_ROI_right_bridge'] = pd.concat([fly_data[fly_id]['PVA_ROI_right_bridge'],PVA_ROI_assigned_right_bridge])
            fly_data[fly_id]['FWHM_left_bridge'] = pd.concat([fly_data[fly_id]['FWHM_left_bridge'],FWHM_left_bridge])
            fly_data[fly_id]['FWHM_right_bridge'] = pd.concat([fly_data[fly_id]['FWHM_right_bridge'],FWHM_right_bridge])
            fly_data[fly_id]['FWHM_dff_left_bridge'] = pd.concat([fly_data[fly_id]['FWHM_dff_left_bridge'],FWHM_dff_left_bridge])
            fly_data[fly_id]['FWHM_dff_right_bridge'] = pd.concat([fly_data[fly_id]['FWHM_dff_right_bridge'],FWHM_dff_right_bridge])
            
        else:
            fly_data[fly_id]['z_score_data_left_bridge'] = pd.DataFrame(z_score_data_left_bridge)
            fly_data[fly_id]['z_score_data_right_bridge'] = pd.DataFrame(z_score_data_right_bridge)
            fly_data[fly_id]['dff_data_left_bridge'] = pd.DataFrame(dF_F_array_left)
            fly_data[fly_id]['dff_data_right_bridge'] = pd.DataFrame(dF_F_array_right)
            fly_data[fly_id]['dff_data_norm_left_bridge'] = pd.DataFrame(dF_F_array_norm_left)
            fly_data[fly_id]['dff_data_norm_right_bridge'] = pd.DataFrame(dF_F_array_norm_right)
            fly_data[fly_id]['dff_data_norm_TQ_left_bridge'] = pd.DataFrame(dF_F_array_norm_TQ_left)
            fly_data[fly_id]['dff_data_norm_TQ_right_bridge'] = pd.DataFrame(dF_F_array_norm_TQ_right)
            fly_data[fly_id]['Angular_velocity'] = Angular_velocity
            fly_data[fly_id]['Angular_speed'] = Angular_speed_degrees
            fly_data[fly_id]['PVA_left_bridge'] = PVA_angle_array_left_bridge
            fly_data[fly_id]['PVA_right_bridge'] = PVA_angle_array_right_bridge
            fly_data[fly_id]['PVA_strength_left_bridge'] = PVA_left_bridge_strength 
            fly_data[fly_id]['PVA_strength_right_bridge'] = PVA_right_bridge_strength 
            fly_data[fly_id]['PVA_ROI_left_bridge'] =  PVA_ROI_assigned_left_bridge
            fly_data[fly_id]['PVA_ROI_right_bridge'] =  PVA_ROI_assigned_right_bridge
            fly_data[fly_id]['FWHM_left_bridge'] =  FWHM_left_bridge
            fly_data[fly_id]['FWHM_right_bridge'] = FWHM_right_bridge
            fly_data[fly_id]['FWHM_dff_left_bridge'] =  FWHM_dff_left_bridge
            fly_data[fly_id]['FWHM_dff_right_bridge'] = FWHM_dff_right_bridge
            
            

        
        
    return fly_data



def get_bump_property_speed_correlation_for_analysis(directory, if_combine_2_bridge):
    
    # Part 1: Create a data dictionary
    fly_data = {}
    # Process each file in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            # Extract the fly identifier (assuming format like 'TQfly109-001.csv')
            fly_id = filename.split('-')[0]  # This splits the filename and takes the first part
            file_path = os.path.join(directory, filename)
            current_file = pd.read_csv(file_path)
        
        
        #Get time info
        volume_cycle = len(current_file )
        volume_time = current_file['Time_Stamp'][1]
        volume_rate = 1/volume_time
        time_array_imaging = np.arange(volume_cycle)/volume_rate
        gaussian_kernel_width_ms = 600
        gaussian_sigma = (gaussian_kernel_width_ms / 1000) / volume_time / 2.355
        #Get necessary parameters
        Angular_velocity = current_file['Angular_Velocity'].values
        Angular_speed_degrees =  np.abs(Angular_velocity*180/np.pi)
        Unwrapped_heading = current_file['Unwrapped_Bar_Position/Heading'].values
        Wrapped_heading = current_file['Bar_Position/Heading'].values

        
        if (if_combine_2_bridge == 0):
            # Check if the fly_id already has an entry in the dictionary
            if fly_id not in fly_data:
                fly_data[fly_id] = {
                    'lagged_Bump_amplitude_PVA_z_smooth': [],
                    'lagged_Bump_amplitude_opposite_z_smooth':[],
                    'lagged_Bump_amplitude_max_min_z_smooth': [],
                    'lagged_Bump_amplitude_two_max_average_z_smooth': [],
                    'lagged_PVA_strength': [],
                    'lagged_Bump_width_z' :[],
                    'Angular_speed_lagged' :[],
                }    


            columns = ['F_Roi_1', 'F_Roi_2', 'F_Roi_3', 'F_Roi_4', 'F_Roi_5', 'F_Roi_6', 'F_Roi_7', 'F_Roi_8']
            raw_data_ROI = [current_file[col].to_numpy() for col in columns]
            raw_data_ROI = np.array(raw_data_ROI).T
            z_score_data = zscore(raw_data_ROI)
            PVA, PVA_strength = PVA_radian_calcul_norm(z_score_data, volume_cycle,8,norm="contrast")
            FWHM = calculateBumpWidth_v1(z_score_data,8)
            
            #calculate bump amplitude by getting the normalized dff at current PVA angle                 
            #First way of calculating bump amplitude: Amplitude coming from the PVA-resided Glomerulus
            Bump_amplitude_PVA, Bump_amplitude_PVA_opposite = calcualteBumpAmplitude_V3(z_score_data,PVA)
            Bump_amplitude_smooth_for_speed_cor_PVA = gaussian_filter1d(Bump_amplitude_PVA, sigma =gaussian_sigma)
            Bump_amplitude_smooth_for_speed_cor_PVA_opposite = gaussian_filter1d(Bump_amplitude_PVA_opposite, sigma =gaussian_sigma)
        
                                                                                 
            #Second way of calculating the bump amplitude: Max-min signal at each time frame
            Bump_amplitude_max_min = calcualteBumpAmplitude(z_score_data)           
            Bump_amplitude_smooth_for_speed_cor_max_min = gaussian_filter1d(Bump_amplitude_max_min, sigma =gaussian_sigma)
                                                         

            #Third way of calculating the bump amplitude: Averaging two max glomerulus at each time frame (Green et.al 17)
            Bump_amplitude_two_max_average = calcualteBumpAmplitude_V2_green(z_score_data)
            Bump_amplitude_smooth_for_speed_cor_two_max_average = gaussian_filter1d(Bump_amplitude_two_max_average, sigma =gaussian_sigma)
               
            lag = int(-0.3/volume_time)                                                                     
                                                                                 
            #shift back the bump_amplitude or PVA strength by drop the lag frame at the beginning
            lagged_Bump_amplitude_smooth_for_speed_cor_PVA = Bump_amplitude_smooth_for_speed_cor_PVA[-lag:]
            lagged_Bump_amplitude_smooth_for_speed_cor_PVA_opposite = Bump_amplitude_smooth_for_speed_cor_PVA_opposite[-lag:]
         
            lagged_Bump_amplitude_smooth_for_speed_cor_max_min = Bump_amplitude_smooth_for_speed_cor_max_min[-lag:]
           
            lagged_Bump_amplitude_smooth_for_speed_cor_two_max_average = Bump_amplitude_smooth_for_speed_cor_two_max_average[-lag:]
           
            Angular_speed_degrees_lagged = Angular_speed_degrees[:len(lagged_Bump_amplitude_smooth_for_speed_cor_PVA)]
            PVA_strength_lagged = PVA_strength[-lag:]
            FWHM_lagged  = FWHM[-lag:]
            
            # Append or concatenate new session data to existing fly data
            if len(fly_data[fly_id]['lagged_Bump_amplitude_PVA_z_smooth']) > 0:
                fly_data[fly_id]['lagged_Bump_amplitude_PVA_z_smooth'] = np.concatenate((fly_data[fly_id]['lagged_Bump_amplitude_PVA_z_smooth'], lagged_Bump_amplitude_smooth_for_speed_cor_PVA))
                fly_data[fly_id]['lagged_Bump_amplitude_opposite_z_smooth'] = np.concatenate((fly_data[fly_id]['lagged_Bump_amplitude_opposite_z_smooth'], lagged_Bump_amplitude_smooth_for_speed_cor_PVA_opposite))
                fly_data[fly_id]['lagged_Bump_amplitude_max_min_z_smooth']  = np.concatenate((fly_data[fly_id]['lagged_Bump_amplitude_max_min_z_smooth'], lagged_Bump_amplitude_smooth_for_speed_cor_max_min))
                fly_data[fly_id]['lagged_Bump_amplitude_two_max_average_z_smooth']  = np.concatenate((fly_data[fly_id]['lagged_Bump_amplitude_two_max_average_z_smooth'], lagged_Bump_amplitude_smooth_for_speed_cor_two_max_average))


                fly_data[fly_id]['Angular_speed_lagged'] = np.concatenate((fly_data[fly_id]['Angular_speed_lagged'], Angular_speed_degrees_lagged))
                fly_data[fly_id]['lagged_PVA_strength'] = np.concatenate((fly_data[fly_id]['lagged_PVA_strength'], PVA_strength_lagged))             
                fly_data[fly_id]['lagged_Bump_width_z'] = np.concatenate((fly_data[fly_id]['lagged_Bump_width_z'], FWHM_lagged))
               



            else:
                fly_data[fly_id]['lagged_Bump_amplitude_PVA_z_smooth'] =  lagged_Bump_amplitude_smooth_for_speed_cor_PVA
                fly_data[fly_id]['lagged_Bump_amplitude_opposite_z_smooth'] =  lagged_Bump_amplitude_smooth_for_speed_cor_PVA_opposite
                fly_data[fly_id]['lagged_Bump_amplitude_max_min_z_smooth'] =  lagged_Bump_amplitude_smooth_for_speed_cor_max_min
                fly_data[fly_id]['lagged_Bump_amplitude_two_max_average_z_smooth'] =  lagged_Bump_amplitude_smooth_for_speed_cor_two_max_average                                                     
                fly_data[fly_id]['Angular_speed_lagged'] = Angular_speed_degrees_lagged
                fly_data[fly_id]['lagged_PVA_strength'] = PVA_strength_lagged                
                fly_data[fly_id]['lagged_Bump_width_z'] = FWHM_lagged
                
                                                                  
                                                                                 
                                                                                 
                                                                                 
                                                                                 
                                                                                 
        #if calculate bump amplitude/width separately for left/right bridge
        else:
            if fly_id not in fly_data:
                fly_data[fly_id] = {
                    'lagged_Bump_amplitude_PVA_z_smooth_left': [],
                    'lagged_Bump_amplitude_opposite_z_smooth_left':[],
                    'lagged_Bump_amplitude_max_min_z_smooth_left': [],
                    'lagged_Bump_amplitude_two_max_average_z_smooth_left': [],
                    'lagged_Bump_amplitude_PVA_z_smooth_right': [],
                    'lagged_Bump_amplitude_opposite_z_smooth_right':[],
                    'lagged_Bump_amplitude_max_min_z_smooth_right': [],
                    'lagged_Bump_amplitude_two_max_average_z_smooth_right': [],
                    'lagged_PVA_strength_left_bridge': [],
                    'lagged_PVA_strength_right_bridge': [],
                    'lagged_Bump_width_left_z' :[],
                    'lagged_Bump_width_right_z' :[],
                    'Angular_speed_lagged' :[],

                }
                #get raw F data
                columns_left = ['Raw_F_1', 'Raw_F_2', 'Raw_F_3', 'Raw_F_4', 'Raw_F_5', 'Raw_F_6', 'Raw_F_7', 'Raw_F_8']
                columns_right = ['Raw_F_9','Raw_F_10','Raw_F_11','Raw_F_12','Raw_F_13','Raw_F_14','Raw_F_15','Raw_F_16']
                raw_data_left_ROI = [current_file[col].to_numpy() for col in columns_left]
                raw_data_left_ROI = np.array(raw_data_left_ROI).T
                raw_data_right_ROI = [current_file[col].to_numpy() for col in columns_right]
                raw_data_right_ROI = np.array(raw_data_right_ROI).T

                dF_F_array_left = get_dff_array(raw_F_array = raw_data_left_ROI, ROI_num = 8, F_zero_cutoff = 0.05, if_plot = 0)
                dF_F_array_right = get_dff_array(raw_F_array = raw_data_right_ROI, ROI_num = 8, F_zero_cutoff = 0.05, if_plot = 0)
                dF_F_array_norm_left = normalizing_dff_array_lyu(f_input = raw_data_left_ROI, ROI_num = 8, if_plot = 0)
                dF_F_array_norm_right = normalizing_dff_array_lyu(f_input = raw_data_right_ROI, ROI_num = 8, if_plot = 0)
                dF_F_array_norm_TQ_left = normalizing_dff_array(df_f_input= dF_F_array_left,ROI_num=8, normalize_cutoff=0.95, if_plot=0)
                dF_F_array_norm_TQ_right = normalizing_dff_array(df_f_input=dF_F_array_right,ROI_num=8, normalize_cutoff=0.95, if_plot=0)
                z_score_data_left_bridge = zscore(raw_data_left_ROI)
                z_score_data_right_bridge = zscore(raw_data_right_ROI)

                PVA_left_bridge, PVA_left_bridge_strength = PVA_radian_calcul_norm(z_score_data_left_bridge, volume_cycle,8,norm="contrast")
                PVA_right_bridge, PVA_right_bridge_strength = PVA_radian_calcul_norm(z_score_data_right_bridge, volume_cycle,8,norm="contrast")
                FWHM_left_bridge = calculateBumpWidth_v1(z_score_data_left_bridge,8)
                FWHM_right_bridge = calculateBumpWidth_v1(z_score_data_right_bridge,8)
                
              
                
                # calculate bump amplitude by getting the normalized dff at current PVA angle 
                
                #First way of calculating bump amplitude: Amplitude coming from the PVA-resided Glomerulus
                Bump_amplitude_PVA_left, Bump_amplitude_PVA_opposite_left = calcualteBumpAmplitude_V3(z_score_data_left_bridge,PVA_left_bridge)
                Bump_amplitude_PVA_right, Bump_amplitude_PVA_opposite_right = calcualteBumpAmplitude_V3(z_score_data_right_bridge,PVA_right_bridge)     
                Bump_amplitude_smooth_for_speed_cor_PVA_left = gaussian_filter1d(Bump_amplitude_PVA_left, sigma =gaussian_sigma)
                Bump_amplitude_smooth_for_speed_cor_PVA_opposite_left = gaussian_filter1d(Bump_amplitude_PVA_opposite_left, sigma =gaussian_sigma)
                Bump_amplitude_smooth_for_speed_cor_PVA_right = gaussian_filter1d(Bump_amplitude_PVA_right, sigma =gaussian_sigma)
                Bump_amplitude_smooth_for_speed_cor_PVA_opposite_right = gaussian_filter1d(Bump_amplitude_PVA_opposite_right, sigma =gaussian_sigma)
                
                
                #Second way of calculating the bump amplitude: Max-min signal at each time frame
                Bump_amplitude_max_min_left = calcualteBumpAmplitude(z_score_data_left_bridge)
                Bump_amplitude_max_min_right= calcualteBumpAmplitude(z_score_data_right_bridge)
                Bump_amplitude_smooth_for_speed_cor_max_min_left = gaussian_filter1d(Bump_amplitude_max_min_left, sigma =gaussian_sigma)
                Bump_amplitude_smooth_for_speed_cor_max_min_right = gaussian_filter1d(Bump_amplitude_max_min_right, sigma =gaussian_sigma)
                
                #Third way of calculating the bump amplitude: Averaging two max glomerulus at each time frame (Green et.al 17)
                Bump_amplitude_two_max_average_left = calcualteBumpAmplitude_V2_green(z_score_data_left_bridge)
                Bump_amplitude_two_max_average_right= calcualteBumpAmplitude_V2_green(z_score_data_right_bridge)
                Bump_amplitude_smooth_for_speed_cor_two_max_average_left = gaussian_filter1d(Bump_amplitude_two_max_average_left, sigma =gaussian_sigma)
                Bump_amplitude_smooth_for_speed_cor_two_max_average_right = gaussian_filter1d(Bump_amplitude_two_max_average_right, sigma =gaussian_sigma)
               
                lag = int(-0.3/volume_time)
                
                #shift back the bump_amplitude or PVA strength by drop the lag frame at the beginning
                lagged_Bump_amplitude_smooth_for_speed_cor_PVA_left = Bump_amplitude_smooth_for_speed_cor_PVA_left[-lag:]
                lagged_Bump_amplitude_smooth_for_speed_cor_PVA_opposite_left = Bump_amplitude_smooth_for_speed_cor_PVA_opposite_left[-lag:]
                lagged_Bump_amplitude_smooth_for_speed_cor_PVA_right = Bump_amplitude_smooth_for_speed_cor_PVA_right[-lag:]
                lagged_Bump_amplitude_smooth_for_speed_cor_PVA_opposite_right = Bump_amplitude_smooth_for_speed_cor_PVA_opposite_right[-lag:]
                lagged_Bump_amplitude_smooth_for_speed_cor_max_min_left = Bump_amplitude_smooth_for_speed_cor_max_min_left[-lag:]
                lagged_Bump_amplitude_smooth_for_speed_cor_max_min_right = Bump_amplitude_smooth_for_speed_cor_max_min_right[-lag:]
                lagged_Bump_amplitude_smooth_for_speed_cor_two_max_average_left = Bump_amplitude_smooth_for_speed_cor_two_max_average_left[-lag:]
                lagged_Bump_amplitude_smooth_for_speed_cor_two_max_average_right = Bump_amplitude_smooth_for_speed_cor_two_max_average_right[-lag:]
                
                Angular_speed_degrees_lagged = Angular_speed_degrees[:len(lagged_Bump_amplitude_smooth_for_speed_cor_PVA_left)]
                PVA_left_bridge_strength_lagged = PVA_left_bridge_strength[-lag:]
                PVA_right_bridge_strength_lagged = PVA_right_bridge_strength[-lag:]
                FWHM_left_bridge_lagged  = FWHM_left_bridge[-lag:]
                FWHM_right_bridge_lagged = FWHM_right_bridge[-lag:]

                
                
                
                
                
                # Append or concatenate new session data to existing fly data
                if len(fly_data[fly_id]['lagged_Bump_amplitude_PVA_z_smooth_left']) > 0:
                    fly_data[fly_id]['lagged_Bump_amplitude_PVA_z_smooth_left'] = np.concatenate((fly_data[fly_id]['lagged_Bump_amplitude_PVA_z_smooth_left'], lagged_Bump_amplitude_smooth_for_speed_cor_PVA_left))
                    fly_data[fly_id]['lagged_Bump_amplitude_opposite_z_smooth_left'] = np.concatenate((fly_data[fly_id]['lagged_Bump_amplitude_opposite_z_smooth_left'], lagged_Bump_amplitude_smooth_for_speed_cor_PVA_opposite_left))
                    fly_data[fly_id]['lagged_Bump_amplitude_max_min_z_smooth_left']  = np.concatenate((fly_data[fly_id]['lagged_Bump_amplitude_max_min_z_smooth_left'], lagged_Bump_amplitude_smooth_for_speed_cor_max_min_left))
                    fly_data[fly_id]['lagged_Bump_amplitude_two_max_average_z_smooth_left']  = np.concatenate((fly_data[fly_id]['lagged_Bump_amplitude_two_max_average_z_smooth_left'], lagged_Bump_amplitude_smooth_for_speed_cor_two_max_average_left))
                    
                    
                    fly_data[fly_id]['lagged_Bump_amplitude_PVA_z_smooth_right'] = np.concatenate((fly_data[fly_id]['lagged_Bump_amplitude_PVA_z_smooth_right'], lagged_Bump_amplitude_smooth_for_speed_cor_PVA_right))
                    fly_data[fly_id]['lagged_Bump_amplitude_opposite_z_smooth_right'] = np.concatenate((fly_data[fly_id]['lagged_Bump_amplitude_opposite_z_smooth_right'], lagged_Bump_amplitude_smooth_for_speed_cor_PVA_opposite_right))
                    fly_data[fly_id]['lagged_Bump_amplitude_max_min_z_smooth_right']  = np.concatenate((fly_data[fly_id]['lagged_Bump_amplitude_max_min_z_smooth_right'], lagged_Bump_amplitude_smooth_for_speed_cor_max_min_right))
                    fly_data[fly_id]['lagged_Bump_amplitude_two_max_average_z_smooth_right']  = np.concatenate((fly_data[fly_id]['lagged_Bump_amplitude_two_max_average_z_smooth_right'], lagged_Bump_amplitude_smooth_for_speed_cor_two_max_average_right))
                    
                    
                    
                    fly_data[fly_id]['Angular_speed_lagged'] = np.concatenate((fly_data[fly_id]['Angular_speed_lagged'], Angular_speed_degrees_lagged))
                    fly_data[fly_id]['lagged_PVA_strength_left_bridge'] = np.concatenate((fly_data[fly_id]['lagged_PVA_strength_left_bridge'], PVA_left_bridge_strength_lagged))
                    fly_data[fly_id]['lagged_PVA_strength_right_bridge'] = np.concatenate((fly_data[fly_id]['lagged_PVA_strength_right_bridge'], PVA_right_bridge_strength_lagged))
                    fly_data[fly_id]['lagged_Bump_width_left_z'] = np.concatenate((fly_data[fly_id]['lagged_Bump_width_left_z'], FWHM_left_bridge_lagged))
                    fly_data[fly_id]['lagged_Bump_width_right_z'] = np.concatenate((fly_data[fly_id]['lagged_Bump_width_right_z'], FWHM_right_bridge_lagged))
                    

                
                else:
                    fly_data[fly_id]['lagged_Bump_amplitude_PVA_z_smooth_left'] =  lagged_Bump_amplitude_smooth_for_speed_cor_PVA_left
                    fly_data[fly_id]['lagged_Bump_amplitude_opposite_z_smooth_left'] =  lagged_Bump_amplitude_smooth_for_speed_cor_PVA_opposite_left
                    fly_data[fly_id]['lagged_Bump_amplitude_max_min_z_smooth_left'] =  lagged_Bump_amplitude_smooth_for_speed_cor_max_min_left
                    fly_data[fly_id]['lagged_Bump_amplitude_two_max_average_z_smooth_left'] =  lagged_Bump_amplitude_smooth_for_speed_cor_two_max_average_left
                    
                    
                    fly_data[fly_id]['lagged_Bump_amplitude_PVA_z_smooth_right'] =  lagged_Bump_amplitude_smooth_for_speed_cor_PVA_right
                    fly_data[fly_id]['lagged_Bump_amplitude_opposite_z_smooth_right'] =  lagged_Bump_amplitude_smooth_for_speed_cor_PVA_opposite_right
                    fly_data[fly_id]['lagged_Bump_amplitude_max_min_z_smooth_right'] =  lagged_Bump_amplitude_smooth_for_speed_cor_max_min_right
                    fly_data[fly_id]['lagged_Bump_amplitude_two_max_average_z_smooth_right'] =  lagged_Bump_amplitude_smooth_for_speed_cor_two_max_average_right
                    
                    
                    
                    
                    fly_data[fly_id]['Angular_speed_lagged'] = Angular_speed_degrees_lagged
                    fly_data[fly_id]['lagged_PVA_strength_left_bridge'] = PVA_left_bridge_strength_lagged
                    fly_data[fly_id]['lagged_PVA_strength_right_bridge'] = PVA_right_bridge_strength_lagged
                    fly_data[fly_id]['lagged_Bump_width_left_z'] = FWHM_left_bridge_lagged
                    fly_data[fly_id]['lagged_Bump_width_right_z'] = FWHM_right_bridge_lagged


               
                
    return fly_data      






def correlation_across_speed_range(
    flydata,
    binning_method='uniform',  # 'uniform' or 'quantile'
    bin_Width=None,
    bin_Edges_low=None,
    bin_Edges_up=None,
    num_bins=None,
    x_column='Angular_speed',
    y_column='lagged_Bump_amplitude_z_smooth',
    strength_column='lagged_PVA_strength',
    strength_threshold=None,
    max_per_bin=None  # Optional: for uniform binning only
):
    # Decide bin edges
    if binning_method == 'uniform':
        assert bin_Width is not None and bin_Edges_low is not None and bin_Edges_up is not None, \
            "For uniform binning, bin_Width, bin_Edges_low, and bin_Edges_up must be specified."
        bin_edges = np.arange(bin_Edges_low, bin_Edges_up + bin_Width, bin_Width)

    elif binning_method == 'quantile':
        assert num_bins is not None, "For quantile binning, num_bins must be specified."
        all_x = []
        for data in flydata.values():
            x_data = data[x_column]
            if strength_threshold is not None and strength_column in data:
                x_data = x_data[data[strength_column] >= strength_threshold]
            all_x.append(x_data)
        all_x_combined = np.concatenate(all_x)
        range_mask = (all_x_combined >= bin_Edges_low) & (all_x_combined <= bin_Edges_up)
        filtered_x = all_x_combined[range_mask]
        bin_edges = np.unique(np.quantile(filtered_x, np.linspace(0, 1, num_bins + 1)))
    else:
        raise ValueError("Invalid binning_method. Use 'uniform' or 'quantile'.")

    bin_centers = bin_edges[:-1] + 0.5 * np.diff(bin_edges)
    binned_results_df = pd.DataFrame(index=bin_centers)
    binned_counts_df = pd.DataFrame(index=bin_centers)

    for fly_id, data in flydata.items():
        x_data = data[x_column]
        y_data = data[y_column]
        strength_data = data.get(strength_column, None)

        # Apply strength threshold
        if strength_threshold is not None and strength_data is not None:
            valid_mask = strength_data >= strength_threshold
            x_data = x_data[valid_mask]
            y_data = y_data[valid_mask]

        # Optional downsampling (only if uniform binning and max_per_bin set)
        if binning_method == 'uniform' and max_per_bin is not None:
            bin_indices = np.digitize(x_data, bins=bin_edges) - 1
            valid_mask = (bin_indices >= 0) & (bin_indices < len(bin_edges) - 1)
            x_data = x_data[valid_mask]
            y_data = y_data[valid_mask]
            bin_indices = bin_indices[valid_mask]

            x_subsampled = []
            y_subsampled = []

            for i in range(len(bin_edges) - 1):
                in_bin = bin_indices == i
                x_in_bin = x_data[in_bin]
                y_in_bin = y_data[in_bin]

                if len(x_in_bin) > max_per_bin:
                    idx = resample(np.arange(len(x_in_bin)), n_samples=max_per_bin, replace=False)
                    x_in_bin = x_in_bin[idx]
                    y_in_bin = y_in_bin[idx]

                x_subsampled.append(x_in_bin)
                y_subsampled.append(y_in_bin)

            x_data = np.concatenate(x_subsampled)
            y_data = np.concatenate(y_subsampled)

        # Binned statistics
        binned_stats = binned_statistic(x_data, y_data, statistic='mean', bins=bin_edges)
        binned_counts = binned_statistic(x_data, y_data, statistic='count', bins=bin_edges).statistic

        # Store results
        binned_results_df[fly_id] = binned_stats.statistic
        binned_counts_df[fly_id] = binned_counts

        # Warnings
        for i, count in enumerate(binned_counts):
            if count < 10:
                print(f"Warning: For fly {fly_id}, bin {bin_edges[i]:.2f}–{bin_edges[i+1]:.2f} has only {int(count)} samples.")

        # Plotting
        valid_bins = ~np.isnan(binned_stats.statistic)
        if np.sum(valid_bins) > 1:
            corr = np.corrcoef(bin_centers[valid_bins], binned_stats.statistic[valid_bins])[0, 1]
        else:
            corr = np.nan

        plt.figure(figsize=(15, 5))
        plt.bar(bin_centers[valid_bins], binned_stats.statistic[valid_bins],
                width=np.diff(bin_edges)[valid_bins], align='center', edgecolor='black')
        plt.xlabel(f'{x_column} (°/sec)')
        plt.ylabel(f'{y_column} (units)')
        title = f"{fly_id} – {binning_method.capitalize()}-Binned {y_column} vs. {x_column}"
        if strength_threshold is not None:
            title += f"\n(Strength threshold: {strength_threshold})"
        if max_per_bin is not None and binning_method == 'uniform':
            title += f"\n(Downsampled to ≤ {max_per_bin} per bin)"
        title += f"\nCorrelation: {corr:.2f}"
        plt.title(title)
        plt.show()

    return binned_results_df, binned_counts_df
