import numpy as np
import math
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt
from scipy.stats import circmean
from scipy.stats import circvar

#1.2023  Tianhao Qiu Fisher Lab

#Rule calculate PVA as from -180 degree to 180 degree (jump happens at 180 to -180) This rules apply to all function calculating offset between PVA and heading

#Rule 2: Here we assuming 8 ROIs in total





def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


#Calculate PVA
def PVA_radian_calcul (dff_array, frame_number, ROI_NUM):
    #Assign an angle-array for 8 ROIs ROI1(1R/1L)is0,  ROI2(2R/8L) is 45
    if ROI_NUM == 8:
        angle_array_roi_8 = [np.pi/8,np.pi*3/8,np.pi*5/8,np.pi*7/8,-np.pi*7/8,-np.pi*5/8,-np.pi*3/8,-np.pi/8]
    elif ROI_NUM == 16:
        angle_array_roi_8 = [np.pi/16,np.pi*3/16,np.pi*5/16,np.pi*7/16,np.pi*9/16,np.pi*11/16,np.pi*13/16,np.pi*15/16,-np.pi*15/16,-np.pi*13/16,-np.pi*11/16,-np.pi*9/16,-np.pi*7/16,-np.pi*5/16,-np.pi*3/16,-np.pi/16]
  
    #Initialize PVA_array 
    PVA_radianArray = np.zeros(frame_number)
    PVAStrength = np.zeros(frame_number)
    
    for current_PVA_index in range(len(dff_array)):
        temp_x= np.zeros(ROI_NUM)
        temp_y= np.zeros(ROI_NUM)
        for current_ROI_index in range(ROI_NUM):
            temp_x[current_ROI_index], temp_y[current_ROI_index] = pol2cart(dff_array[current_PVA_index,current_ROI_index],angle_array_roi_8[current_ROI_index])
            x_value_PVA = sum(temp_x)
            y_value_PVA = sum(temp_y)
            PVA_radianArray[current_PVA_index] = np.arctan2(y_value_PVA , x_value_PVA )
            PVAStrength[current_PVA_index] = np.sqrt(x_value_PVA **2 + y_value_PVA **2)
    
    return PVA_radianArray,PVAStrength 



def PVAangleToRoi (PVA_angle):
    PVA_ROI = np.zeros(PVA_angle.size)
    for current_frame in range(len(PVA_angle)):
        if 0 <= PVA_angle[current_frame] < 45:
            PVA_ROI[current_frame] = 0
        elif 45 <= PVA_angle[current_frame] < 90:
            PVA_ROI[current_frame] = 1
        elif 90 <= PVA_angle[current_frame] < 135:
            PVA_ROI[current_frame] = 2
        elif 135 <= PVA_angle[current_frame] <= 180:
            PVA_ROI[current_frame] = 3
        elif -180 <= PVA_angle[current_frame] < -135:
            PVA_ROI[current_frame] = 4
        elif -135 <= PVA_angle[current_frame] < -90:
            PVA_ROI[current_frame] = 5
        elif -90 <= PVA_angle[current_frame] < -45:
            PVA_ROI[current_frame] = 6
        else:
            PVA_ROI[current_frame] = 7
            
    return PVA_ROI



def PVA_radian_to_angle(PVA_radian):
    PVA_angle = np.zeros(PVA_radian.size)
    for current_frame in range(len(PVA_radian)):
        #if PVA_radian[current_frame] >= 0:
            #PVA_angle[current_frame] = math.degrees(PVA_radian[current_frame])
        #else:
            #PVA_angle[current_frame] = 360 + math.degrees(PVA_radian[current_frame])
        PVA_angle[current_frame] = math.degrees(PVA_radian[current_frame])
    return PVA_angle




def PVA_angle_to_radian(PVA_angle):
    PVA_radian = np.zeros(PVA_angle.size)
    for current_frame in range(len(PVA_angle)):
        PVA_radian[current_frame] = math.radians(PVA_angle[current_frame])
    return PVA_radian




#Calculate real-time bumop amplitude baased on method in Fisher, Marquis et al. 2022 
def calcualteBumpAmplitude (signal_array):
    amplitude_array = np.zeros(len(signal_array))
    for i in range(len(signal_array)):
        max_signal = np.max(signal_array[i,:])
        min_signal = np.min(signal_array[i,:])
        amplitude_array[i] = max_signal - min_signal
    return amplitude_array



#Calculate real-time bumop amplitude baased on method in Green et al. 2017 
def calcualteBumpAmplitude_V2_green (signal_array):
    amplitude_array = np.zeros(len(signal_array))
    for i in range(len(signal_array)):
        max_signal = np.max(signal_array[i,:])
        #Find second largest temp
        temp = [a for i,a in enumerate(signal_array[i,:]) if a < max_signal]
        sec_max_signal = np.max(temp)
        amplitude_array[i] = (max_signal + sec_max_signal) / 2
    return amplitude_array


# Another way to calculate bump amplitude by getting the normalized dff at current PVA angle and return the amp at 180 degrees opposite
def calcualteBumpAmplitude_V3 (signal_array, PVA_array_radian):
    amplitude_array = np.zeros(len(signal_array))
    amplitude_array_opposite = np.zeros(len(signal_array))
    for i in range(len(signal_array)):
        if 0 <= PVA_array_radian[i] < (np.pi/4.0):
            amplitude_array[i] = signal_array[i,0]
            amplitude_array_opposite[i] = signal_array[i,4]
        elif (np.pi/4.0) <= PVA_array_radian[i] < (np.pi/2.0):
            amplitude_array[i] = signal_array[i,1]
            amplitude_array_opposite[i] = signal_array[i,5]
        elif (np.pi/2.0) <= PVA_array_radian[i] < (3*np.pi/4.0):
            amplitude_array[i] = signal_array[i,2]
            amplitude_array_opposite[i] = signal_array[i,6]
        elif (3*np.pi/4) <= PVA_array_radian[i] <= np.pi:
            amplitude_array[i] = signal_array[i,3]
            amplitude_array_opposite[i] = signal_array[i,7]
        elif (-np.pi) <= PVA_array_radian[i] < (-3*np.pi/4.0):
            amplitude_array[i] = signal_array[i,4]
            amplitude_array_opposite[i] = signal_array[i,0]
        elif (-3*np.pi/4.0) <= PVA_array_radian[i] < -np.pi/2:
            amplitude_array[i] = signal_array[i,5]
            amplitude_array_opposite[i] = signal_array[i,1]
        elif (-np.pi/2.0) <= PVA_array_radian[i] < (-np.pi/4.0):
            amplitude_array[i] = signal_array[i,6]
            amplitude_array_opposite[i] = signal_array[i,2]
        else:
            amplitude_array[i] = signal_array[i,7]
            amplitude_array_opposite[i] = signal_array[i,3]
    return amplitude_array, amplitude_array_opposite


# Another way to calculate bump amplitude based on PVA, but contrast to V3 it is the same PVA represented ROI during the stoppiing period
def calcualteBumpAmplitude_V4 (bump_amplitude_given, signal_array, PVA_array_radian, stopping_array):
    amplitude_array_V4 = bump_amplitude_given.copy()
    for current_index in range(len(stopping_array)):
        start_index = stopping_array[current_index,0]-stopping_array[current_index,1]+1
        end_index = stopping_array[current_index,0]
        if 0 <= PVA_array_radian[start_index] < np.pi/4:
            amplitude_array_V4 [start_index:end_index+1] = signal_array[start_index:end_index+1,0]
            #amplitude_array_opposite[i] = signal_array[i,4]
        elif np.pi/4 <= PVA_array_radian[start_index] < np.pi/2:
            amplitude_array_V4 [start_index:end_index+1] = signal_array[start_index:end_index+1,1]
            #amplitude_array_opposite[i] = signal_array[i,5]
        elif np.pi/2 <= PVA_array_radian[start_index] < 3*np.pi/4:
            amplitude_array_V4 [start_index:end_index+1] = signal_array[start_index:end_index+1,2]
            #amplitude_array_opposite[i] = signal_array[i,6]
        elif 3*np.pi/4 <= PVA_array_radian[start_index] <= np.pi:
            amplitude_array_V4 [start_index:end_index+1] = signal_array[start_index:end_index+1,3]
            #amplitude_array_opposite[i] = signal_array[i,7]
        elif -np.pi <= PVA_array_radian[start_index] < -3*np.pi/4:
            amplitude_array_V4[start_index:end_index+1] = signal_array[start_index:end_index+1,4]
            #amplitude_array_opposite[i] = signal_array[i,0]
        elif -3*np.pi/4 <= PVA_array_radian[start_index] < -np.pi/2:
            amplitude_array_V4[start_index:end_index+1] = signal_array[start_index:end_index+1,5]
            #amplitude_array_opposite[i] = signal_array[i,1]
        elif -np.pi/2 <= PVA_array_radian[start_index] < -np.pi/4:
            amplitude_array_V4[start_index:end_index+1] = signal_array[start_index:end_index+1,6]
            #amplitude_array_opposite[i] = signal_array[i,2]
        else:
            amplitude_array_V4[start_index:end_index+1] = signal_array[start_index:end_index+1,7]
            #amplitude_array_opposite[i] = signal_array[i,3]
    return amplitude_array_V4



#Calculate Bump width as long as it is >= 50% of (min+half(max-min)) values (Tuner-evans et al. 2021)
def calculateBumpWidth_v1 (signal_array, ROI_number):
    width_array = np.zeros(len(signal_array))
    for i in range(len(signal_array)):
        max_signal = np.max(signal_array[i,:])
        min_signal = np.min(signal_array[i,:])
        half_max_signal = min_signal + (max_signal- min_signal)/2
        # compare dff of each glomeruli to half max
        count = 0
        for j in range (ROI_number):
            if signal_array[i,j] >= half_max_signal:
                count = count + 1
        width_array[i] = count * (360/ROI_number)
    return width_array



# Fit a von Mises distribution for bump position in radian, using non-linear least square and  trust-region-reflexive optimization
def von_Mises_fitting_dff_TQ(function, x_data, y_data):
    parameters_array = np.zeros((y_data.shape[1],3))
    fitting_value_radian_array = np.zeros(y_data.shape[1])
    goodnees_of_fit_vm_rsquare = np.zeros(y_data.shape[1])
    for i in range(y_data.shape[1]):
        popt,pcov = curve_fit(function, x_data, y_data[:,i] ,method = 'trf',bounds=([0,-np.pi,-100],[10,np.pi,100]))
        
        #Assign three paramters to output array
        parameters_array[i,0] = popt[0]
        parameters_array[i,1] = popt[1]
        parameters_array[i,2] = popt[2]
        
        
        #Find and assign a fit value on range -pi to pi
        x = np.linspace(-np.pi, np.pi, 1000)
        fitting_value_radian_array[i] = -np.pi+(2*np.pi*np.argmax(function(x,  parameters_array[i,0],parameters_array[i,1],parameters_array[i,2]))/1000)
        
        #Find and assign goodness of fit (r-square)
        
        #residuals = y_data[:,i] - function(x_data, *popt)
        #ss_res = np.sum(residuals**2)
        #ss_tot = np.sum((y_data[:,i]-np.mean(y_data[:,i]))**2)
        #goodnees_of_fit_vm_rsquare[i] = 1-(ss_res/ss_tot)
        goodnees_of_fit_vm_rsquare[i] = r2_score(y_data[:,i],  function(x_data, *popt))
    
    
    return parameters_array,  fitting_value_radian_array, goodnees_of_fit_vm_rsquare



def strong_PVA_duration(PVA_strength_array, strength_threshold,time_per_frame,minimum_window_s):
    chunk_array = []
    count = 0
    for i in range(len(PVA_strength_array)):
        if PVA_strength_array[i] >= strength_threshold:
            count = count + 1
        else:
            current_chunk_time = count*time_per_frame
            if current_chunk_time >= minimum_window_s:
                chunk_array.append(current_chunk_time)
            count = 0
    
    # Handle the case where the last chunk reaches the end of the array
    if count > 0:
        current_chunk_time = count * time_per_frame
        if current_chunk_time >= minimum_window_s:
            chunk_array.append(current_chunk_time)
    
    return chunk_array


def strong_PVA_index(PVA_strength_array, strength_threshold,time_per_frame,minimum_window_s):
    index_array = []
    count = 0
    for i in range(len(PVA_strength_array)):
        if PVA_strength_array[i] >= strength_threshold:
            count = count + 1
        else:
            current_chunk_time = count*time_per_frame
            if current_chunk_time >= minimum_window_s:
                index_array.extend(range(i - count, i))
            count = 0
            
    # Handle the case where the last chunk reaches the end of the array
    if count > 0:
        current_chunk_time = count * time_per_frame
        if current_chunk_time >= minimum_window_s:
            index_array.extend(range(len(PVA_strength_array) - count, len(PVA_strength_array)))

    return index_array


def strong_signal_index(signal_array, strength_threshold,time_per_frame,minimum_window_s):
    index_array = []
    count = 0
    for i in range(len(signal_array)):
        if signal_array[i] >= strength_threshold:
            count = count + 1
        else:
            current_chunk_time = count*time_per_frame
            if current_chunk_time >= minimum_window_s:
                index_array.extend(range(i - count, i))
            count = 0
            
    # Handle the case where the last chunk reaches the end of the array
    if count > 0:
        current_chunk_time = count * time_per_frame
        if current_chunk_time >= minimum_window_s:
            index_array.extend(range(len(signal_array) - count, len(signal_array)))

    return index_array


def weak_signal_index(signal_array, strength_threshold,time_per_frame,minimum_window_s):
    index_array = []
    count = 0
    for i in range(len(signal_array)):
        if signal_array[i] <= strength_threshold:
            count = count + 1
        else:
            current_chunk_time = count*time_per_frame
            if current_chunk_time >= minimum_window_s:
                index_array.extend(range(i - count, i))
            count = 0
            
    # Handle the case where the last chunk reaches the end of the array
    if count > 0:
        current_chunk_time = count * time_per_frame
        if current_chunk_time >= minimum_window_s:
            index_array.extend(range(len(signal_array) - count, len(signal_array)))

    return index_array



def get_behavior_state_of_strong_PVA(strong_PVA_index_array,behavior_state_index_array):
    state_array = np.zeros(len(strong_PVA_index_array))
    for i in range(len(state_array)):
        if behavior_state_index_array[strong_PVA_index_array[i]] == 0:
            state_array[i] = 0
        else:
            state_array[i] = 1
    return state_array



def get_bump_shape_at_strong_signal(Bump_shape_array,signal_index_array,ROI_number):
    if ROI_number == 8:
        bump_shape_at_strong_signal_array =  np.empty((8, 0))
    else:
        bump_shape_at_strong_signal_array =  np.empty((16, 0))
    
    for i in range(len(signal_index_array)):
        current_index = signal_index_array[i]
        bump_shape_at_strong_signal_array =  np.hstack((bump_shape_at_strong_signal_array ,Bump_shape_array[:,current_index].reshape(-1, 1)))
        
    average_bump_shape_at_strong_signal_array = np.mean(bump_shape_at_strong_signal_array,axis = 1)    
    
    
    return average_bump_shape_at_strong_signal_array



def get_PVA_at_strong_signal(PVA_array,signal_index_array):
    PVA_at_strong_signal_array = np.zeros(len(signal_index_array))
    for i in range(len(signal_index_array)):
        current_index = signal_index_array[i]
        PVA_at_strong_signal_array[i] = PVA_array[signal_index_array[i]]
    
    
    return PVA_at_strong_signal_array






def plot_bump_shape_comparison(df1, df2,label2, label1="Bump right at stop", color1='dodgerblue', color2='darkorange'):
    """
    Function to plot the mean and SEM for two datasets with a shaded region representing SEM.

    Args:
        df1 (array-like): First dataset (2D: trials x timepoints).
        df2 (array-like): Second dataset (2D: trials x timepoints).
        label1 (str): Label for the first dataset in the legend.
        label2 (str): Label for the second dataset in the legend.
        color1 (str): Color for the first dataset.
        color2 (str): Color for the second dataset.
    """

    # Calculate the mean and SEM for the first dataset (df1)
    mean_df1 = np.nanmean(df1, axis=1)
    sem_df1 = np.nanstd(df1, axis=1) / np.sqrt(df1.shape[1])

    # Calculate the mean and SEM for the second dataset (df2)
    mean_df2 = np.nanmean(df2, axis=1)
    sem_df2 = np.nanstd(df2, axis=1) / np.sqrt(df2.shape[1])

    # Create the figure
    plt.figure(figsize=(10, 6))

    # Plot the first dataset
    plt.plot(range(1, len(mean_df1) + 1), mean_df1, color=color1, linewidth=3, label=f'{label1} (Mean)')
    plt.fill_between(range(1, len(mean_df1) + 1), mean_df1 - sem_df1, mean_df1 + sem_df1, color=color1, alpha=0.3)

    # Plot the second dataset
    plt.plot(range(1, len(mean_df2) + 1), mean_df2, color=color2, linewidth=3, label=f'{label2} (Mean)')
    plt.fill_between(range(1, len(mean_df2) + 1), mean_df2 - sem_df2, mean_df2 + sem_df2, color=color2, alpha=0.3)

    # Add title, labels, and enhance ticks
    plt.title("Bump Shape Comparison", fontsize=18, fontweight='bold', color='darkblue')
    plt.xlabel("Glomerulus Index", fontsize=14)
    plt.ylabel("Normalized dF/F", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Add a legend for only the mean
    plt.legend(fontsize=12)

    # Show the plot
    plt.show()
    
    
def plot_strong_signal_frame_pva_histogram(data_all, selected_animal=None):
    # Define colors dynamically
    num_animals = len(data_all['PVA_at_strong_signal'])
    color_list = [plt.cm.viridis(i / num_animals) for i in range(num_animals)]  # Generates distinct colors

    # Define bins
    bins = np.linspace(-180, 180, 30)  # 20 bins from -180 to 180 degrees

    # Plot histograms in requested style
    fig, ax = plt.subplots(figsize=(10, 5))

    for (animal, data), color in zip(data_all['PVA_at_strong_signal'].items(), color_list):
        # If selected_animal is specified, skip other animals
        if selected_animal and animal != selected_animal:
            continue

        # Convert to a flat NumPy array (handles lists of lists)
        flat_data = np.concatenate([np.asarray(d, dtype=float).flatten() for d in data])

        # Compute histogram
        hist, bin_edges = np.histogram(flat_data, bins=bins, density=True)
        hist =hist/hist.sum()
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Get bin centers

        # Plot filled histogram
        ax.fill_between(bin_centers, hist, alpha=0.4, color=color, label=f"{animal}")

    # Formatting
    ax.set_xlim(-180, 180)  # Adjusted for degree scale
    ax.set_ylim(0, None)  # Auto adjust y-limit
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_xticklabels([r'$-180$', r'$-90$', '0', r'$90$', r'$180$'])
    ax.set_xlabel("PVA Position")
    ax.set_ylabel("Proportion")
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title("Distribution of  PVA at strong signal frame (Per Animal)", fontsize=16)

    # Show plot
    plt.show()

    





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

    