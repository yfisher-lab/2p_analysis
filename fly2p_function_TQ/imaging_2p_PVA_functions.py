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

import numpy as np

def PVA_radian_calcul_norm(
    dff_array,
    frame_number,
    ROI_NUM,
    norm="contrast",        # "contrast" | "mass-clip" | "mass-offset"
    return_degrees=False
):
    """
    Population Vector Average (PVA) for PB/EB ROIs.

    Parameters
    ----------
    dff_array : array-like, shape (T, R)
        Activity per frame and ROI. Can be z-scored (may include negatives).
    frame_number : int
        Number of frames T.
    ROI_NUM : int
        Number of ROIs R (must be 8 or 16 for PB angles).
    norm : str
        - "contrast"   : strength = |Σ w_i e^{jθ_i}| / Σ |w_i|
                          (contrast-sensitive; ideal for z-scored data; in [0,1])
        - "mass-clip"  : w⁺=max(w,0); strength = |Σ w⁺ e^{jθ_i}| / Σ w⁺
                          (classic circ_r on non-negative mass; in [0,1])
        - "mass-offset": w⁺=w - min(w) (if min<0 else w); same formula as mass-clip
                          (preserves per-frame contrasts while making weights ≥0)
    return_degrees : bool
        If True, returns angles in degrees; else radians.

    Returns
    -------
    angles : (T,) ndarray
        PVA angle per frame (radians by default, or degrees if return_degrees=True).
    strength : (T,) ndarray
        PVA strength per frame, bounded in [0,1].
    """
    # --- validate & coerce ---
    X = np.asarray(dff_array, dtype=float)
    if X.shape != (frame_number, ROI_NUM):
        raise ValueError(f"Shape mismatch: expected {(frame_number, ROI_NUM)}, got {X.shape}.")
    if ROI_NUM not in (8, 16):
        raise ValueError("ROI_NUM must be 8 or 16 for PB angles.")

    # --- PB angle conventions (centered in [-π, π]) ---
    if ROI_NUM == 8:
        theta = np.array([np.pi/8, 3*np.pi/8, 5*np.pi/8, 7*np.pi/8,
                          -7*np.pi/8, -5*np.pi/8, -3*np.pi/8, -np.pi/8], float)
    else:  # 16
        theta = np.pi * (2*np.arange(ROI_NUM) + 1) / (2.0 * ROI_NUM)
        theta = (theta + np.pi) % (2*np.pi) - np.pi  # wrap to [-π, π]

    # --- choose normalization mode & (possibly) transform weights ---
    if norm == "contrast":
        # signed numerator, abs denominator (good for z-scored data)
        W_for_vec = X
        denom = np.sum(np.abs(X), axis=1)
    elif norm == "mass-clip":
        # classic circ_r on non-negative "mass"
        W_for_vec = np.clip(X, 0, None)
        denom = np.sum(W_for_vec, axis=1)
    elif norm == "mass-offset":
        # shift each frame so min becomes 0 (if negative)
        mins = X.min(axis=1, keepdims=True)
        W_for_vec = np.where(mins < 0, X - mins, X)
        denom = np.sum(W_for_vec, axis=1)
    else:
        raise ValueError("norm must be 'contrast', 'mass-clip', or 'mass-offset'.")

    # --- vector sums ---
    c = np.cos(theta)[None, :]
    s = np.sin(theta)[None, :]
    Rx = np.sum(W_for_vec * c, axis=1)
    Ry = np.sum(W_for_vec * s, axis=1)

    # --- angle & strength (bounded [0,1]) ---
    angles = np.arctan2(Ry, Rx)
    Rlen = np.hypot(Rx, Ry)
    strength = np.where(denom > 0, Rlen / denom, 0.0)
    strength = np.clip(strength, 0.0, 1.0)

    if return_degrees:
        angles = np.degrees(angles)

    return angles, strength


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

def get_bump_shape_at_strong_signal_various_speed(
    Bump_shape_array,
    signal_index_array,
    ROI_number,
    Angular_speed_array=None,
    angular_speed_range=None
):
    """
    Extracts and averages bump shapes from selected frames based on signal strength 
    and optional angular speed range.

    Parameters:
        Bump_shape_array (ndarray): shape (ROIs, time)
        signal_index_array (array-like): frame indices where signal passes threshold
        ROI_number (int): 8 or 16
        Angular_speed_array (array-like or None): angular speed per frame
        angular_speed_range (tuple or None): (min_speed, max_speed)

    Returns:
        average_bump_shape_at_strong_signal_array (ndarray): shape (ROIs,)
    """

    if ROI_number == 8:
        bump_shape_at_strong_signal_array = np.empty((8, 0))
    else:
        bump_shape_at_strong_signal_array = np.empty((16, 0))

    for current_index in signal_index_array:
        # If angular speed condition is given, apply it
        if Angular_speed_array is not None and angular_speed_range is not None:
            speed = Angular_speed_array[current_index]
            if not (angular_speed_range[0] <= speed <= angular_speed_range[1]):
                continue  # Skip this frame

        bump_vec = Bump_shape_array[:, current_index].reshape(-1, 1)
        bump_shape_at_strong_signal_array = np.hstack((bump_shape_at_strong_signal_array, bump_vec))

    # Handle case where no valid frames remain
    if bump_shape_at_strong_signal_array.shape[1] == 0:
        return np.full((ROI_number,), np.nan)

    average_bump_shape_at_strong_signal_array = np.mean(bump_shape_at_strong_signal_array, axis=1)
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








    