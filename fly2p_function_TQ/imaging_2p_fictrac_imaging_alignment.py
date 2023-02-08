import numpy as np
import math
from matplotlib import pyplot as plt


#Downsampling the heading_array to the 2p volume rate
def downsample_heading_to_total_volume(cycle_number,heading_array_initial):
    from scipy.interpolate import interp1d
    cycle_num = cycle_number
    #Generate  a function based on initial heading arrat to use interpolation to find the value of new points.
    interpolated = interp1d(np.arange(len(heading_array_initial)), heading_array_initial, axis = 0, fill_value = 'extrapolate') 
    downsampled = interpolated(np.linspace(0,len(heading_array_initial), cycle_num))
    
    return downsampled




def offset_calculation (array_behavior, array_imaging, IfRadian):
    offset_array = array_behavior - array_imaging
    if IfRadian == True:
        for current_frame in range(len(offset_array)):
            if offset_array[current_frame] <= -np.pi:
                offset_array[current_frame] = np.pi * 2 + offset_array[current_frame]
            if offset_array[current_frame] >=  np.pi:
                offset_array[current_frame] = np.pi * 2 - offset_array[current_frame]
    else:
        for current_frame in range(len(offset_array)):
            if offset_array[current_frame] <= -180:
                offset_array[current_frame] = -(360 + offset_array[current_frame])
            if offset_array[current_frame] >=  180:
                offset_array[current_frame] = 360 - offset_array[current_frame]
        
    return  offset_array



# Assume all array in range -180 to 180 degrees (or-pi to pi)
# Moving the wrapped array by certain offset 
def moving_wrapped_plot_by_offset(wrapped_array, offset_Value, ifRadian):
    output_array = np.zeros(len(wrapped_array))
    if ifRadian == True:
        wrapped_array_degree = np.zeros(wrapped_array.size)
        for current_frame in range(len(wrapped_array)):
            wrapped_array_degree[current_frame] = math.degrees(wrapped_array[current_frame])
    else:
        wrapped_array_degree = wrapped_array
    
    for current_frame in range(len(wrapped_array)):
        if (np.abs(wrapped_array_degree[current_frame] + offset_Value) > 180):
            
            if wrapped_array_degree[current_frame] + offset_Value > 180:
                output_array[current_frame] = wrapped_array_degree[current_frame] + offset_Value -360                            
            else:
                output_array[current_frame] = wrapped_array_degree[current_frame] + offset_Value +360
        
        else:
            output_array[current_frame] = wrapped_array_degree[current_frame] + offset_Value
    
    if ifRadian == True:
        output_array_radian  = np.zeros(output_array.size)
        for current_frame in range(len(output_array)):
            output_array_radian[current_frame] = math.radians(output_array[current_frame])
        return output_array_radian
    else:
        return output_array



def fictrack_signal_decoding(unprocessed_heading, time_array, max_voltage, already_radian):
    #Prepocessing the heading array for further analysis
    from matplotlib import pyplot as plt
    if already_radian == False:
        RadiansArray_heading =  (np.pi * 2) - unprocessed_heading * np.pi * 2 / max_voltage
        #RadiansArray_heading =  unprocessed_heading * np.pi * 2 / max_voltage 
    else:
        RadiansArray_heading = unprocessed_heading
    unwrapped_heading = np.unwrap(RadiansArray_heading)
    nanIDX =[]
    #Find where is the big jump happening 
    upwrappedIndexes  = np.argwhere(np.absolute(np.diff(unwrapped_heading)) > np.pi)
    NUM_SAMPLES_FROM_WRAP_TO_REPLACE = 2
    
    for i in range(len(upwrappedIndexes)):
        if upwrappedIndexes[i] < (NUM_SAMPLES_FROM_WRAP_TO_REPLACE -1) and upwrappedIndexes[i] > len(unwrapped_heading) - (NUM_SAMPLES_FROM_WRAP_TO_REPLACE+1):
            upwrappedIndexes.pop(i)
            i = i - 1
    
    #Repplace potentially problematic indexs with NaN
    cleanedPos = unwrapped_heading
    for j in range(len(upwrappedIndexes)):
        index_start = upwrappedIndexes[j] - NUM_SAMPLES_FROM_WRAP_TO_REPLACE 
        index_end = upwrappedIndexes[j] + NUM_SAMPLES_FROM_WRAP_TO_REPLACE 
        cleanedPos[index_start[0]-1:index_end[0]] = np.NaN
        
    #Replace NaN values with the last preceding value that was a real number
    nanIDX = np.argwhere(np.isnan(cleanedPos))
    
    while (len(nanIDX) > 0):
        cleanedPos[nanIDX] = cleanedPos[nanIDX-1]
        nanIDX = np.argwhere(np.isnan(cleanedPos))
    
    #cleanedPos = np.absolute(cleanedPos)
    
    plt.figure(figsize= (25,7))
    plt.plot(time_array, cleanedPos)
    plt.xlabel('Time(s)', fontsize=20)
    plt.ylabel('Accumulated rotation (rad)', fontsize=20)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.show()
    
    return cleanedPos
