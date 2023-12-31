import numpy as np
import math
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

#1.2023  Tianhao Qiu Fisher Lab

#Rule calculate PVA as from -180 degree to 180 degree (jump happens at 180 to -180) This rules apply to all function calculating offset between PVA and heading

#Rule 2: Here we assuming 8 ROIs in total





#Calculate PVA
def PVA_radian_calcul (dff_array, frame_number, ROI_NUM):
    #Assign an angle-array for 8 ROIs ROI1(1R/1L)is0,  ROI2(2R/8L) is 45
    if ROI_NUM == 8:
        angle_array_roi_8 = [np.pi/8,np.pi*3/8,np.pi*5/8,np.pi*7/8,-np.pi*7/8,-np.pi*5/8,-np.pi*3/8,-np.pi/8]
    elif ROI_NUM == 16:
        angle_array_roi_8 = [np.pi/16,np.pi*3/16,np.pi*5/16,np.pi*7/16,np.pi*9/16,np.pi*11/16,np.pi*13/16,np.pi*15/16,-np.pi*15/16,-np.pi*13/16,-np.pi*11/16,-np.pi*9/16,-np.pi*7/16,-np.pi*5/16,-np.pi*3/16,-np.pi/16]
    #Define a polar 2 cartesian function 
    def pol2cart(rho, phi):
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return x, y
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
