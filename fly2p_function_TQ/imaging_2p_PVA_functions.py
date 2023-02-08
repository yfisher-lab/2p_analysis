import numpy as np
import math

#1.2023  Tianhao Qiu Fisher Lab

#Rule calculate PVA as from -180 degree to 180 degree (jump happens at 180 to -180) This rules apply to all function calculating offset between PVA and heading

#Rule 2: Here we assuming 8 ROIs in total





#Calculate PVA
def PVA_radian_calcul (dff_array, frame_number, ROI_NUM):
    #Assign an angle-array for 8 ROIs ROI1(1R/1L)is0,  ROI2(2R/8L) is 45
    #angle_array_roi_8 = [0,np.pi/4,np.pi/2,np.pi*3/4,np.pi,np.pi*5/4,np.pi*3/2,np.pi*7/4]
    #angle_array_roi_8 = [np.pi/8,np.pi*3/8,np.pi*5/8,np.pi*7/8,np.pi*9/8,np.pi*11/8,np.pi*13/8,np.pi*15/8]
    angle_array_roi_8 = [np.pi/8,np.pi*3/8,np.pi*5/8,np.pi*7/8,-np.pi*7/8,-np.pi*5/8,-np.pi*3/8,-np.pi/8]
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


#Calculate Bump width as long as it is >= half maximum values (Tuner-evans et al. 2021)
def calculateBumpWidth_v1 (signal_array, ROI_number):
    width_array = np.zeros(len(signal_array))
    for i in range(len(signal_array)):
        half_max_signal = np.max(signal_array[i,:])/2
        # compare dff of each glomeruli to half max
        count = 0
        for j in range (ROI_number):
            if signal_array[i,j] >= half_max_signal:
                count = count + 1
        width_array[i] = count * (360/ROI_number)
    return width_array