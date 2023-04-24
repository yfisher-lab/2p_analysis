import numpy as np
import os
import xarray as xr
from os.path import sep
from matplotlib import pyplot as plt
from tifffile import tifffile



#This is for fly before TQfly034, then brucker changes the setting and now one ome.tif file is a volume but not a slice
def combine_single_tiff(slice_num, cycle_num, file_dir):
    #Create a list for the directory of every single ome-tiff file of the selected trial
    import tifftools
    import os
    from matplotlib import pyplot as plt
    tiff_files_li = []
    for ti in os.listdir(file_dir):
        if '.ome.tif' and 'Ch2' in ti:
            tiff_files_li.append(sep.join([file_dir, ti]))
    tiff_files_li.sort()
    
    #Find the size of a single image
    single_image_x = plt.imread(tiff_files_li[0]).shape[0]
    single_image_y = plt.imread(tiff_files_li[0]).shape[1]
    #Create the combined image array with the right size
    image_combi = np.zeros([cycle_num, slice_num, single_image_x,single_image_y])
    count = 0
    for current_cycle in range(cycle_num):
        for current_slice in range(slice_num):
            image_combi[current_cycle, current_slice] = plt.imread(tiff_files_li[count])
            count = count + 1
    
    
    return image_combi, tiff_files_li



def combine_single_tiff_V2(slice_num, cycle_num, file_dir):
    #Create a list for the directory of every single ome-tiff file of the selected trial
    import tifftools
    import os
    from matplotlib import pyplot as plt
    tiff_files_li = []
    for ti in os.listdir(file_dir):
        if '.ome.tif' and 'Ch2' in ti:
            tiff_files_li.append(sep.join([file_dir, ti]))
    tiff_files_li.sort()
    
    #Find the size of a single image
    single_image_x = plt.imread(tiff_files_li[0]).shape[0]
    single_image_y = plt.imread(tiff_files_li[0]).shape[1]
    #Create the combined image array with the right size
    image_combi = np.zeros([cycle_num, slice_num, single_image_x,single_image_y])
    count = 0
    for current_cycle in range(cycle_num):
        image_combi[current_cycle] = tifffile.imread(tiff_files_li[count])
        count = count + 1
    
    
    return image_combi, tiff_files_li







def computeMotionShift(stack, refImage, upsampleFactor, sigmaval = 2, doFilter = False, stdFactor = 2, showShiftFig = True):
    from skimage.registration import phase_cross_correlation
    from scipy.ndimage.filters import gaussian_filter
    if len(refImage.shape) == 3:
        print('perform motion correction on a volume')
        refImgFilt = refImage.copy()
        for p in range(stack['planes [µm]'].size):
            refImgFilt[p,:,:] = gaussian_filter(refImage[p,:,:], sigma=sigmaval)
        shift = np.zeros((2, stack['planes [µm]'].size,stack['volumes [s]'].size))
        error = np.zeros((stack['planes [µm]'].size,stack['volumes [s]'].size))
        diffphase = np.zeros((stack['planes [µm]'].size,stack['volumes [s]'].size))
    else:
        print('perform motion correction on a single plane/max projection')
        refImgFilt = gaussian_filter(refImage, sigma=sigmaval)

        shift = np.zeros((2, stack['volumes [s]'].size))
        error = np.zeros(stack['volumes [s]'].size)
        diffphase = np.zeros(stack['volumes [s]'].size)

    # compute shift
    for i in range(stack['volumes [s]'].size):
        if len(refImage.shape) == 3:
            for p in range(stack['planes [µm]'].size):
                shifImg = stack[i,p,:,:]
                shifImgFilt = gaussian_filter(shifImg, sigma=sigmaval)

                # compute shift
                shift[:,p,i], error[p,i], diffphase[p,i] = phase_cross_correlation(refImgFilt[p,:,:].data, shifImgFilt,
                                                                             upsample_factor = upsampleFactor)
        else:
            shifImg = stack[i,:,:]
            shifImgFilt = gaussian_filter(shifImg, sigma=sigmaval)

            # compute shift
            shift[:,i], error[i], diffphase[i] = phase_cross_correlation(refImgFilt, shifImgFilt,
                                                                         upsample_factor = upsampleFactor)
    if showShiftFig:
        if len(refImage.shape) == 3:
            fig, axs = plt.subplots(2,1,figsize=(15,6))
            axlab = ['x','y']
            for i, ax in enumerate(axs):
                ax.plot(shift[i,:].T)
                ax.set_xlabel('frames')
                ax.set_ylabel('image shift for {}'.format(axlab[i]))
        else:
            fig, ax = plt.subplots(1,1,figsize=(15,5))
            ax.plot(shift[0,:])
            ax.plot(shift[1,:])
            ax.set_xlabel('frames')
            ax.set_ylabel('image shift/pixel')

    if doFilter:
        shiftFilt_x = shift[0,:].copy()
        shiftFilt_y = shift[1,:].copy()
        shiftFilt_x[abs(shiftFilt_x) > stdFactor*np.std(shiftFilt_x)] = np.nan
        shiftFilt_y[abs(shiftFilt_y) > stdFactor*np.std(shiftFilt_y)] = np.nan

        allT = np.arange(len(shiftFilt_x))
        shiftFilt_x_interp = np.interp(allT, allT[~np.isnan(shiftFilt_x)], shiftFilt_x[~np.isnan(shiftFilt_x)])
        shiftFilt_y_interp = np.interp(allT, allT[~np.isnan(shiftFilt_y)], shiftFilt_y[~np.isnan(shiftFilt_y)])

        if showShiftFig:
            ax.plot(shiftFilt_x_interp,'b')
            ax.plot(shiftFilt_y_interp,'c')

        return np.vstack((shiftFilt_x_interp,shiftFilt_y_interp))
    else:
        return shift




    
    
def motionCorrection(stack, shift):
    from scipy.ndimage import shift as spshift

    #check if shift was calculated for each plane in a volume separately, then check if stack to be aligned is 3d or 4d

    #stack should be an xarray
    stackMC = stack.copy()

    if len(shift.shape) == 3:
        # separate shifts for each plane in a volume
        if len(stack.shape) < 4:
            print("Imaging stack needs to be 4D.")
            return np.nan*stackMC
        for p in range(stack['planes [µm]'].size):
            for i in range(stack['volumes [s]'].size):
                shifImg = stack[i,p,:,:]
                stackMC[i,p,:,:] = spshift(shifImg, shift[:,p,i], order=1,mode='reflect')

    else:
        #one shift per volume per time point
        if len(stack.shape) < 4:
            # motion correction on single plane or max projection
            for i in range(stack['volumes [s]'].size):
                shifImg = stack[i,:,:]
                stackMC[i,:,:] = spshift(shifImg, shift[:,i], order=1,mode='reflect')
        else:
            #motion correction of 4D stack
            for v in range(stack["volumes [s]"].size):  #move one volume at a time
                tmpVol = stack[{"volumes [s]": v}]
                for p in range(tmpVol["planes [µm]"].size):
                    stackMC[v,p,:,:]  = spshift(tmpVol[p,:,:], shift[:,v], order=1,mode='reflect')

    return stackMC
    
    

    
    
    
    
# Write a low pass filter function for filtering imaging/heading data (same idea as written by YF)
def low_pass_filter_TQ(data, lowPassCutoff, sampleRate):
#   Inputs
#   data - trace to be filtered 
#   lowPassCutOff - value (Hz) that willbe the top limit of the filter
#   sampleRate- rate data is sampled at to allow correcrt conversion in to Hz
#   Outpput: filtered version of the data
#   Tianhao Qiu  10/2022

# build a butter function
    from scipy.signal import butter, filtfilt
    [b,a] = butter(1, lowPassCutoff / (sampleRate/2), 'lowpass')
#Filter data using butter function *(axis,padtype padlen set to be exactly same as Matlab filtfilt function)    
    out =filtfilt(b,a, data, axis = 0,padtype = 'odd', padlen=3*(max(len(b),len(a))-1))
    
    return out 






# Write a high pass filter function for filtering imaging/heading data (same idea as written by YF)
def high_pass_filter_TQ(data, highPassCutoff, sampleRate):
#   Inputs
#   data - trace to be filtered 
#   highPassCutOff - value (Hz) that willbe the top limit of the filter
#   sampleRate- rate data is sampled at to allow correcrt conversion in to Hz
#   Outpput: filtered version of the data
#   Tianhao Qiu  10/2022

# build a butter function
    from scipy.signal import butter, filtfilt
    [b,a] = butter(1, highPassCutoff / (sampleRate/2), 'highpass')
#Filter data using butter function *(axis,padtype padlen set to be exactly same as Matlab filtfilt function)    
    out =filtfilt(b,a, data, axis = 0,padtype = 'odd', padlen=3*(max(len(b),len(a))-1))
    
    return out 




def averaging_frame(frame_array, average_window):
    averaged_array = np.zeros(frame_array.size)
    for current_frame in range(len(frame_array)):
        if (current_frame + 1) <= int(np.floor(average_window/2)):
             averaged_array[current_frame] = frame_array[current_frame]
        elif (len(frame_array) <= int((np.floor(average_window/2)) +current_frame + 1)):
            averaged_array[current_frame] = frame_array[current_frame]
        else:
            averaged_array[current_frame] = np.mean(frame_array[(current_frame - int(np.floor(average_window/2))) :(current_frame + int(np.ceil(average_window/2)))])
            
    return averaged_array







def get_raw_F(ROI_NUMBER, napari_roi, raw_data):
    ROI_number = ROI_NUMBER
    F_array_output = np.zeros((raw_data['volumes [s]'].size, ROI_number))
    stack4dMC_numpy = raw_data.to_numpy()
    for time_point in range(raw_data['volumes [s]'].size):
        current_volume = stack4dMC_numpy[time_point,:,:,:]
        for ROI_index in range(1, ROI_number + 1):
            mask = napari_roi == ROI_index
            F_array_output[time_point, ROI_index - 1] =current_volume[mask].mean() 
            
    fig, axs = plt.subplots(ROI_number, 1, figsize=(13, 12))
    for i in range(ROI_number):
        ax = axs[i]
        ax.plot(F_array_output[:,i])
    fig.supylabel('F',fontsize=20)
    plt.xlabel('Frame Number', fontsize=20)
    plt.show()
    return F_array_output




def get_dff_array(raw_F_array, ROI_num, F_zero_cutoff):
    dF_F_array_output = np.zeros((len(raw_F_array), ROI_num))
    F_zero = np.quantile(raw_F_array, F_zero_cutoff, axis = 0)
    for F_zero_index in range(ROI_num):
        dF_F_array_output[:,F_zero_index] = (raw_F_array[:,F_zero_index] - F_zero[F_zero_index])/F_zero[F_zero_index]
    
    fig, axs = plt.subplots(ROI_num, 1, figsize=(13, 12))
    for i in range(ROI_num):
        ax = axs[i]
        ax.plot(dF_F_array_output[:,i])
    fig.supylabel('dF/F',fontsize=20)
    plt.xlabel('Frame Number', fontsize=20)
    plt.show()
    return dF_F_array_output



def normalizing_dff_array(df_f_input,ROI_num, normalize_cutoff):
    dF_F_array_normalized_output = np.zeros((len(df_f_input), ROI_num))
    dFF_95 = np.quantile(df_f_input, normalize_cutoff, axis = 0)
    for current_ROI in range(ROI_num):
        dF_F_array_normalized_output[:,current_ROI ] = df_f_input[:,current_ROI ]/dFF_95[current_ROI]

    fig, axs = plt.subplots(ROI_num, 1, figsize=(13, 12))
    for i in range(ROI_num):
        ax = axs[i]
        ax.plot(dF_F_array_normalized_output[:,i])
    fig.supylabel('dF/F-Normalized',fontsize=20)
    plt.xlabel('Frame Number', fontsize=20)
    plt.show()
    return dF_F_array_normalized_output





def combine_PB_corresponding_ROI(dff_array_input, napari_ROI, ROI_num, mode, time_array_imaging):
    ROI_number_combined = ROI_num
    dF_F_array_8_roi_output = np.zeros((len(dff_array_input), ROI_number_combined))
    if mode == 1:
        #mode 1 for E-PG and P-EG
        #Combine corresponding glomeruli in left and right PBs, leaving 8 ROIs for calculating PVA
        #Rule of E-PG combination: L1+R1 (Label in napari 8,9), L8+R2(1,10), L7+R3 (2,11), L6+R4(3,12), L5+R5(4,13), L4+R6(5,14) #L3+R7(6,15), L2+R8(7,16)
        for combined_ROI_index in range(ROI_number_combined):
            if combined_ROI_index == ROI_number_combined - 1:
                #Count pixel number to determine the weight of glomeruli L1 & R1 (And put it at first)
                pixel_number_left_bridge = np.count_nonzero(napari_ROI == combined_ROI_index + 1)
                pixel_number_right_bridge =  np.count_nonzero(napari_ROI == combined_ROI_index + 1 + 1)
                left_weight = pixel_number_left_bridge/( pixel_number_left_bridge + pixel_number_right_bridge)
                right_weight = pixel_number_right_bridge/( pixel_number_left_bridge + pixel_number_right_bridge)
                #Put it at first
                dF_F_array_8_roi_output[:,0] = dff_array_input[:,combined_ROI_index] * left_weight +  dff_array_input[:,combined_ROI_index + 1] * right_weight
            else:
                #Count pixel number to determine the weight of the rest of each glomeruli 
                pixel_number_left_bridge = np.count_nonzero(napari_ROI == combined_ROI_index + 1)
                pixel_number_right_bridge =  np.count_nonzero(napari_ROI == combined_ROI_index + 1 + 9)
                left_weight = pixel_number_left_bridge/( pixel_number_left_bridge + pixel_number_right_bridge)
                right_weight = pixel_number_right_bridge/( pixel_number_left_bridge + pixel_number_right_bridge)
                dF_F_array_8_roi_output[:,combined_ROI_index + 1] = dff_array_input[:,combined_ROI_index] * left_weight +  dff_array_input[:,combined_ROI_index + 9] * right_weight
      
    if mode == 2:
        #mode 2 for delta 7
        #Combine corresponding glomeruli in left and right PBs, leaving 8 ROIs for calculating PVA
        #Rule of delta combination: L9+L1+R8 (Label in napari 8,8,16), L8+R1+R9(1,9,9), L7+R2 (2,10), L6+R3(3,11), L5+R4(4,12), L4+R5(5,13) #L3+R6(6,14), L2+R7(7,15)
        for combined_ROI_index in range(ROI_number_combined):
            pixel_number_left_bridge = np.count_nonzero(napari_ROI == combined_ROI_index + 1)
            pixel_number_right_bridge =  np.count_nonzero(napari_ROI == combined_ROI_index + 1 + 8)
            left_weight = pixel_number_left_bridge/( pixel_number_left_bridge + pixel_number_right_bridge)
            right_weight = pixel_number_right_bridge/( pixel_number_left_bridge + pixel_number_right_bridge)
            dF_F_array_8_roi_output[:,combined_ROI_index] = dff_array_input[:,combined_ROI_index] * left_weight +  dff_array_input[:,combined_ROI_index + 8] * right_weight
    
    return dF_F_array_8_roi_output
        


def combine_EB_corresponding_ROI(dff_array_input, napari_ROI, ROI_num, mode, time_array_imaging):
    ROI_number_combined = ROI_num
    dF_F_array_8_roi_output = np.zeros((len(dff_array_input), ROI_number_combined))
    for combined_ROI_index in range(1, ROI_number_combined+1):
        #combine ROI1 with ROI 16
        if combined_ROI_index == 1:
            #Count pixel number to determine the weight of glomeruli L1 & R1 (And put it at first)
            pixel_number_wedge_one = np.count_nonzero(napari_ROI == combined_ROI_index )
            pixel_number_wedge_two =  np.count_nonzero(napari_ROI == combined_ROI_index + 15)
            wedge1_weight = pixel_number_wedge_one/( pixel_number_wedge_one + pixel_number_wedge_one)
            wedge2_weight = pixel_number_wedge_two/( pixel_number_wedge_one + pixel_number_wedge_two)
            #Put it at first
            dF_F_array_8_roi_output[:,0] = dff_array_input[:,combined_ROI_index - 1] * wedge1_weight +  dff_array_input[:,combined_ROI_index + 14] * wedge2_weight
        
        #Rest just neighboring wedge (23,45,67,89,1011,1213,1415)        
        else:
            index_adjust = combined_ROI_index -2
            pixel_number_wedge_one = np.count_nonzero(napari_ROI == combined_ROI_index+index_adjust )
            pixel_number_wedge_two =  np.count_nonzero(napari_ROI == combined_ROI_index + index_adjust+1)
            wedge1_weight = pixel_number_wedge_one/( pixel_number_wedge_one + pixel_number_wedge_one)
            wedge2_weight = pixel_number_wedge_two/( pixel_number_wedge_one + pixel_number_wedge_two)
            dF_F_array_8_roi_output[:,combined_ROI_index - 1] = dff_array_input[:,combined_ROI_index  + index_adjust- 1] * wedge1_weight +  dff_array_input[:,combined_ROI_index + index_adjust] * wedge2_weight
         
            
               
    return dF_F_array_8_roi_output
            
            
            
  