import numpy as np
import os
from os.path import sep
from matplotlib import pyplot as plt




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