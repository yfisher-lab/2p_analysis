import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def analyze_peak_shifted_activity(df_dff_signal, bump_width, cut_indices, roi_to_degrees, peak_center_index, plot_results):
    """
    Analyzes dF/F activity by shifting peaks to a common index, computing mean activity, 
    extracting FWHM, and optionally plotting smooth curves.

    Parameters:
    - df_dff_signal: 2D NumPy array (ROIs x Frames), raw dF/F activity
    - bump_width: 1D NumPy array, precomputed bump width (FWHM) for each frame
    - cut_indices: List or array of frame indices corresponding to post-stimulation periods
    - roi_to_degrees: int, degrees per ROI (default=45)
    - peak_center_index: int, index to align peaks (default=4, meaning 5th ROI)
    - plot_results: bool, whether to plot the results (default=True)

    Returns:
    - df_dff_shifted: 2D NumPy array (ROIs x Frames), peak-centered activity
    - post_cut_mean: 1D NumPy array, mean activity per ROI after stimulation
    - random_mean: 1D NumPy array, mean activity per ROI from random sampling
    - post_cut_fwhm_mean: float, mean FWHM for post-cut frames
    - random_fwhm_mean: float, mean FWHM for random frames
    """

    # Convert to NumPy array if not already
    df_dff_array = df_dff_signal if isinstance(df_dff_signal, np.ndarray) else df_dff_signal.to_numpy()

    # Define total number of frames
    total_frames = df_dff_array.shape[1]

    # Step 1: Shift Peaks to Align Them at peak_center_index
    df_dff_shifted = np.zeros_like(df_dff_array)

    for i in range(df_dff_array.shape[1]):  # Iterate over frames
        original_order = np.array([0,1,2,3,4,5,6,7])  # Original ROI order
        current_peak = np.argmax(df_dff_array[:, i])  # Find peak in current column
        shift_by = current_peak - peak_center_index  # Align peak at desired index
        shifted_order = np.roll(original_order, -shift_by)  # Roll the array
        df_dff_shifted[:, i] = df_dff_array[shifted_order, i]

    # Step 2: Extract 30-frame activity after each post-stim index (excluding first 10 frames)
    post_cut_activity, post_cut_fwhm_values = [], []

    for cut in cut_indices:
        start_frame = cut + 10  # Start at cut index + 10
        end_frame = start_frame + 30  # Collect next 30 frames (cut+20 to cut+50)
        
        if end_frame < total_frames:
            post_cut_activity.append(df_dff_shifted[:, start_frame:end_frame])
            post_cut_fwhm_values.append(bump_width[start_frame:end_frame])

    post_cut_activity = np.hstack(post_cut_activity) if post_cut_activity else np.array([])
    post_cut_fwhm_values = np.hstack(post_cut_fwhm_values) if post_cut_fwhm_values else np.array([])

    # Step 3: Randomly sample 300 frames (excluding the first 50 frames after cut_indices)
    excluded_frames = set()
    for cut in cut_indices:
        excluded_frames.update(range(cut, cut + 40))  # Exclude cut_index to cut_index+40

    # Create list of valid frames (all frames except excluded ones)
    valid_frames = [i for i in range(total_frames) if i not in excluded_frames]

    # Ensure enough frames remain for sampling
    if len(valid_frames) < 300:
        raise ValueError("Not enough valid frames to sample 300 non-stim frames. Reduce exclusion window.")

    # Randomly select 300 valid frames
    random_frames = np.random.choice(valid_frames, 300, replace=False)
    random_activity = df_dff_shifted[:, random_frames]
    random_fwhm_values = bump_width[random_frames]

    # Step 4: Compute Mean Activity for Each ROI
    post_cut_mean = np.mean(post_cut_activity, axis=1)  # **Added Return**
    random_mean = np.mean(random_activity, axis=1)  # **Added Return**

    # Step 5: Compute Mean FWHM from `bump_width`
    post_cut_fwhm_mean = np.mean(post_cut_fwhm_values)
    random_fwhm_mean = np.mean(random_fwhm_values)

    # Step 6: Optional Plotting
    if plot_results:
        roi_labels = np.linspace(0, 360, 8, endpoint=False)  # Angles (0° to 360°)
        fine_x = np.linspace(roi_labels.min(), roi_labels.max(), 100)  # 100 points for smoothness

        interp_post_cut = interp1d(roi_labels, post_cut_mean, kind='cubic')
        interp_random = interp1d(roi_labels, random_mean, kind='cubic')

        smooth_post_cut = interp_post_cut(fine_x)
        smooth_random = interp_random(fine_x)

        plt.figure(figsize=(10, 6))
        plt.plot(fine_x, smooth_post_cut, linestyle='-', color='blue', linewidth=3, label="Post-Cut Frames")
        plt.plot(fine_x, smooth_random, linestyle='--', color='red', linewidth=3, label="Random Frames")

        plt.scatter(roi_labels, post_cut_mean, color='blue', s=80, label="Original Post-Cut Data")
        plt.scatter(roi_labels, random_mean, color='red', s=80, label="Original Random Data")

        plt.xlabel("Angle (degrees)")
        plt.ylabel("Mean dF/F Activity")
        plt.xticks(roi_labels)
        plt.title("Comparison of Post-Cut Activity vs. Random Frames")
        plt.legend()
        plt.show()

    # Step 7: Print FWHM values
    print(f"Mean FWHM of Post-Cut Distribution: {post_cut_fwhm_mean:.2f} degrees")
    print(f"Mean FWHM of Random Distribution: {random_fwhm_mean:.2f} degrees")

    # **Updated Return Statement**
    return df_dff_shifted, post_cut_mean, random_mean, post_cut_fwhm_mean, random_fwhm_mean





def process_group(folder_path, roi_to_degrees=45, peak_center_index=4):
    """
    Loads all .npz files in a folder, extracts data, and runs the analysis function.

    Parameters:
    - folder_path: str, path to the folder containing .npz files
    - roi_to_degrees: int, degrees per ROI (default=45)
    - peak_center_index: int, index to align peaks (default=4)

    Returns:
    - group_post_mean_activity: List of mean activity (per ROI) for post-stim frames
    - group_random_mean_activity: List of mean activity (per ROI) for random frames
    - mean_post_fwhm: Mean FWHM for post-stim frames
    - mean_random_fwhm: Mean FWHM for random frames
    """

    def load_data(npz_file):
        """Loads dF/F, bump width, and cut indices from an .npz file."""
        data = np.load(npz_file)
        df_dff_signal = np.vstack([data[f"dFF_Roi_{i+1}"] for i in range(8)])  # Stack 8 ROIs
        bump_width = data["Bump_width"]
        cut_indices = data["post_stim_frame"]
        return df_dff_signal, bump_width, cut_indices

    # Find all .npz files in the given folder
    npz_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".npz")]
    if not npz_files:
        raise ValueError("No .npz files found in the specified folder.")

    # Storage for results
    group_post_mean_activity = []
    group_random_mean_activity = []
    group_post_fwhm = []
    group_random_fwhm = []

    # Process each file
    for file in npz_files:
        print(f"Processing file: {file}")
        df_dff_signal, bump_width, cut_indices = load_data(file)

        # **FIXED UNPACKING HERE**
        df_dff_shifted, post_mean_activity, random_mean_activity, post_fwhm, random_fwhm = analyze_peak_shifted_activity(
            df_dff_signal, bump_width, cut_indices, roi_to_degrees, peak_center_index, plot_results=True
        )

        group_post_mean_activity.append(post_mean_activity)
        group_random_mean_activity.append(random_mean_activity)
        group_post_fwhm.append(post_fwhm)
        group_random_fwhm.append(random_fwhm)

    # Compute mean FWHM for the group
    mean_post_fwhm = np.mean(group_post_fwhm)
    mean_random_fwhm = np.mean(group_random_fwhm)

    return group_post_mean_activity, group_random_mean_activity, mean_post_fwhm, mean_random_fwhm






def plot_group_comparison(roi_to_degrees, group1_data, group2_data, group1_label, group2_label):
    """
    Plots the comparison of post-stimulation activity between two groups, including mean and SEM.

    Parameters:
    - roi_to_degrees: int, degrees per ROI (default=45)
    - group1_data: list of post-cut mean activity arrays (one per file) for group 1
    - group2_data: list of post-cut mean activity arrays (one per file) for group 2
    - group1_label: str, label for the first group
    - group2_label: str, label for the second group
    """

    # Ensure input data is a NumPy array
    group1_array = np.array(group1_data)  # Shape: (num_files, 8)
    group2_array = np.array(group2_data)  # Shape: (num_files, 8)

    # Compute Mean and SEM across files for each group
    group1_mean = np.mean(group1_array, axis=0)  # Mean across subjects
    group1_sem = np.std(group1_array, axis=0) / np.sqrt(group1_array.shape[0])  # SEM

    group2_mean = np.mean(group2_array, axis=0)
    group2_sem = np.std(group2_array, axis=0) / np.sqrt(group2_array.shape[0])

    # Define angles (0 to 360 degrees, since 8 ROIs each cover 45°)
    roi_labels = np.linspace(0, 360, 8, endpoint=False)
    fine_x = np.linspace(roi_labels.min(), roi_labels.max(), 100)  # Smooth x-axis

    # Interpolation for smooth curves
    interp_group1 = interp1d(roi_labels, group1_mean, kind='cubic')
    interp_group2 = interp1d(roi_labels, group2_mean, kind='cubic')

    smooth_group1 = interp_group1(fine_x)
    smooth_group2 = interp_group2(fine_x)

    # Plot
    plt.figure(figsize=(10, 6))

    # Plot mean curves
    plt.plot(fine_x, smooth_group1, linestyle='-', color='blue', linewidth=3, label=f"{group1_label} (Mean)")
    plt.plot(fine_x, smooth_group2, linestyle='--', color='red', linewidth=3, label=f"{group2_label} (Mean)")

    # Plot SEM as shaded areas
    plt.fill_between(roi_labels, group1_mean - group1_sem, group1_mean + group1_sem, color='blue', alpha=0.3)
    plt.fill_between(roi_labels, group2_mean - group2_sem, group2_mean + group2_sem, color='red', alpha=0.3)

    # Scatter points for raw means
    plt.scatter(roi_labels, group1_mean, color='blue', s=80, label=f"{group1_label} (Raw Mean)")
    plt.scatter(roi_labels, group2_mean, color='red', s=80, label=f"{group2_label} (Raw Mean)")

    # Plot settings
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Mean dF/F Activity (Post-Stim)")
    plt.xticks(roi_labels)
    plt.title(f"Group Comparison: {group1_label} vs. {group2_label}")
    plt.legend()
    plt.show()


