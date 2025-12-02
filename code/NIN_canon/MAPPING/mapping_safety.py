# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 18:25:44 2024

@author: Lozano



 >>> IN CONSTRUCTION
 
 
 The "mapping.py" script is meant to illustrate, clarify and make possible the mapping between
 Jump, channel, instance, electrode, and stimulator channel IDs.
 

  TODO: THE STIMBANKS AND OTHER PARTS STILL NEED REORDERING IN THE RIGHT SIDE OF THE H
  (CHANNELS HIGHER THAN 512)


  TODO Use fastplotlib to:
      Show neural traces for many trials in Exp1, all channels at once, 
      Show neural traces but in an image, like in the LFP paper, too
      Show movie of the electrodes in cortex and in RF locations, with their neural responses in time,
      at the same time. Maybe in the stimulating array you can plot also the stim signals?
      
      
      Do this FIRST with the visual trials data, you can also plot the RFs obtained with the mapping
      experiments
      
      Use jupyterlab demos if its useful https://github.com/fastplotlib





    
%%%%%%%%%%%%%%%%%%%% Some info* (also in README):
   
    --- SYSTEM ---
    We have up to 1024 recording channels:
        recorded by the 16 arrays in the monkey's cortex,
        each array having a total of 64 electrodes, connected to 8 recording instances, with 4 cereplexM bank,
        with 32 channels per bank, a total of 128 channels per instance.
        The total of 32 banks (4 per instance) are connected physically through 32 Jumps,
        each two jumps having 32 jump positions connected to each array.
        When connected, the total of 16 stimulators wich A and B banks, have 32 channels per bank,
        this is, 64 channels per stimulator are connected to each array through two jumps with 32 positions per jump.
        That means that each of the stimulator channels are mapped to specific electrodes in the arrays and have 
        a corresponding recording channel within the banks in the instances
    
    
    --- JUMPS ---
    We have 32 Jumps. 
    Each jumpIndex which goes from 1 to 1024, in this order, and
    has a respective jumpNumber from 1 to 32, because it belongs to one of these jumps
    Each jumpIndex has their associated instanceChannels_128, instanceChannels_1024 channels,
    which is key to understand this mapping.
    
    
    --- INSTANCES ---
    We have 8 recording instances that in total record from 1028 channels
    Each instance has 4 banks: A, B, C and D
    Each bank has 32 channels, and in total they make 128 channels per instance.
    Bank A has channels 1 to 32
    Bank B has channels 33 to 64
    Bank C has channels 65 to 96
    Bank D has channels 97 to 128
    
    Each instance has its banks connected in the following order: DABC, which makes
    a very specific mapping between channel (recorded by instances through bankds), jump numbers, electrode
    numbers and stimulator channel numbers necessary to be clarified.
    
    Each channel has a corresponding jumpIndex number which goes from 1 to 1024
    Each channel has a corresponding jumpNumber number which goes from 1 to 32
    Each channel has a corresponding instanceChannels_128 number which goes from 1 to 128
    Each channel has a corresponding instanceChannels_1024 number  which goes from 1 to 1024
    Each channel has a corresponding instanceNumbers number which goes from 1 to 8
    Each channel has a corresponding electrodeNumbers number which goes from 1 to 64
    Each channel has a corresponding arrayNumbers number which goes from 1 to 16
    Each channel has a corresponding cereM_banks letter (D x32, A x32, Bx32 and Cx32) repeated 8 times, one per instance. 
    This order is key to understand the mapping
    Each channel has a corresponding stimBanks letter (A x32, B x32) repeated 16 times, one per stimulator. 
    This order is key to understand the mapping
    
    
    --- STIMULATORS ---
    We have up to 16 stimulators, with stimBanks A, B,
    Each Stim bank has stimChannels 1 to 32 and 33 to 64 respectively
    Each stimChannel has stimNumbers from 1 to 16 respectively

    
    --- ARRAYS ---
    We have 16 arrays. Each array has 64 electrodes.
    

%%%%%%%%%%%%%%%%%%%% CHATGPT 4.0 SUMMARY OF THE INFORMATION ABOVE

System Overview
Recording Channels: 
    The system supports up to 1024 recording channels across the monkey's cortex, captured by 16 arrays. 
    Each array consists of 64 electrodes.

Instances and Banks: 
    There are 8 recording instances in total, each with 4 banks (A, B, C, D), 
    contributing to 128 channels per instance. The unique order of these banks (DABC) is crucial for the mapping strategy.

Stimulators: 
    The system includes up to 16 stimulators, each with A and B banks, supporting 64 channels per stimulator. 
    These channels are intricately mapped to specific electrodes within the arrays.


Jumps: 
    The architecture utilizes 32 Jumps to physically connect the components. 
    Each jump index ranges from 1 to 1024 and corresponds to a jump number from 1 to 32. 
    These jumps are crucial for understanding the channel-to-electrode mapping.

Instances
    Configuration: 
        Each instance's 4 banks (A, B, C, D) collectively support 128 channels, 
        with a specific sequence (DABC) for connecting these banks.

    Channel Mapping: 
        Every channel within the system is associated with a specific jump index and number, 
        instance channel numbers (both within 128 and 1024 scales), instance number (1 to 8), electrode number (1 to 64), and array number (1 to 16). The order of the cereM_banks (D, A, B, C) repeated across instances and the stimBanks (A, B) repeated across stimulators are pivotal for the mapping.

Stimulators
    Bank Configuration: 
        Each stimulator's banks (A, B) have channels numbered 1 to 32 and 33 to 64, respectively. 
        These channels correspond to stim numbers from 1 to 16.
    Arrays
        Electrode Configuration: There are 16 arrays, each equipped with 64 electrodes, 
        facilitating extensive neural recording coverage.



* This information is valid only with the specific physical connections
  done for Experiment 1, and those that use the same physical connection
  
  
  
  
"""



#           IMPORTANT: 
    
    # CHECK AGAINST PAOLOS MAPPING
    
    # THIS IS PAOLO'S MAPPING  load('\\vs03\vs03-vandc-1\Latents\Passive_Fixation\monkeyN\_logs\1024chns_mapping_20220105.mat')



#%%
import os
import numpy as np
import scipy.io
import pickle
import matplotlib.pyplot as plt

rootpath = r"\\vs03\VS03-VandC-INTENSE2\mind_writing\data_analysis\MAPPING"
os.chdir(rootpath)

%matplotlib inline

#%% Function definition

def convert_mat_struct_to_dict(mat_struct):
    """
    Convert a MATLAB struct loaded via scipy.io.loadmat into a Python dictionary.
    
    Parameters:
    - mat_struct: The MATLAB struct loaded as a NumPy structured array.
    
    Returns:
    - A Python dictionary representing the MATLAB struct.
    """
    data_dict = {}
    
    # Iterate through each field in the MATLAB struct
    for field in mat_struct.dtype.names:
        field_data = mat_struct[field][0, 0]
        # Check if the field data is a structured array itself (indicative of nested structs)
        if field_data.dtype.kind == 'V':
            # Recursively convert nested structs
            data_dict[field] = convert_mat_struct_to_dict(field_data)
        else:
            # Directly assign if not a struct (handles arrays, strings, etc.)
            data_dict[field] = field_data[0]
    
    return data_dict

#%% Creating colors per array (to be reordered later to match the array order)

# Colors for 16 arrays in LICK
colsL = [(1.0000, 0,  0,), (1.0000, 0.3750, 0), (1.0000, 0.7500, 0), (0.8750, 1.0000, 0), (0.5000, 1.0000, 0), (0.1250, 1.0000, 0), (0, 1.0000, 0.2500), (0.5451, 0.2706, 0.0745),
         (0, 1.0000, 1.0000), (0, 0.6250, 1.0000), (0, 0.2500, 1.0000), (0.1250, 0, 1.0000), (0.5000, 0, 1.0000), (0.8750, 0, 1.0000), (1.0000, 0, 0.7500), (1.0000, 0, 0.3750)]
colors = []
allColors = []
#repeatinag color for each array
for i in range(16):
    colors.append(colsL[i])
    for j in range(64):
        allColors.append(colsL[i])

colors = np.array(colors)     
allColors = np.array(allColors) 

#%% Load channel maps from matlab to python

# Assuming 'rootpath' is already defined in your code.
maps_filepath = rootpath + r"/data_mapping"  # Use forward slash for compatibility with different OS
channel_map_exp1_filename = maps_filepath + r"/channel_map_Exp1_ERF_MrNilson.mat"  # Same here
# Load the .mat file
channel_map_exp1 = scipy.io.loadmat(channel_map_exp1_filename)['map']
# Assuming `channel_map_exp1` is the structured array loaded from the .mat file
channel_map_exp1_dict = convert_mat_struct_to_dict(channel_map_exp1)
print(channel_map_exp1_dict.keys())

#%% Adding new entries with numbers 1, 2, 3 and 4 for cereM banks A, B, C and D,
#   also adding entries with values 1 and 2 for stimBanks A and B 

# Dictionary mapping from cereM_banks letters to numbers
cereM_banks_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4}
cereM_banks_values = channel_map_exp1_dict['cereM_banks']
cereM_banks_numbers = [cereM_banks_mapping[val] for val in cereM_banks_values]
channel_map_exp1_dict['cereM_banks_number'] = cereM_banks_numbers

# Dictionary mapping from stimBanks letters to numbers
stimBanks_mapping = {'A': 1, 'B': 2}
stimBanks_values = channel_map_exp1_dict['stimBanks']
stimBanks_numbers = [stimBanks_mapping[val] for val in stimBanks_values]
channel_map_exp1_dict['stimBanks_number'] = stimBanks_numbers

#%% Adding X, Y locations of every electrode in the cortex (from arrays_in_cortex_mr_nilson.py)

array_cortex_info_path = r"\\vs03\VS03-VandC-INTENSE2\mind_writing\data_analysis\MAPPING\results\cortical_array_details.pkl"
# Loading array_details from the saved file
with open(array_cortex_info_path, 'rb') as file:
    array_cortex_info = pickle.load(file)
print("Data loaded successfully.")

# Test to ensure the data is intact
print(array_cortex_info.keys())  # This will print the keys of the dictionary to verify it's loaded correctly
print(array_cortex_info[1].keys()) 
print(array_cortex_info[1]['Utah Array Labels']) 
print(array_cortex_info[1]['Utah Array Positions']) 
print(array_cortex_info.keys())  # This will print the keys of the dictionary to verify it's loaded correctly

#%% Store XY cortical positions into the dictionary

# Initialize a new key in channel_map_exp1_dict for storing XY positions
channel_map_exp1_dict['electrodeXYPositions'] = np.empty((len(channel_map_exp1_dict['electrodeNumbers']), 2), dtype=float)

for index, electrode_number in enumerate(channel_map_exp1_dict['electrodeNumbers']):
    array_number = channel_map_exp1_dict['arrayNumbers'][index]  # Get the array number for this electrode
    # Find the corresponding surgery label in array_cortex_info
    for surgery_label, array_info in array_cortex_info.items():
        if array_info['Surgery Label'] == array_number:
            # Now find the matching Utah array label to get the XY position
            utah_labels = array_info['Utah Array Labels'].flatten()
            utah_positions = array_info['Utah Array Positions'].reshape(-1, 2)  # Reshape for easier indexing
            
            label_index = np.where(utah_labels == electrode_number)[0]
            if label_index.size > 0:  # Check if the electrode number was found in the Utah labels
                # Update the XY positions for this electrode in channel_map_exp1_dict
                channel_map_exp1_dict['electrodeXYPositions'][index] = utah_positions[label_index][0]
                break  # Move to the next electrode number once a match is found



#%% Plotting content of the dict
for key, value in channel_map_exp1_dict.items():
    print(f"Key: {key}")

    # print(value[-128:])
    print(value[-128:])
    
    fig, ax = plt.subplots(dpi = 200)
    plt.plot(value)
    plt.title(key)
    # Turn off the right and left spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    # Show the plot
    plt.show()

#%% Reorder array colors and add them to the dictionary
reorderedColors = np.zeros_like(allColors)
for i, array_num in enumerate(channel_map_exp1_dict['arrayNumbers']):
    # array_num - 1 because arrayNumbers starts from 1, but indexing in Python starts from 0
    reorderedColors[i] = colors[array_num - 1]
channel_map_exp1_dict['arrayColor'] = reorderedColors

#% Creating an alpha value per jump number. Odd jumps have alpha = 0.5, even jumps have alpha = 1
jump_numbers = channel_map_exp1_dict['jumpNumber']
alpha_values = np.zeros_like(jump_numbers, dtype=float)
for i, jump_num in enumerate(jump_numbers):
    alpha_values[i] = 0.8 if jump_num % 2 != 0 else 1.0
channel_map_exp1_dict['alphaValue'] = alpha_values

#%% -------------- IMPORTANT, CORRECTING THE INSTANCE CHANNELS 1024 NUMBERS for the right 
#   half of the 'H' (e.g. channels higher than 512)
import numpy as np
# Assuming the initial setup is already provided as above
import numpy as np

# Extend the existing setup with cereM_banks_letter array
cereM_banks_letter = np.array(channel_map_exp1_dict['cereM_banks'])
instanceChannels_128 = np.array(channel_map_exp1_dict['instanceChannels_128'])
instanceChannels_1024 = np.array(channel_map_exp1_dict['instanceChannels_1024'])

cereM_banks_number = np.array(channel_map_exp1_dict['cereM_banks_number'])

# Convert cereM_banks_letter from a single string to an array of individual characters
cereM_banks_letter = np.array(list(channel_map_exp1_dict['cereM_banks']))

# Proceed with the existing logic, now that cereM_banks_letter is correctly formatted
instance_size = 128  # Number of channels per instance
affected_instances_start = 4 * instance_size  # Starting index for affected instances (5-8), zero-based

# Apply reordering for affected instances
for instance_index in range(affected_instances_start, len(instanceChannels_1024), instance_size):
    # instance_slice_128 = slice(instance_index, instance_index + instance_size)
    instance_slice_1024 = slice(instance_index, instance_index + instance_size)

    current_instance_128 = instanceChannels_128[instance_slice_1024]
    current_instance_1024 = instanceChannels_1024[instance_slice_1024]
    current_cereM_banks_number = cereM_banks_number[instance_slice_1024]
    current_cereM_banks_letter = cereM_banks_letter[instance_slice_1024]

    idx_A = np.where (current_cereM_banks_letter == 'A')[0]
    idx_B = np.where (current_cereM_banks_letter == 'B')[0]
    idx_C = np.where (current_cereM_banks_letter == 'C')[0]
    idx_D = np.where (current_cereM_banks_letter == 'D')[0]

    instanceChannels_128_A = current_instance_128[idx_A]
    instanceChannels_128_B = current_instance_128[idx_B]
    instanceChannels_128_C = current_instance_128[idx_C]
    instanceChannels_128_D = current_instance_128[idx_D]
    
    reordered_instance_128 = np.concatenate([
        instanceChannels_128_C,  # C
        instanceChannels_128_B,  # B
        instanceChannels_128_A,  # A
        instanceChannels_128_D,  # D
    ])
    
    reordered_instance_1024 = reordered_instance_128 + instance_index
    
    reordered_cereM_banks_number = np.concatenate([
        np.full(32, 3),  # C
        np.full(32, 2),  # B
        np.full(32, 1),  # A
        np.full(32, 4),  # D
    ])
    
    reordered_cereM_banks_letter = np.concatenate([
        np.full(32, 'C'),  # C
        np.full(32, 'B'),  # B
        np.full(32, 'A'),  # A
        np.full(32, 'D'),  # D
    ])

    instanceChannels_128[instance_slice_1024] = reordered_instance_128
    instanceChannels_1024[instance_slice_1024] = reordered_instance_1024
    cereM_banks_number[instance_slice_1024] = reordered_cereM_banks_number
    cereM_banks_letter[instance_slice_1024] = reordered_cereM_banks_letter

# Update the dictionary after reordering
channel_map_exp1_dict['instanceChannels_128'] = instanceChannels_128
channel_map_exp1_dict['instanceChannels_1024'] = instanceChannels_1024
channel_map_exp1_dict['cereM_banks_number'] = cereM_banks_number
channel_map_exp1_dict['cereM_banks'] = ''.join(cereM_banks_letter)  # If you want to store it back as a string

#%% Plotting content of the dict again
for key, value in channel_map_exp1_dict.items():
    print(f"Key: {key}")

    # print(value[-128:])
    print(value[-128:])
    
    fig, ax = plt.subplots(dpi = 200)
    plt.plot(value)
    plt.title(key)
    # Turn off the right and left spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    # Show the plot
    plt.show()

#%% Plotting cortical locations


import matplotlib.pyplot as plt
PLOT = False

if PLOT:
    # Precompute constants
    electrode_positions = channel_map_exp1_dict['electrodeXYPositions']
    array_colors = channel_map_exp1_dict['arrayColor']
    alpha_values = channel_map_exp1_dict['alphaValue']
    
    for label_key in channel_map_exp1_dict.keys():
        # Create a new figure for each key
        plt.figure(dpi=200)  # Lower DPI for development; increase for final plots
        ax = plt.gca()

        label_values = channel_map_exp1_dict[label_key]
        
        # Vectorized plotting of points
        x_vals, y_vals = zip(*electrode_positions)
        colors = array_colors
        alphas = [alpha / 3 for alpha in alpha_values]  # Adjust alpha values for plot
        ax.scatter(x_vals, [-y for y in y_vals], c=colors, alpha=alphas, s=20, edgecolor='black')
        
        # Annotating each point
        for (x, y), label in zip(electrode_positions, label_values):
            ax.text(x, -y, str(label), color='black', fontsize=3, fontweight='bold', ha='center', va='center')
        
        plt.title(f'Cortical Locations with {label_key.capitalize()} Labels')
        plt.xlabel('Cortical X Position')
        plt.ylabel('Cortical Y Position')
        plt.axis('equal')
        plt.tight_layout()
        
        plt.show()  # Make sure to display each plot
            
#%% Adding RF data to the mapping dict

# filename = r'C:\Users\Lozano\Desktop\NIN\Mr Nilson\RF_mapping_SparseNoise\code\analyze_sparseNoise_AL\combined_data_NO_PREDICTIONS.pkl'
# # filename = r'C:\Users\Lozano\Desktop\NIN\Mr Nilson\RF_mapping_SparseNoise\code\analyze_sparseNoise_AL\combined_data_PREDICTIONS.pkl'
# # Load combined_data from the file
# with open(filename, 'rb') as f:
#     combined_data = pickle.load(f)

# print("combined_data loaded successfully.")

# # Print the keys of loaded_combined_data
# # keys are dict_keys(['RFX', 'RFY', 'STDX', 'STDY', 'R2'])

# print(combined_data.keys())
# # Accessing the RFX_combined data
# # rfx_combined_data = loaded_combined_data['RFX_combined']# THIS IS IN PREDICTIONS
# rfx_combined_data = combined_data['RFX']# THIS IS IN NO PREDICTIONS

# # Now you can work with rfx_combined_data as needed, such as displaying its shape or performing further analysis
# print(rfx_combined_data.shape)

# Assuming channel_map_exp1_dict is already loaded and available


#%% Choosing RF data to be used

RF_TYPE = 'THINGS' # THINGS, BARS_S_NOISE, MATT
RF_TYPE = 'MATT' # THINGS, BARS_S_NOISE
RF_TYPE = 'BARS_S_NOISE' # THINGS, BARS_S_NOISE

if RF_TYPE == 'BARS_S_NOISE':
    # ---- COMBINED DATA BARS AND SPARSE NOISE ----
    # Load combined_data containing the RF data
    rf_data_filename = r'C:\Users\Lozano\Desktop\NIN\Mr Nilson\RF_mapping_SparseNoise\code\analyze_sparseNoise_AL\combined_data_NO_PREDICTIONS.pkl'
    with open(rf_data_filename, 'rb') as f:
        combined_data = pickle.load(f)
        print(combined_data.keys())
    print("RF combined_data loaded successfully.")
    
elif RF_TYPE == 'THINGS':
    # ---- RF_THINGS natural images data ----
    import scipy.io
    PIX_PER_DEG_THINGS = 25.8601

    # Define the path to your .mat file
    # mat_file_path = r'\\vs03\VS03-VandC-INTENSE2\mind_writing\data_analysis\RFs_combined_bars_noise_and_THINGS\RF_MrNilson_THINGS\THINGS_RFs.mat'
    mat_file_path = r'\\vs03\VS03-VandC-INTENSE2\mind_writing\data_analysis\RFs_combined_bars_noise_and_THINGS\RF_MrNilson_THINGS\THINGS_RF1s.mat'
    
    # Load the RF THINGS .mat file
    data = scipy.io.loadmat(mat_file_path)
    print(data.keys())
        
    # load the bars data dictionary so we can harmonize the keys
    rf_data_filename = r'C:\Users\Lozano\Desktop\NIN\Mr Nilson\RF_mapping_SparseNoise\code\analyze_sparseNoise_AL\combined_data_NO_PREDICTIONS.pkl'
    with open(rf_data_filename, 'rb') as f:
        combined_data = pickle.load(f)
        print(combined_data.keys())
    
    # Now let's harmonize keys and update combined_data with the data from the THINGS dict
    # Mapping from the keys in 'data' (THINGS data) to the keys in 'combined_data'
    key_mapping = {
        'all_centrex': 'RFX',
        'all_centrey': 'RFY',
        'all_szx': 'STDX',
        'all_szy': 'STDY',
        'all_test_corr': 'R2'
    }
    
    new_combined_data = combined_data.copy()

    # Update the combined_data with the THINGS data using the harmonized keys
    for key in key_mapping:
        if key in data:
            print(key)
            # # transform into degrees (THINGS data is in pixels)
            if key != 'all_test_corr':
                data[key] = data[key] / PIX_PER_DEG_THINGS
            
            else:
                data[key] = data[key]
            print( np.squeeze(data[key])[0:5])
            print(  )
            new_combined_data[key_mapping[key]] = np.squeeze(data[key])
    combined_data = new_combined_data
        
#% New mapping from Paolo & Matt
elif RF_TYPE == 'MATT':
    
    # =============================================================================
    #   LOADING BARS for V1
    # =============================================================================
    # Define the path to your .mat file
    mat_file_path = r'\\vs03\VS03-VandC-INTENSE2\mind_writing\data_analysis\ERFs\Matt\BarMap_Nilson.mat'
    
    # Load the RF data from the .mat file
    data = scipy.io.loadmat(mat_file_path)
    print(data.keys())
    
    # Access the 'RF' structured array directly
    rf_struct = data['RF']
    
    # Convert the MATLAB struct to a Python dict
    rf_data = {field: rf_struct[field][0, 0].squeeze() for field in rf_struct.dtype.names}
    
    print(rf_data.keys())  # This should show keys like 'centrex', 'centrey', etc.
    
    # Create 'R2' filled with NaNs of the correct length
    r2_placeholder = np.full_like(rf_data['centrex'], np.nan, dtype=np.float)
    
    # Harmonize the keys with your existing data structure
    combined_data = {
        'RFX': rf_data['centrex'],
        'RFY': rf_data['centrey'],
        'STDX': rf_data['sz'],  # Assuming 'sz' maps directly to standard deviation in your existing data
        'STDY': rf_data['sz'],  # Same as above, adjust if your data differentiates between X and Y
        'R2': r2_placeholder  # Placeholder for 'R2' with NaN values
    }
    
    # Convert sizes from pixels to degrees if 'pixperdeg' is available
    if 'pixperdeg' in data:
        pix_per_deg = data['pixperdeg'][0, 0]  # Assuming it's a scalar
        combined_data['STDX'] /= pix_per_deg
        combined_data['STDY'] /= pix_per_deg
        
    # =============================================================================
    #   LOADING SPARSE NOISE for V4
    # =============================================================================
    # Define the path to your .mat file
    import scipy.io
    import numpy as np
    
    # Load the existing combined data, assuming it's stored in a variable named `combined_data`
    
    # Define the path to the new .mat file
    grid_map_path = r'\\vs03\VS03-VandC-INTENSE2\mind_writing\data_analysis\ERFs\Matt\GridMap_Nilson.mat'
    
    # Load the Grid Map data from the .mat file
    grid_data = scipy.io.loadmat(grid_map_path)
    print(grid_data.keys())
    
    # Access the 'GRF' structured array directly
    grf_struct = grid_data['GRF']
    
    # Convert the MATLAB struct to a Python dictionary
    grf_data = {field: grf_struct[field][0, 0].squeeze() for field in grf_struct.dtype.names}
    
    # Concatenating this new data with the existing data in combined_data
    for key in ['RFX', 'RFY', 'STDX', 'STDY', 'R2']:  # Assuming these fields exist in combined_data
        if key == 'RFX':
            combined_data[key] = np.concatenate((combined_data[key], grf_data['ctx']))
        elif key == 'RFY':
            combined_data[key] = np.concatenate((combined_data[key], grf_data['cty']))
        elif key == 'STDX':
            combined_data[key] = np.concatenate((combined_data[key], grf_data['xsz']))
        elif key == 'STDY':
            combined_data[key] = np.concatenate((combined_data[key], grf_data['ysz']))
        elif key == 'R2':  # If you need to create a placeholder for R2
            new_r2 = np.full_like(grf_data['ctx'], np.nan)  # Create a NaN-filled array for new R2 data
            combined_data[key] = np.concatenate((combined_data[key], new_r2))
    
    print("Updated combined_data with new GRF data:")
    print({k: v[-5:] for k, v in combined_data.items()})  # Printing only the last 5 entries for brevity
    
    # =============================================================================
    #  FILLING WITH NANS to get 1024, since this is data for V1 and V4 only
    # =============================================================================
    
    # Assuming 'combined_data' is your existing dictionary
    # The keys we are padding: 'RFX', 'RFY', 'STDX', 'STDY', 'R2'
    
    # Determine the target length for padding
    target_length = 1024
    
    # Loop through each key and pad with NaNs if necessary
    for key in ['RFX', 'RFY', 'STDX', 'STDY', 'R2']:
        current_length = combined_data[key].shape[0]
        if current_length < target_length:
            # Calculate how many NaNs are needed
            padding_length = target_length - current_length
            # Create a NaN-filled array for padding
            nan_padding = np.full(padding_length, np.nan)
            # Concatenate the existing data with the NaN padding
            combined_data[key] = np.concatenate((combined_data[key], nan_padding))
    
    # To check the final lengths and last few elements:
    print("Updated combined_data with padded NaN values:")
    for key in ['RFX', 'RFY', 'STDX', 'STDY', 'R2']:
        print(f"{key}: Length = {len(combined_data[key])}, Last 5 entries = {combined_data[key][-5:]}")



#%% Integrating the RF data from the loaded dictionaries into our mapping dict

# Initialize new entries in channel_map_exp1_dict for RF data, filled with NaNs initially
# This handles cases where RF data might not be available for all channels
channel_map_exp1_dict['RFX'] = np.full_like(channel_map_exp1_dict['instanceChannels_1024'], np.nan, dtype=np.float)
channel_map_exp1_dict['RFY'] = np.full_like(channel_map_exp1_dict['instanceChannels_1024'], np.nan, dtype=np.float)
channel_map_exp1_dict['STDX'] = np.full_like(channel_map_exp1_dict['instanceChannels_1024'], np.nan, dtype=np.float)
channel_map_exp1_dict['STDY'] = np.full_like(channel_map_exp1_dict['instanceChannels_1024'], np.nan, dtype=np.float)
channel_map_exp1_dict['R2'] = np.full_like(channel_map_exp1_dict['instanceChannels_1024'], np.nan, dtype=np.float)

# For each channel in instanceChannels_1024, find the corresponding RF data
# The channel number in instanceChannels_1024 directly corresponds to the index in the RF data arrays
for idx, channel in enumerate(channel_map_exp1_dict['instanceChannels_1024']):
    print(idx, channel)

    # The RF data index is channel - 1 because Python indexing starts at 0
    rf_index = channel - 1  # Adjusting for 0-based indexing in Python
    channel_map_exp1_dict['RFX'][idx] = combined_data['RFX'][rf_index]
    channel_map_exp1_dict['RFY'][idx] = combined_data['RFY'][rf_index]
    channel_map_exp1_dict['STDX'][idx] = combined_data['STDX'][rf_index]
    channel_map_exp1_dict['STDY'][idx] = combined_data['STDY'][rf_index]
    channel_map_exp1_dict['R2'][idx] = combined_data['R2'][rf_index]

# Verify integration by checking the shape of the corrected arrays
print("RF data correctly added to channel_map_exp1_dict.")
for key in ['RFX', 'RFY', 'STDX', 'STDY', 'R2']:
    print(f"{key}: shape {channel_map_exp1_dict[key].shape}")

#%% Add area labels
# Initialize an empty list to hold the area designations for each channel
area_assignments = []

# Iterate over each array number in the dictionary and assign the area based on the array number
for array_number in channel_map_exp1_dict['arrayNumbers']:
    if array_number in range(1, 3):  # Arrays 1 and 2
        area_assignments.append('V2')  # Corrected to V2 as per the updated instructions
    elif array_number in range(3, 9):  # Arrays 3 to 8
        area_assignments.append('V1')
    elif array_number in range(9, 13):  # Arrays 9 to 12
        area_assignments.append('V4')
    elif array_number > 12:  # Arrays greater than 12
        area_assignments.append('IT')
    else:  # Fallback in case of an unexpected value
        area_assignments.append('Unknown')

# Add the area assignments to the dictionary
channel_map_exp1_dict['Area'] = area_assignments

# To verify, you can print or inspect the first few entries of the 'Area' field
print(channel_map_exp1_dict['Area'][:128*2])


#%%
import matplotlib.pyplot as plt

# Define colors for each area
area_colors = {'V2': 'green', 'V1': 'blue', 'V4': 'red', 'IT': 'purple'}

# Iterate through the unique areas to create a scatter plot for each
for area in set(channel_map_exp1_dict['Area']):
    plt.figure(figsize=(8, 5))  # Create a new figure for each area
    
    # Get the indices of electrodes that belong to the current area
    indices = [i for i, a in enumerate(channel_map_exp1_dict['Area']) if a == area]
    
    # Extract the RF locations for these indices
    rfx = channel_map_exp1_dict['RFX'][indices]
    rfy = channel_map_exp1_dict['RFY'][indices]
    colors = channel_map_exp1_dict['arrayColor'][indices]
    
    # Plot the RF locations with the color specified for the area
    # plt.scatter(rfx, rfy, color=area_colors[area], label=f"Area {area}", alpha=0.6, edgecolors='none')
    plt.scatter(rfx, rfy, color=colors, label=f"Area {area}", alpha=0.9, s = 60,edgecolors='black')
    plt.axvline(0, color='gray', linestyle='dashed', linewidth=0.5)
    plt.axhline(0, color='gray', linestyle='dashed', linewidth=0.5)

    # Enhance the plot for the current area
    plt.xlabel('RF X Location')
    plt.xlim(-1,7)
    plt.ylim(-8,5)
    

    plt.ylabel('RF Y Location')
    plt.title(f'Receptive Field Locations in Area {area}')
    plt.legend()
    plt.grid(False)
    # plt.axis('equal')  # Set equal scaling by changing axis limits
    
    # Show the plot for the current area
    plt.show()

#%%import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Define colors for each area for consistency in visualization
area_colors = {'V2': 'green', 'V1': 'blue', 'V4': 'red', 'IT': 'purple'}

# Areas to iterate over
areas = ['V1', 'V2', 'V4', 'IT']

# Increase default font sizes for all plot elements for readability
plt.rcParams.update({'font.size': 16, 'axes.titlesize': 18, 'axes.labelsize': 16,
                     'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 14})

# Calculate RF eccentricity and polar angle for each channel
eccentricity = np.sqrt(np.array(channel_map_exp1_dict['RFX'])**2 + np.array(channel_map_exp1_dict['RFY'])**2)
polar_angle = np.arctan2(np.array(channel_map_exp1_dict['RFY']), np.array(channel_map_exp1_dict['RFX'])) * (180 / np.pi) # Convert to degrees

# Assuming STDX represents the size of the receptive field
rf_size = np.array(channel_map_exp1_dict['STDX'] + channel_map_exp1_dict['STDY'] ) /2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Convert the 'Area' list of strings to a numpy array for vectorized operations
areas_array = np.array(channel_map_exp1_dict['Area'])

for area in ['V1', 'V2', 'V4', 'IT']:
    # Use numpy's vectorized comparison for element-wise string comparison
    area_indices = (areas_array == area)
    
    # Also ensure we select only entries that have finite eccentricity and rf_size
    valid_indices = area_indices & np.isfinite(eccentricity) & np.isfinite(rf_size)
    
    # Now valid_indices correctly identifies the entries for the current area with valid data
    if np.sum(valid_indices) > 0:  # Proceed only if there are valid data points
        area_eccentricity = eccentricity[valid_indices]
        area_rf_size = rf_size[valid_indices]
        
        # Fit Linear Regression model
        model = LinearRegression()
        model.fit(area_eccentricity.reshape(-1, 1), area_rf_size)
        
        slope = model.coef_[0]
        intercept = model.intercept_
        
        # Prepare x values for the regression line
        x_vals = np.linspace(area_eccentricity.min(), area_eccentricity.max(), 100)
        y_vals = intercept + slope * x_vals  # Calculate y values based on the regression
        
        # Plotting
        plt.figure(figsize=(8, 6))
        plt.scatter(area_eccentricity, area_rf_size, color=area_colors[area], alpha=0.6, label=f'{area} Data')
        plt.plot(x_vals, y_vals, 'k--', label=f'Linear Reg. Slope: {slope:.2f}, Intercept: {intercept:.2f}')
        
        plt.title(f'RF Eccentricity vs. Size in {area}')
        plt.xlabel('RF Eccentricity (degrees)')
        plt.ylabel('RF Size (STDX)')
        plt.legend()
        plt.show()
    else:
        print(f"No valid data points found for area {area}. Skipping...")



#%% Predicting rest of RF values

from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
areas = ['V1', 'V2', 'V4', 'IT']


areas = ['V1', 'V2', 'V4', 'IT']

# # Initialize the predicted fields to NaN, ensuring they cover all indices
# channel_map_exp1_dict['RFX_predicted'] = np.full_like(channel_map_exp1_dict['RFX'], np.nan)
# channel_map_exp1_dict['RFY_predicted'] = np.full_like(channel_map_exp1_dict['RFY'], np.nan)

# for area in areas:
#     # Identifying known and unknown indices for the current area
#     area_indices = np.array(channel_map_exp1_dict['Area']) == area
#     known_indices = np.isfinite(channel_map_exp1_dict['RFX']) & area_indices
#     unknown_indices = ~np.isfinite(channel_map_exp1_dict['RFX']) & area_indices
    
#     # Fill in known values directly into the predicted fields
#     channel_map_exp1_dict['RFX_predicted'][known_indices] = channel_map_exp1_dict['RFX'][known_indices]
#     channel_map_exp1_dict['RFY_predicted'][known_indices] = channel_map_exp1_dict['RFY'][known_indices]

#     # Proceed only if there are known RFX, RFY values for training
#     # if known_indices.any():
#     # Preparing training data
#     X_train = channel_map_exp1_dict['electrodeXYPositions'][known_indices]
#     Y_train_rfx = channel_map_exp1_dict['RFX'][known_indices]
#     Y_train_rfy = channel_map_exp1_dict['RFY'][known_indices]

#     # Training models for RFX and RFY
#     model_rfx = make_pipeline(PolynomialFeatures(degree=1), RidgeCV(alphas=np.linspace(0.01, 10, 100)))
#     model_rfx.fit(X_train, Y_train_rfx)
    
#     model_rfy = make_pipeline(PolynomialFeatures(degree=1), RidgeCV(alphas=np.linspace(0.01, 10, 100)))

#     model_rfy.fit(X_train, Y_train_rfy)

#     # Predicting for unknown values, if any
#     # if unknown_indices.any():
#     X_unknown = channel_map_exp1_dict['electrodeXYPositions'][unknown_indices]
#     channel_map_exp1_dict['RFX_predicted'][unknown_indices] = model_rfx.predict(X_unknown)
#     channel_map_exp1_dict['RFY_predicted'][unknown_indices] = model_rfy.predict(X_unknown)

from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.pipeline import make_pipeline
import numpy as np

# areas = ['V1', 'V2', 'V4', 'IT']

# # Initialize the predicted fields to NaN
# channel_map_exp1_dict['RFX_predicted'] = np.full_like(channel_map_exp1_dict['RFX'], np.nan)
# channel_map_exp1_dict['RFY_predicted'] = np.full_like(channel_map_exp1_dict['RFY'], np.nan)

# for area in areas:
#     area_indices = np.array(channel_map_exp1_dict['Area']) == area
#     known_indices = np.isfinite(channel_map_exp1_dict['RFX']) & area_indices
#     unknown_indices = ~np.isfinite(channel_map_exp1_dict['RFX']) & area_indices
    
#     # Copy known RFX and RFY directly into the predicted fields
#     channel_map_exp1_dict['RFX_predicted'][known_indices] = channel_map_exp1_dict['RFX'][known_indices]
#     channel_map_exp1_dict['RFY_predicted'][known_indices] = channel_map_exp1_dict['RFY'][known_indices]

#     if known_indices.any():
#         X_train = channel_map_exp1_dict['electrodeXYPositions'][known_indices]
#         Y_train_rfx = channel_map_exp1_dict['RFX'][known_indices]
#         Y_train_rfy = channel_map_exp1_dict['RFY'][known_indices]

#         # ------ polinomial REGRESSOR

#         # Define and fit models for RFX and RFY with normalization
#         # Including MinMaxScaler in the pipeline
#         model_rfx = make_pipeline(MinMaxScaler(), PolynomialFeatures(degree=4), RidgeCV(alphas=np.linspace(0.001, 100, 1000)))
#         model_rfx.fit(X_train, Y_train_rfx)
        
#         model_rfy = make_pipeline(MinMaxScaler(), PolynomialFeatures(degree=4), RidgeCV(alphas=np.linspace(0.001, 100, 1000)))
#         model_rfy.fit(X_train, Y_train_rfy)
#         # ------ polinomial REGRESSOR


#         # ------ HUBER REGRESSOR
#         from sklearn.linear_model import HuberRegressor
#         from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
#         from sklearn.pipeline import make_pipeline
#         from sklearn.linear_model import RANSACRegressor
#         from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
#         from sklearn.pipeline import make_pipeline
        
#         # RANSACRegressor with a linear model
#         # model_rfx = make_pipeline(MinMaxScaler(), PolynomialFeatures(degree=3), RANSACRegressor(base_estimator=HuberRegressor()))
#         # model_rfy = make_pipeline(MinMaxScaler(), PolynomialFeatures(degree=3), RANSACRegressor(base_estimator=HuberRegressor()))
   
#         # Example pipeline with HuberRegressor
#         # model_rfx = make_pipeline(MinMaxScaler(), PolynomialFeatures(degree=1), HuberRegressor())
#         # model_rfy = make_pipeline(MinMaxScaler(), PolynomialFeatures(degree=1), HuberRegressor())

#         # model_rfx.fit(X_train, Y_train_rfx)
#         # model_rfy.fit(X_train, Y_train_rfy)
#         # ------ HUBER REGRESSOR

#         # Predict for unknown values
#         if unknown_indices.any():
#             X_unknown = channel_map_exp1_dict['electrodeXYPositions'][unknown_indices]
#             channel_map_exp1_dict['RFX_predicted'][unknown_indices] = model_rfx.predict(X_unknown)
#             channel_map_exp1_dict['RFY_predicted'][unknown_indices] = model_rfy.predict(X_unknown)

#%%

# Assuming 'channel_map_exp1_dict' has a key 'ArrayNumber' with array numbers for each electrode

# Initialize the predicted fields to NaN
channel_map_exp1_dict['RFX_predicted'] = np.full_like(channel_map_exp1_dict['RFX'], np.nan)
channel_map_exp1_dict['RFY_predicted'] = np.full_like(channel_map_exp1_dict['RFY'], np.nan)

# Group electrodes by every two arrays: (1,2), (3,4), ..., (15,16)
# for start_array in range(1, 17, 2):  # 1, 3, ..., 15
# # for start_array in range(1, 3, 2):  # 1, 3, ..., 15
#     end_array = start_array + 1  # 2, 4, ..., 16
#     array_indices = (np.array(channel_map_exp1_dict['arrayNumbers']) == start_array) | \
#                     (np.array(channel_map_exp1_dict['arrayNumbers']) == end_array)
#     known_indices = np.isfinite(channel_map_exp1_dict['RFX']) & array_indices
#     unknown_indices = ~np.isfinite(channel_map_exp1_dict['RFX']) & array_indices
                  
                    
                    
for area in areas:
    area_indices = np.array(channel_map_exp1_dict['Area']) == area
    known_indices = np.isfinite(channel_map_exp1_dict['RFX']) & area_indices
    unknown_indices = ~np.isfinite(channel_map_exp1_dict['RFX']) & area_indices
     
    array_indices =              area_indices
                    

    known_indices = np.isfinite(channel_map_exp1_dict['RFX']) & array_indices
    unknown_indices = ~np.isfinite(channel_map_exp1_dict['RFX']) & array_indices

    # Proceed only if there are known RFX, RFY values for training

    X_train = channel_map_exp1_dict['electrodeXYPositions'][known_indices]
    Y_train_rfx = channel_map_exp1_dict['RFX'][known_indices]
    Y_train_rfy = channel_map_exp1_dict['RFY'][known_indices]

    # Define and fit the models (polynomial)
    # model_rfx = make_pipeline(MinMaxScaler(), PolynomialFeatures(degree=2), RidgeCV(alphas=np.linspace(0.01, 10, 1000)))
    # model_rfy = make_pipeline(MinMaxScaler(), PolynomialFeatures(degree=2), RidgeCV(alphas=np.linspace(0.01, 10, 1000)))
    


    model_rfx = make_pipeline(MinMaxScaler(), PolynomialFeatures(degree=1), RidgeCV(alphas=np.linspace(0.001, 100, 1000)))
    model_rfx.fit(X_train, Y_train_rfx)
    
    model_rfy = make_pipeline(MinMaxScaler(), PolynomialFeatures(degree=1), RidgeCV(alphas=np.linspace(0.001, 100, 1000)))
    model_rfy.fit(X_train, Y_train_rfy)
    
    
    #  Define and fit the models (huber regressor)
    # model_rfx = make_pipeline(MinMaxScaler(), PolynomialFeatures(degree=3), HuberRegressor())
    # model_rfy = make_pipeline(MinMaxScaler(), PolynomialFeatures(degree=3), HuberRegressor())

    model_rfx.fit(X_train, Y_train_rfx)
    model_rfy.fit(X_train, Y_train_rfy)
  
    X_unknown = channel_map_exp1_dict['electrodeXYPositions'][unknown_indices]
    channel_map_exp1_dict['RFX_predicted'][unknown_indices] = model_rfx.predict(X_unknown)
    channel_map_exp1_dict['RFY_predicted'][unknown_indices] = model_rfy.predict(X_unknown)
    
    
    # Predict RFX and RFY for all electrodes in the current pair of arrays
    X = channel_map_exp1_dict['electrodeXYPositions'][array_indices]

    # to use predictions only
    # channel_map_exp1_dict['RFX_predicted'][array_indices] = model_rfx.predict(X)
    # channel_map_exp1_dict['RFY_predicted'][array_indices] = model_rfy.predict(X)
  
    # channel_map_exp1_dict['RFX_predicted'][known_indices] = channel_map_exp1_dict['RFX'][known_indices]
    # channel_map_exp1_dict['RFY_predicted'][known_indices] = channel_map_exp1_dict['RFY'][known_indices]
    
   
# Define colors for each area
area_colors = {'V2': 'green', 'V1': 'blue', 'V4': 'red', 'IT': 'purple'}

# Iterate through the unique areas to create a scatter plot for each
for area in set(channel_map_exp1_dict['Area']):
# for area in ['V2']:
    plt.figure(figsize=(8, 5))  # Create a new figure for each area
    
    # Get the indices of electrodes that belong to the current area
    indices = [i for i, a in enumerate(channel_map_exp1_dict['Area']) if a == area]
    
    # Extract the RF locations for these indices
    rfx = channel_map_exp1_dict['RFX_predicted'][indices]
    rfy = channel_map_exp1_dict['RFY_predicted'][indices]
    
    # Plot the RF locations with the color specified for the area
    plt.scatter(rfx, rfy, color=area_colors[area], label=f"Area {area}", alpha=0.6, edgecolors='none')
    plt.axvline(0, color='gray', linestyle='dashed', linewidth=0.5)
    plt.axhline(0, color='gray', linestyle='dashed', linewidth=0.5)

    # Enhance the plot for the current area
    plt.xlabel('RF X Location')
    # plt.xlim(-1,9)
    # plt.ylim(-8,5)
    
    
    plt.xlim(-9,9)
    plt.ylim(-8,8)

    plt.ylabel('RF Y Location')
    plt.title(f'Receptive Field Locations in Area {area}')
    plt.legend()
    plt.grid(False)
    # plt.axis('equal')  # Set equal scaling by changing axis limits
    
    # Show the plot for the current area
    plt.show()
    
#%%  
# =============================================================================
#     # 
# =============================================================================
    
    
#%% Saving the mapping file

results_folder_path = os.path.join(rootpath, 'results')
mapping_file_path = os.path.join(results_folder_path, 'mapping_MrNilson.pkl')
with open(mapping_file_path, 'wb') as file:
    pickle.dump(channel_map_exp1_dict, file)
print(f"'channel_map_exp1_dict' saved successfully to {mapping_file_path}")

#%% Loading it and show content


# Define the path to the 'mapping.kl' file within the 'results' folder
rootpath = r"\\vs03\VS03-VandC-INTENSE2\mind_writing\data_analysis\MAPPING"  # Ensure this is your correct rootpath
mapping_file_path = os.path.join(rootpath, 'results', 'mapping_MrNilson.pkl')

# Load the 'channel_map_exp1_dict' dictionary from the 'mapping.kl' file
with open(mapping_file_path, 'rb') as file:
    channel_map_exp1_dict = pickle.load(file)

print(f"'channel_map_exp1_dict' loaded successfully from {mapping_file_path}")

# Print all the keys of the loaded dictionary
print("Keys in 'channel_map_exp1_dict':")
for key in channel_map_exp1_dict.keys():
    print(key)
    
# Iterate over the dictionary and display the size of the content for each key
print("Content size for each key in 'channel_map_exp1_dict':")
for key, value in channel_map_exp1_dict.items():
    if hasattr(value, 'shape'):
        # For NumPy arrays, display the shape
        print(f"{key}: shape {value.shape}")
    elif hasattr(value, '__len__'):
        # For objects that support len(), display the length
        print(f"{key}: length {len(value)}")
    else:
        # Fallback for other types
        print(f"{key}: type {type(value)}")
        
        
#%% Get RFX and RFY values for a specific electrode (for Daniela)

electrode_number = 10  # Your specific electrode number
array_number = 1  # Your specific array number

electrode_nums = channel_map_exp1_dict['electrodeNumbers']
array_nums = channel_map_exp1_dict['arrayNumbers']

# Find the index for the specific electrode and array number
index = None
for i, (e_num, a_num) in enumerate(zip(electrode_nums, array_nums)):
    if e_num == electrode_number and a_num == array_number:
        index = i
        break

rfx = channel_map_exp1_dict['RFX'][index]
rfy = channel_map_exp1_dict['RFY'][index]

print(f"RFX: {rfx}, RFY: {rfy}")


#%% Plotting jumps in physical space, with associated channel values 

# dict_keys(['jumpIndex', 'jumpNumber', 'instanceChannels_128', 'instanceChannels_1024', 
#            'instanceNumbers', 'electrodeNumbers', 'arrayNumbers', 'stimChannels',
#            'stimNumbers', 'cereM_banks', 'stimBanks', 'arrayColor', 'alphaValue'])
# %matplotlib qt
%matplotlib inline

PLOT = False
SAVE_FIGURE = True


if PLOT:

    fontsize = 12  # Adjust font size for readability within boxes
    array_number_key = 'arrayNumbers'  # Key for accessing the array numbers
    
    # Parameters for plot layout adjustments
    jumps_per_column = 16
    columns = 2
    box_width = 25
    box_height = 15
    space_between_columns = 2
    space_between_rows = 3
    row_within_box_sep = 3
    
    for key, value in channel_map_exp1_dict.items():
    
        if key != 'electrodeXYPositions' and key != 'arrayColor' and key != 'alphaValue' and key != 'jumpIndex' and key != 'jumpNumber':
        # if key == 'instanceNumbers' or key == 'instanceChannels_128' or key == 'cereM_banks':
            print(f"Key: {key}")
        
            plot_key = key # Change this to 'jumpIndex', 'instanceChannels_128', or any other key
            if plot_key != 'arrayColor' and plot_key != 'alphaValue':
          
                plt.figure(dpi=200, figsize=(14, 12))
                
                # Unpack necessary arrays from the dictionary for easy access
                selected_values = channel_map_exp1_dict[plot_key]  # Example key
                array_numbers = channel_map_exp1_dict[array_number_key]  # Accessing array numbers
                array_colors = channel_map_exp1_dict['arrayColor']
                alpha_values = channel_map_exp1_dict['alphaValue']
                
                # Iterate over all jump boxes
                for box_num in range(1, 33):
                    col = 0 if box_num <= 16 else 1
                    row = 16 - box_num if col == 0 else box_num - 17
                
                    x_base = col * (box_width + space_between_columns)
                    y_base = row * (box_height + space_between_rows)
                
                    first_jump_index_in_box = (box_num - 1) * 32
                    color = array_colors[first_jump_index_in_box]
                    alpha = alpha_values[first_jump_index_in_box]
                
                    plt.fill_between([x_base, x_base + box_width], y_base, y_base + box_height, color=color, alpha=alpha, edgecolor='black', linewidth=1)
                
                    # Annotation for array number on the left or right of the box
                    array_number_text = 'A' + str(array_numbers[first_jump_index_in_box])  # Access first element's array number
                    text_x_offset = -1 if col == 0 else box_width + 1  # Position text left or right based on column
                    plt.text(x_base + text_x_offset, y_base + box_height / 2, array_number_text, color='black', ha='center', va='center', fontsize=fontsize, fontweight='bold')
                
                    # Plot selected values within each box
                    x_padding = 0.5
                    x_spacing = (box_width - 2 * x_padding) / 15
                    
                    for i in range(32):
                        x_text = x_base + x_padding + (i % 16) * x_spacing
                        y_text = y_base + (box_height / 2) + (row_within_box_sep if i < 16 else -row_within_box_sep)
                        
                        value_to_plot = selected_values[first_jump_index_in_box + i]
                        plt.text(x_text, y_text, str(value_to_plot), color='black', ha='center', va='center', fontsize=fontsize, fontweight='bold')
        
                plt.title(f'{plot_key.capitalize()}', fontsize=30)
                plt.axis('off')
                plt.tight_layout()
                
                if SAVE_FIGURE:
                    # Saving the figure with the name corresponding to plot_key, in PNG format with DPI 300
                    figure_filename = rootpath + "/results/" + f"{plot_key.capitalize()}.png"
                    plt.savefig(figure_filename, dpi=300, format='png', bbox_inches='tight')
                
                plt.show()