# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 12:53:40 2024

@author: Lozano
"""




#%% Loading Bars and static Antonio

# RF_TYPE = 'THINGS' # THINGS, BARS_S_NOISE, MATT
RF_TYPE = 'BARS_S_NOISE' # THINGS, BARS_S_NOISE
# RF_TYPE = 'MATT' # THINGS, BARS_S_NOISE

if RF_TYPE == 'BARS_S_NOISE':
    # ---- COMBINED DATA BARS AND SPARSE NOISE ----
    # Load combined_data containing the RF data
    rf_data_filename = r'C:\Users\Lozano\Desktop\NIN\Mr Nilson\RF_mapping_SparseNoise\code\analyze_sparseNoise_AL\combined_data_NO_PREDICTIONS.pkl'
    with open(rf_data_filename, 'rb') as f:
        combined_data_AL = pickle.load(f)
        print(combined_data.keys())
    print("RF combined_data loaded successfully.")

#%% Loading Things Paolo

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
    combined_data_THINGS = pickle.load(f)
    print(combined_data_THINGS.keys())

# Now let's harmonize keys and update combined_data with the data from the THINGS dict
# Mapping from the keys in 'data' (THINGS data) to the keys in 'combined_data'
key_mapping = {
    'all_centrex': 'RFX',
    'all_centrey': 'RFY',
    'all_szx': 'STDX',
    'all_szy': 'STDY',
    'all_test_corr': 'R2'
}

new_combined_data = combined_data_THINGS.copy()

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
combined_data_THINGS = new_combined_data
    
#%% New mapping from Paolo & Matt
# =============================================================================
#   LOADING BARS for V1
# =============================================================================
# Define the path to your .mat file
mat_file_path = r'\\vs03\VS03-VandC-INTENSE2\mind_writing\data_analysis\MAPPING\Matt\BarMap_Nilson.mat'

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
combined_data_MATT = {
    'RFX': rf_data['centrex'],
    'RFY': rf_data['centrey'],
    'STDX': rf_data['sz'],  # Assuming 'sz' maps directly to standard deviation in your existing data
    'STDY': rf_data['sz'],  # Same as above, adjust if your data differentiates between X and Y
    'R2': r2_placeholder  # Placeholder for 'R2' with NaN values
}

# Convert sizes from pixels to degrees if 'pixperdeg' is available
if 'pixperdeg' in data:
    pix_per_deg = data['pixperdeg'][0, 0]  # Assuming it's a scalar
    combined_data_MATT['STDX'] /= pix_per_deg
    combined_data_MATT['STDY'] /= pix_per_deg
    
# =============================================================================
#   LOADING SPARSE NOISE for V4
# =============================================================================
# Define the path to your .mat file
# Load the existing combined data, assuming it's stored in a variable named `combined_data`
# Define the path to the new .mat file
grid_map_path = r'\\vs03\VS03-VandC-INTENSE2\mind_writing\data_analysis\MAPPING\Matt\GridMap_Nilson.mat'

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
        combined_data_MATT[key] = np.concatenate((combined_data_MATT[key], grf_data['ctx']))
    elif key == 'RFY':
        combined_data_MATT[key] = np.concatenate((combined_data_MATT[key], grf_data['cty']))
    elif key == 'STDX':
        combined_data_MATT[key] = np.concatenate((combined_data_MATT[key], grf_data['xsz']))
    elif key == 'STDY':
        combined_data_MATT[key] = np.concatenate((combined_data_MATT[key], grf_data['ysz']))
    elif key == 'R2':  # If you need to create a placeholder for R2
        new_r2 = np.full_like(grf_data['ctx'], np.nan)  # Create a NaN-filled array for new R2 data
        combined_data_MATT[key] = np.concatenate((combined_data_MATT[key], new_r2))

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
        combined_data_MATT[key] = np.concatenate((combined_data_MATT[key], nan_padding))

# To check the final lengths and last few elements:
print("Updated combined_data with padded NaN values:")
for key in ['RFX', 'RFY', 'STDX', 'STDY', 'R2']:
    print(f"{key}: Length = {len(combined_data_MATT[key])}, Last 5 entries = {combined_data[key][-5:]}")



#%% Ordering data

















