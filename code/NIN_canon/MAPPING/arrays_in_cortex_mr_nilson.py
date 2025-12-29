# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 16:14:32 2024

@author: Lozano


 >>> IN CONSTRUCTION
 
 
 The script is meant to go from the surgical picture to XY cortical locations of each
 array (and electrode).  
 



Input: 
    surgical picture (colored)
    a map that helps ordering the arrays to match the surgical picture
       
       
       
       
Output:
    dictionary with electrode positions, array IDs, area IDs and colors
    
    
    
 
 TODO: 
     
     - make sure the order of the arrays is correct
     - make sure the orientation of the electrode numbers is correct
     - would be nice to modify the color code to match BR's mapping excel files
     - map each electrode to a jumper number, array number, channel number, instance number, 
       cereplex M bank (A,B,C,D), stimulator number and stimulator bank (A, B), given the mapping
       used in Experiment 1. (This will be done in "mapping.py")
       
     



"""

#%%
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

rootpath = r"C:\Users\Radovan\OneDrive\Radboud\a_Internship\Antonio Lonzano\root\SlavsForSight\code\NIN_canon\MAPPING"
os.chdir(rootpath)


# Read the image
# image_path = r"C:\Users\Lozano\Desktop\NIN\Mr Nilson\nilson_excel_surgery_photo_arrays_masked.png"
image_path = rootpath + r"/data_surgery/nilson_excel_surgery_photo_arrays_masked.png"
image = cv2.imread(image_path)

# Define RGB values for red, green, and blue
red_rgb = np.array([255, 0, 0])       # Red color
green_rgb = np.array([0, 255, 0])     # Green color
blue_rgb = np.array([0, 0, 255])      # Blue color

# Define a function to create masks for a specified color
def create_color_mask(image, color_rgb, tolerance=20):
    lower_bound = color_rgb - tolerance
    upper_bound = color_rgb + tolerance
    lower_bound[lower_bound < 0] = 0
    upper_bound[upper_bound > 255] = 255
    mask = cv2.inRange(image, lower_bound, upper_bound)
    return mask


def interpolate_points(p1, p2, num_points=10):
    """Interpolate `num_points` between `p1` and `p2` (inclusive)."""
    return np.linspace(p1, p2, num=num_points)


# def interpolate_grid_points(p1, p2, p3, p4, grid_size=8):
def interpolate_grid_points(p1, p3, p4, p2, grid_size=8):
    """Interpolate grid points within a square defined by four corners.
    
    :param p1, p2, p3, p4: Corners of the square, sorted clockwise or counter-clockwise.
    :param grid_size: The number of points per side in the grid.
    :return: A 2D array of grid points.
    """
    # Interpolate points along the top and bottom edges
    top_edge = interpolate_points(p1, p2, num_points=grid_size)
    bottom_edge = interpolate_points(p3, p4, num_points=grid_size)
    
    # Prepare the grid
    grid = np.zeros((grid_size, grid_size, 2))  # Initialize a grid of zeros
    
    # Fill the grid by interpolating between the top and bottom edges
    for i in range(grid_size):
        # grid[:, i, :] = interpolate_points(F_edge[i], bottom_edge[i], num_points=grid_size)
        grid[:, i, :] = interpolate_points(top_edge[i], bottom_edge[i], num_points=grid_size)
    
    return grid

def generate_utah_array_with_labels(corners, grid_size=8):
    """Generate an 8x8 Utah array grid with labels for given square corners."""
    # Assuming corners are [top-left, top-right, bottom-right, bottom-left]
    utah_array_grid = interpolate_grid_points(corners[0], corners[1], corners[2], corners[3], grid_size=grid_size)
    
    # Assign labels from 1 to 64 (or grid_size^2)
    labels = np.arange(1, grid_size**2 + 1).reshape((grid_size, grid_size))
    
    return utah_array_grid, labels

color_to_area = {
    'Blue': 'IT',  # Assuming 'Red' corresponds to the 'IT' area
    'Green': 'V4',  # Assuming 'Green' corresponds to the 'V4' area
    'Red': 'V1'  # Assuming 'Blue' corresponds to the 'V1' area
}


#%% mapping NEW to SURGERY labels

new_to_surgery_labels_mapping = {
    16: 13,
    15: 16,
    14: 15,
    13: 14,
    12: 12,
    11: 11,
    10: 10,
    9: 9,
    8: 8,
    7: 6,
    6: 1,
    5: 4,
    4: 7,
    3: 5,
    2: 2,
    1: 3,
}

#%%
# Create masks for each color
red_mask = create_color_mask(image, red_rgb)
green_mask = create_color_mask(image, green_rgb)
blue_mask = create_color_mask(image, blue_rgb)

# Apply erosion to each mask
kernel = np.ones((3, 3), np.uint8)
red_mask = cv2.erode(red_mask, kernel, iterations=2)
green_mask = cv2.erode(green_mask, kernel, iterations=2)
blue_mask = cv2.erode(blue_mask, kernel, iterations=2)

ArrCounter = 0

# %matplotlib inline

# Iterate through each mask
for color, mask in zip(['Red', 'Green', 'Blue'], [red_mask, green_mask, blue_mask]):
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
    
    # Initialize list to store square corners
    square_corners = []
    
    # Iterate through contours and fit rectangles
    for contour in contours:
        # Approximate a polygonal curve from the contour
        epsilon = 0.03 * cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, epsilon, True)
        
        # If the contour has 4 vertices, it's likely a square
        if len(polygon) == 4:
            square_corners.append(polygon.reshape(-1, 2))
    
    # Draw white dots at the corners of each square and label them
    for i, corners in enumerate(square_corners):
        
        # Create the array number label
        array_label = f'{ArrCounter+1}'
        ArrCounter = ArrCounter + 1
        
        # Find the centroid of the square
        centroid = np.mean(corners, axis=0)
        
        # Sort corners based on their position relative to the centroid
        sorted_corners = sorted(corners, key=lambda x: np.arctan2(x[1]-centroid[1], x[0]-centroid[0]), reverse = True)
        
        # Draw white dots and label each corner
        for j, corner in enumerate(sorted_corners):
            label = str(j + 1)  # Start labeling from 1
            cv2.circle(image, tuple(corner), 3, (255, 255, 255), -1)
            cv2.putText(image, label, tuple(corner), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 2)
        cv2.putText(image, array_label, (int(centroid[0]), int(centroid[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    
# Display the image with white dots and labels
plt.figure(figsize=(8, 6))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title(f'{color} Squares with Corner Labels')
plt.show()



#%% Storing data in dicts

# Initialize the data structure
array_details = {}

# Reset the counter for consistency in example
ArrCounter = 0

# Iterate through each mask
for color, mask in zip(['Red', 'Green', 'Blue'], [red_mask, green_mask, blue_mask]):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        epsilon = 0.03 * cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(polygon) == 4:
            corners_array = polygon.reshape(-1, 2)
            centroid = np.mean(corners_array, axis=0)
            sorted_corners = sorted(corners_array, key=lambda x: np.arctan2(x[1]-centroid[1], x[0]-centroid[0]), reverse=True)
            
            # Increment the array counter and update the array details
            ArrCounter += 1
            new_label = ArrCounter
            
            # Map the new label to the surgery label using the provided mapping
            surgery_label = new_to_surgery_labels_mapping.get(new_label, "Unknown")
            
            # Store details in the dictionary
            array_details[new_label] = {
                'Visual Area': color_to_area[color],
                'Sorted Corners': sorted_corners,
                'Surgery Label': surgery_label,
                'New Label': new_label
            }

# At this point, `array_details` will contain all the required information for each array

#%% creating utah arrays

# Update array details with Utah array positions and labels
for new_label, details in array_details.items():
    sorted_corners = details['Sorted Corners']
    utah_array_grid, utah_array_labels = generate_utah_array_with_labels(sorted_corners)
    array_details[new_label]['Utah Array Positions'] = utah_array_grid
    array_details[new_label]['Utah Array Labels'] = utah_array_labels

# At this point, `array_details` includes the Utah array positions and labels (1 to 64) for each array

# %matplotlib inline

plt.figure(figsize=(12, 8))  # Increased figure size for better visibility

area_colors = {
    'V1': 'blue',
    'V4': 'green',
    'IT': 'red'
}

# To ensure each visual area is represented once in the legend
legend_handles = {}

for new_label, details in array_details.items():
    positions = details['Utah Array Positions']
    labels = details['Utah Array Labels']
    visual_area = details['Visual Area']
    
    # if visual_area == 'V1':
    surgery_label = details['Surgery Label']  # Assuming this contains the surgical label
    
    x, y = positions[:,:,0].flatten(), positions[:,:,1].flatten() * -1
    centroid_x, centroid_y = np.mean(x), np.mean(y)  # Calculate the centroid for the surgical label placement
    color = area_colors.get(visual_area, 'gray')
    
    # Plot points and check for legend handle
    if visual_area not in legend_handles:
        scatter = plt.scatter(x, y, color=color, label=visual_area)
        legend_handles[visual_area] = scatter
    else:
        plt.scatter(x, y, color=color)
    
    # Add electrode labels
    for i, label in enumerate(labels.flatten()):
        plt.text(x[i], y[i], str(label), color='black', fontsize=8)
    
    # Add the surgical label in a larger, bold font at the centroid
    plt.text(centroid_x, centroid_y, f'{surgery_label}', color='black', fontsize=20, fontweight='bold', ha='center', va='center')

# Adjust legend
plt.legend(handles=legend_handles.values(), title='Visual Area')

plt.title('Scatter Plot of Utah Array Positions with Corresponding Colors, Labels, and Bold Surgical Labels')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.grid(False)
plt.tight_layout()
plt.show()

#%% Plotting Utah Arrais with Array-wise color code


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


# Update array details with Utah array positions and labels
for new_label, details in array_details.items():
    sorted_corners = details['Sorted Corners']
    utah_array_grid, utah_array_labels = generate_utah_array_with_labels(sorted_corners)
    array_details[new_label]['Utah Array Positions'] = utah_array_grid
    array_details[new_label]['Utah Array Labels'] = utah_array_labels

# At this point, `array_details` includes the Utah array positions and labels (1 to 64) for each array

#%matplotlib inline
#%matplotlib qt

plt.figure(figsize=(12, 8))  # Increased figure size for better visibility

area_colors = {
    'V1': 'blue',
    'V4': 'green',
    'IT': 'red'
}

# To ensure each visual area is represented once in the legend
legend_handles = {}


for new_label, details in array_details.items():
    positions = details['Utah Array Positions']
    labels = details['Utah Array Labels']
    visual_area = details['Visual Area']
    
    # if visual_area == 'V1':
    surgery_label = details['Surgery Label']  # Assuming this contains the surgical label
    
    x, y = positions[:,:,0].flatten(), positions[:,:,1].flatten() * -1
    centroid_x, centroid_y = np.mean(x), np.mean(y)  # Calculate the centroid for the surgical label placement
    # color = area_colors.get(visual_area, 'gray')
    # color = allColors[ details['New Label'] * 64 - 64 +  labels[0][0] ]
    color = allColors[ details['Surgery Label'] * 64 - 64 +  labels[0][0] ]
    

    # Plot points and check for legend handle
    if visual_area not in legend_handles:
        scatter = plt.scatter(x, y, color=color, label=visual_area, s = 50)
        legend_handles[visual_area] = scatter
    else:
        plt.scatter(x, y, color=color, s = 50)
    
    # Add electrode labels
    for i, label in enumerate(labels.flatten()):
        plt.text(x[i], y[i], str(label), color='black', fontsize=8)
    
    # Add the surgical label in a larger, bold font at the centroid
    plt.text(centroid_x, centroid_y, f'{surgery_label}', color='black', fontsize=20, fontweight='bold', ha='center', va='center')

# Adjust legend
plt.legend(handles=legend_handles.values(), title='Visual Area')

plt.title('Scatter Plot of Utah Array Positions with Corresponding Colors, Labels, and Bold Surgical Labels')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.grid(False)
plt.tight_layout()
plt.show()


#%% SAVE DATA

import pickle

# File path where you want to save the data
# save_path = r"C:\Users\Lozano\Desktop\NIN\RF_mapping_SparseNoise\code\analyze_sparseNoise_AL\cortical_array_details.pkl"
save_path = r"\\vs03\VS03-VandC-INTENSE2\mind_writing\data_analysis\MAPPING\results\cortical_array_details.pkl"
# Saving array_details to a file
with open(save_path, 'wb') as file:
    pickle.dump(array_details, file)

print("Data saved successfully.")

#%% LOAD DATA
# save_path = r"C:\Users\Lozano\Desktop\NIN\RF_mapping_SparseNoise\code\analyze_sparseNoise_AL\cortical_array_details.pkl"

# Loading array_details from the saved file
with open(save_path, 'rb') as file:
    loaded_array_details = pickle.load(file)

print("Data loaded successfully.")

# Test to ensure the data is intact
print(loaded_array_details.keys())  # This will print the keys of the dictionary to verify it's loaded correctly












