import time
start_time = time.time()
import numpy as np
import pandas as pd 
import sys
import os
from os.path import exists
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from tqdm import tqdm
from numpy import hstack
import pickle
import math
import xlsxwriter

#############################################################################################################################
class DataPreprocessor:
    def __init__(self):
        self.tracks_data_down_sampled = pd.DataFrame()
    
    def downsample(self, tracks_data, skip_width):
        tracks_data_downsampled = tracks_data.iloc[skip_width::skip_width+1]
        tracks_data_downsampled.reset_index(drop=True, inplace=True)
        return tracks_data_downsampled 

#################################################################################################  
def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    
#################################################################################################
def average_displacement_error(file_path):
    # Read the data from the file into a DataFrame
    df = pd.read_csv(file_path)

    # Calculate Displacement Errors and store them
    displacement_errors = []

    for index, row in df.iterrows():
        x_pred = row['xCenter_pred']
        y_pred = row['yCenter_pred']
        x_gt = row['xCenter_gt']
        y_gt = row['yCenter_gt']

        # Calculate Euclidean Displacement Error for this row
        displacement_error = np.sqrt((x_pred - x_gt)**2 + (y_pred - y_gt)**2)

        # Append the displacement error to the list
        displacement_errors.append(displacement_error)
    return displacement_errors

#################################################################################################
def final_displacement_error(file_path):
    # Read the data from the file into a DataFrame
    df = pd.read_csv(file_path)

    # Get the last row of the DataFrame
    last_row = df.iloc[-1]

    # Calculate Displacement for the last row
    x_displacement = last_row['xCenter_pred'] - last_row['xCenter_gt']
    y_displacement = last_row['yCenter_pred'] - last_row['yCenter_gt']

    # Calculate Final Displacement Error for the last row
    final_displacement_error = np.sqrt(x_displacement**2 + y_displacement**2)

    return final_displacement_error
##################################################################################################
recording_id = "25"
data_track = pd.read_csv('data_processing/dataset/data/' + recording_id + '_tracks.csv')
data_meta = pd.read_csv('data_processing/dataset/data/' + recording_id + '_tracksMeta.csv')
data_track_path = f'data_processing/dataset/data/25_tracks.csv'
data_meta_path = f'data_processing/dataset/data/25_tracksMeta.csv'

new_data_track = data_meta[data_meta['class'] == 'car']
new_data_track_id = new_data_track['trackId']

data_track_filtered = data_track[data_track['trackId'].isin(new_data_track_id)]

track_data_raw = data_track_filtered
track_meta_data_raw = data_meta # i didnt do any changes to meta data hence giving it directly

def count_rows_with_start_trackId(data, start_row, specific_column, specific_value):
    # Check if the specific column exists in the DataFrame
    if specific_column not in data.columns:
        print(f"Column '{specific_column}' not found in the CSV file.")
        return

    # Initialize a counter to count rows with the specific value
    count = 0

    # Iterate through rows starting from the specified start_row
    for index, row in data.iterrows():
        if index >= start_row:
            if row[specific_column] == specific_value:
                count += 1
            else:
                # Break the loop if the specific value is not found to optimize the search
                break

    return count

def mse_using_pandas(df_y, df_y_hat):
    mse = ((df_y - df_y_hat) ** 2).mean()
    return mse

def rmse_using_pandas(df_y, df_y_hat):
    mse = ((df_y - df_y_hat) ** 2).mean()
    rmse = np.sqrt(mse)
    return rmse

def bicycle_model(current_x, current_y, current_vx, current_vy, current_ax, current_ay,
                  current_heading, heading_change, wheelbase, dt, estimated_steering_angle):


    # Calculate the change in velocity along the x and y directions

    delta_vx = (current_ax * np.cos(current_heading) - current_ay * np.sin(current_heading)) * dt
    delta_vy = (current_ax * np.sin(current_heading) + current_ay * np.cos(current_heading)) * dt


    new_vx = current_vx + delta_vx
    new_vy = current_vy + delta_vy


    # Calculate the change in x and y positions

    delta_x = (new_vx * np.cos(current_heading) - new_vy * np.sin(current_heading)) * dt                
    delta_y = (new_vx * np.sin(current_heading) + new_vy * np.cos(current_heading)) * dt 


    #delta_heading is yaw rate              

    if isinstance(wheelbase, (int, float)) and wheelbase != 0:  
        delta_heading = (new_vx / wheelbase) * np.tan(estimated_steering_angle) * dt
    else:
        delta_heading = heading_change

    estimated_steering_angle =  np.arctan((wheelbase* delta_heading) / (new_vx * dt))

    # Predict the new x and y positions
    new_x = current_x + delta_x
    new_y = current_y + delta_y
    new_heading = current_heading + delta_heading

    return new_x, new_y, new_heading
################################################ Preprocessing the data ###############################################
pre_process_obj = DataPreprocessor()
skip_width = 0
start_row = 0
num_frames = 25
# Time step in seconds (assuming 1/25 seconds based on the information you provided)
dt = (skip_width + 1)/25

tracks_data_down_sampled = pre_process_obj.downsample(data_track_filtered, skip_width)
tracks_data_down_sampled.to_csv('downsampled_data.csv', index=False)
columns_to_keep = ["trackId", "xCenter", "yCenter", "heading", "length", "xVelocity", "yVelocity", "xAcceleration", "yAcceleration"]
data = tracks_data_down_sampled.loc[:, columns_to_keep]

data['wheelbase'] = data['length'] * 0.90
headings = data['heading']
delta_heading = headings.diff()
# Fill the first NaN value with 0, as there is no previous value to calculate the difference
delta_heading = delta_heading.fillna(0)

# Add the delta_heading as a new column to the DataFrame
data['delta_heading'] = delta_heading
data['velocity'] = np.sqrt(data['xVelocity']**2 + data['yVelocity']**2)
data['estimated_steering_angles'] = np.arctan((data['wheelbase'] * data['delta_heading']) / (data['velocity'] * dt))
data.to_csv('filtered_data.csv', index=False)

##################################################

# Get the row for the specified start_row number
data_row = data.iloc[start_row]

# Extract the necessary data from the data_row
xCenter = data_row['xCenter']
yCenter = data_row['yCenter']
heading = data_row['heading']
length = data_row['length']
xVelocity = data_row['xVelocity']
yVelocity = data_row['yVelocity']
xAcceleration = data_row['xAcceleration']
yAcceleration = data_row['yAcceleration']
wheelbase = data_row['wheelbase']
heading = data_row['heading']
delta_heading = data_row['delta_heading']
trackId_check = data_row['trackId']
estimated_steering_angle = data_row['estimated_steering_angles']
specific_column = 'trackId'
rows_available = count_rows_with_start_trackId(data, start_row, specific_column, trackId_check)
actual_rows_to_predict = min(num_frames, rows_available)

current_x = xCenter
current_y = yCenter
current_vx = xVelocity  # m/s
current_vy = yVelocity   # m/s (initially no lateral velocity)
current_ax = xAcceleration   # m/s² (initially no longitudinal acceleration)
current_ay =  yAcceleration  # m/s² (initially no lateral acceleration)
current_heading = np.deg2rad(heading)  # Initial heading angle in radians
heading_change = np.deg2rad(delta_heading)    # Change in heading angle in radians   # Example wheelbase in meters
time_step = dt  # Example time step in seconds
num_iterations = actual_rows_to_predict   # Number of iterations

# Lists to store the predictions at each iteration
predicted_x_values = []
predicted_y_values = []
predicted_heading_values = []

# Perform multiple iterations using the bicycle model
for _ in range(num_iterations):
    new_x, new_y, new_heading = bicycle_model(current_x, current_y, current_vx, current_vy,
                                               current_ax, current_ay, current_heading,
                                               heading_change, wheelbase, time_step, estimated_steering_angle)
    # Store the predicted values
    predicted_x_values.append(new_x)
    predicted_y_values.append(new_y)
    predicted_heading_values.append(new_heading)

    # Update the current state for the next iteration
    current_x = new_x
    current_y = new_y
    current_heading = new_heading

# Extract subsets for xCenter and yCenter
subset_xCenter = data.loc[start_row:start_row + actual_rows_to_predict - 1, 'xCenter'].values.flatten()
subset_yCenter = data.loc[start_row:start_row + actual_rows_to_predict - 1, 'yCenter'].values.flatten()

# Create a DataFrame with the specified column names
total_data = pd.DataFrame({
    'xCenter_gt': subset_xCenter,
    'xCenter_pred': predicted_x_values,
    'yCenter_gt': subset_yCenter,
    'yCenter_pred': predicted_y_values
})

# Define the file path for the CSV file
output_csv_path = 'bicycle_predictions.csv'
# Save the DataFrame to a CSV file
total_data.to_csv(output_csv_path, index=False)
################################ visual representations ###################
# Save the updated DataFrame to a new CSV file
output_csv_path = 'bicycle_errors.csv'
total_data.to_csv(output_csv_path, index=False)
ade_values = average_displacement_error(output_csv_path)
ade_value = np.mean(ade_values)

# Initialize a list to store cumulative averages
cumulative_averages = []

# Initialize variables for cumulative calculations
cumulative_sum = 0
cumulative_count = 0

# Calculate cumulative averages
for value in ade_values:
    cumulative_sum += value
    cumulative_count += 1
    cumulative_average = cumulative_sum / cumulative_count
    cumulative_averages.append(cumulative_average)

fde_value = final_displacement_error(output_csv_path)

# Step 4: Print the evaluation results
print("Average Displacement Error (ADE):", ade_value)
print("Final Displacement Error (FDE):", fde_value)
print(f'Data with distances has been saved to {output_csv_path}.')
timeVec = np.arange(0, dt * num_frames, dt)
plt.figure(figsize=(10, 6))
plt.plot(timeVec, cumulative_averages, marker='o', linestyle='-', label='Average Distance Error')
plt.plot(timeVec, ade_values, marker='x', linestyle='-', label='Deviated Distance')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Error in location prediction [m]')
plt.title('Bicycle model prediciton error ')
plt.tight_layout()
plt.savefig('Bicycle.png', dpi=300, bbox_inches='tight', format='png')
plt.show()
elapsed_time = time.time() - start_time
print(f"Elapsed Time is {elapsed_time}")