import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read data from Excel file
recording_id = "25"
# Create a Pandas Dataframe
data_track = pd.read_csv('data_processing/dataset/data/' + recording_id + '_tracks.csv')
data_meta = pd.read_csv('data_processing/dataset/data/' + recording_id + '_tracksMeta.csv')

new_data_track = data_meta[data_meta['class'] == 'car']
new_data_track_id = new_data_track['trackId']

data_track_filtered = data_track[data_track['trackId'].isin(new_data_track_id)]

data = data_track_filtered

#Define Kalman filter parameters
# Replace these with your actual initial values
x_initial = data['xCenter'].iloc[0]  # Initial x position
y_initial = data['yCenter'].iloc[0]  # Initial y position
x_velocity_initial = data['xVelocity'].iloc[0]  # Initial x velocity
y_velocity_initial = data['yVelocity'].iloc[0]  # Initial y velocity

dt = 0.04  # Time step (delta t)

# Define Kalman filter parameters
initial_state = np.array([x_initial, y_initial, x_velocity_initial, y_velocity_initial])
initial_covariance = np.eye(4)  # Initial state covariance matrix

# We have measurements for both position and velocity, so
process_noise = np.array([[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 0.1, 0],
                          [0, 0, 0, 0.1]])

measurement_noise = np.eye(4) * 0.01  

# Define state transition matrix A 
# Following one is Constant Velocity model
A = np.array([[1, 0, dt, 0],
              [0, 1, 0, dt],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])

# Define measurement matrix H
H = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])



# Initialize Kalman filter
state = initial_state
covariance = initial_covariance

# Initialize lists to store predicted positions
predicted_x = []
predicted_y = []
measurement_x = []  
measurement_y = []
deviated_distances = []

frames_predicted = 0
num_time_steps_to_predict = 50

# For 4x4 H matrix
for index, row in data.iterrows():
    # Extract measurements from the data
    measurement = np.array([row['xCenter'], row['yCenter'], row['xVelocity'], row['yVelocity']])  # Update measurement vector
    measurement_x.append(measurement[0])
    measurement_y.append(measurement[1])

    # Predict
    state_estimate = np.dot(A, state)
    covariance_estimate = np.dot(np.dot(A, covariance), A.T) + process_noise

    # Update
    kalman_gain = np.dot(np.dot(covariance_estimate, H.T), np.linalg.inv(np.dot(np.dot(H, covariance_estimate), H.T) + measurement_noise))
    state = state_estimate + np.dot(kalman_gain, (measurement - np.dot(H, state_estimate)))
    covariance = np.dot((np.eye(4) - np.dot(kalman_gain, H)), covariance_estimate)
    
    deviated_distance = np.sqrt((measurement[0] - state[0])**2 + (measurement[1] - state[1])**2)
    deviated_distances.append(deviated_distance)
    
    frames_predicted += 1

    predicted_x.append(state[0])
    predicted_y.append(state[1])

    if frames_predicted >= num_time_steps_to_predict:
        break  # Exit the loop if we've predicted for the desired number of frames
    
    
predicted_x = np.array(predicted_x)
predicted_y = np.array(predicted_y)
measurement_x = np.array(measurement_x)
measurement_y = np.array(measurement_y)

mse_x = np.mean((measurement_x - predicted_x)**2)
mse_y = np.mean((measurement_y - predicted_y)**2)


# Calculate Mean Squared Error (MSE) for x and y positions
mse_total = np.sqrt(mse_x**2 + mse_y**2)

print(f"Total Mean Squared Error (MSE): {mse_total} meters")

# Now, predicted_x and predicted_y contain the predicted positions of the vehicle.
# Calculate the deviated distance at the last point of prediction
last_deviated_distance = deviated_distances[-1]
print(f"Deviated Distance at Last Point of Prediction: {last_deviated_distance} meters")

# Calculate the displacement errors at each point
displacement_errors = np.sqrt((predicted_x - measurement_x)**2 + (predicted_y - measurement_y)**2)

# Calculate the average displacement error
average_displacement_error = np.mean(displacement_errors)

# Print the average displacement error
print(f"Average Displacement Error: {average_displacement_error} meters")


time_range = np.linspace(1/25, num_time_steps_to_predict/25, num=num_time_steps_to_predict)
plt.figure(figsize=(10, 5))
plt.plot(time_range, deviated_distances, label='Deviated Distance')
plt.xlabel('Time (s)')
plt.ylabel('Deviated Distance')
plt.title('Deviated Distance vs. Time')
plt.legend()
plt.grid(True)
plt.savefig('Kalman Distance Deviation.png', dpi=300)
plt.show()
#######################################################################

xCenter = data['xCenter'].tolist()  # Extract "xCenter" column to a list
xCenter = xCenter[:num_time_steps_to_predict]
yCenter = data['yCenter'].tolist()  # Extract "yCenter" column to a list
yCenter = yCenter[:num_time_steps_to_predict]


# Create a new DataFrame
result_df = pd.DataFrame({'xCenter': xCenter, 'measurement_x': measurement_x, 'predicted_x': predicted_x, 'yCenter': yCenter, 'measurement_y': measurement_y, 'predicted_y': predicted_y})

# Define the Excel file name to save the data
output_excel_file = 'Kalman_output_data.xlsx'

# Save the DataFrame to an Excel file
result_df.to_excel(output_excel_file, index=False)

print(f"Data saved to {output_excel_file}")