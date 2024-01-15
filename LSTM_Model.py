import numpy as np
import pandas as pd 
import sys
import os
from os.path import exists
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from tqdm import tqdm
from numpy import hstack
import math
import xlsxwriter
from sklearn.model_selection import train_test_split
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler



recording_id = "25"
# Create a Pandas Dataframe
data_track = pd.read_csv('data_processing/dataset/data/' + recording_id + '_tracks.csv')
data_meta = pd.read_csv('data_processing/dataset/data/' + recording_id + '_tracksMeta.csv')

new_data_track = data_meta[data_meta['class'] == 'car']
new_data_track_id = new_data_track['trackId']

data_track_filtered = data_track[data_track['trackId'].isin(new_data_track_id)]

track_data_raw = data_track_filtered
track_meta_data_raw = data_meta # i didnt do any changes to meta data hence giving it directly

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pandas as pd

class DataPreprocessor:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.tracks_data_down_sampled = pd.DataFrame()
        self.min_max_scaler_list = [MinMaxScaler(feature_range=(0, 1))] * 13
        self.tracks_data_norm = pd.DataFrame()
    
    def downsample(self, tracks_data, skip_width):
        tracks_data_downsampled = tracks_data.iloc[skip_width::skip_width+1]
        tracks_data_downsampled.reset_index(drop=True, inplace=True)
        return tracks_data_downsampled
        

    def normalize(self, df, start_column):
        scaler = MinMaxScaler()
        df.iloc[:, start_column:] = scaler.fit_transform(df.iloc[:, start_column:])
        return df
##################################################################################
pre_process_obj = DataPreprocessor()

skip_width = 4
tracks_data_down_sampled = pre_process_obj.downsample(track_data_raw, skip_width)
tracks_data_norm = pre_process_obj.normalize(tracks_data_down_sampled, 4)

###########################################
# here i am using a new method to train the model 

# Step 1: Preprocess the data
# Assume you have loaded the dataset into a pandas DataFrame called 'data'
data = tracks_data_norm

# Step 2: Define input and output sequences
input_features = ['xCenter', 'yCenter', 'heading', 'xVelocity', 'yVelocity', 'lonVelocity', 'latVelocity', 'latAcceleration', ]
output_features = ['xCenter', 'yCenter', 'heading']
num_input_frames = 20
num_output_frames = 10

# Step 3: Split the data
input_data = data[input_features].values
output_data = data[output_features].values

# Step 4: Prepare the input and output sequences
X = []
y = []

for i in range(num_input_frames, len(input_data) - num_output_frames + 1):
    X.append(input_data[i - num_input_frames : i])
    y.append(output_data[i : i + num_output_frames])

X = np.array(X)
y = np.array(y)

# Reshape y to match the expected output shape
y = np.reshape(y, (y.shape[0], num_output_frames * len(output_features)))

# Step 5: Split the data into training and testing datasets
X_train, X_test_full, y_train, y_test_full = train_test_split(X, y, test_size=0.2, shuffle=False)
X_val, X_test, y_val, y_test = train_test_split(X_test_full, y_test_full, test_size=0.5, shuffle=False)

############  Learning rate
# Step 6: Design and train a model

model = Sequential()
model.add(LSTM(32, input_shape=(num_input_frames, len(input_features))))
model.add(Dense(128, activation = 'relu'))  # Adding a hidden Dense layer with ReLU activation
model.add(Dense(64, activation = 'relu'))  # Adding another hidden Dense layer with ReLU activation
#model.add(Dense(no_of_neurons_layer3, activation = activation)) 
model.add(Dense(num_output_frames * len(output_features)))

model.compile(loss='mse', optimizer=Adam(learning_rate= 0.001))

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Learning rate scheduling 
def lr_schedule(epoch, lr):
    if epoch < 20:
        return lr
    else:
        return lr * 0.1

lr_scheduler = LearningRateScheduler(lr_schedule)

# Add the callbacks to your model's fit function
history = model.fit(X_train, y_train, epochs=150, batch_size=32,
                    validation_data=(X_val, y_val), callbacks=[early_stopping, lr_scheduler])


# Step 7: Evaluate the model
X_test_eval = X_val[:, :num_input_frames, :]
y_test_eval = y_val[:, :num_output_frames * len(output_features)]

mse = model.evaluate(X_test_eval, y_test_eval)
print('Mean Squared Error:', mse)

# Step 8: Make predictions
X_test_pred = X_test[:, :num_input_frames, :]
predictions = model.predict(X_test_pred)

################### saving the predictions in excel file ###################
import pandas as pd

# Get the number of samples and total number of output features in the predictions
number_of_samples, total_output_features = predictions.shape

# Determine the number of output frames and output features per frame
num_output_frames = num_output_frames
num_output_features_per_frame = len(output_features)

# Create a list to store the data for each row in the Excel file
excel_data = []

# Loop through each sample
for i in range(number_of_samples):
    row_data = []

    # Loop through each output frame
    for j in range(num_output_frames):
        # Loop through each output feature per frame
        for k in range(num_output_features_per_frame):
            # Append the prediction and actual value for each feature and frame to the row_data list
            prediction_value = predictions[i, j * num_output_features_per_frame + k]
            actual_value = y_test[i, j * num_output_features_per_frame + k]

            row_data.append(prediction_value)
            row_data.append(actual_value)

    # Append the row_data list to the excel_data list
    excel_data.append(row_data)

# Create a list of column headers for the Excel file
column_headers = []
for j in range(num_output_frames):
    for k in range(num_output_features_per_frame):
        column_headers.append(f'Frame_{j+1}_{output_features[k]}_Prediction')
        column_headers.append(f'Frame_{j+1}_{output_features[k]}_Actual_Value')

# Create a DataFrame using the excel_data and column_headers
result_df = pd.DataFrame(excel_data, columns=column_headers)

# Save the DataFrame to an Excel file
result_df.to_excel('predictions_and_actual_values.xlsx', index=False)


######################### calculating the errors ############################
# Step 1: Load the data from the Excel file
file_name = 'predictions_and_actual_values.xlsx'
df = pd.read_excel(file_name)

# Step 2: Implement functions to calculate the evaluation metrics
def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def average_displacement_error(df, num_frames):
    total_distance = 0.0
    num_samples = df.shape[0]

    for i in range(num_samples):
        for frame_num in range(1, num_frames + 1):
            if f'Frame_{frame_num}_xCenter_Prediction' not in df.columns:
                break  # Skip if the column does not exist

            x_pred = df[f'Frame_{frame_num}_xCenter_Prediction'].iloc[i]
            y_pred = df[f'Frame_{frame_num}_yCenter_Prediction'].iloc[i]
            x_gt = df[f'Frame_{frame_num}_xCenter_Actual_Value'].iloc[i]
            y_gt = df[f'Frame_{frame_num}_yCenter_Actual_Value'].iloc[i]

            distance = euclidean_distance(x_pred, y_pred, x_gt, y_gt)
            total_distance += distance

    ade = total_distance / (num_samples * num_frames)
    return ade

def average_final_displacement_error(df, num_frames):
    total_distance = 0.0
    num_samples = df.shape[0]

    for i in range(num_samples):
        if f'Frame_{num_frames}_xCenter_Prediction' not in df.columns:
            break  # Skip if the column does not exist

        x_pred = df[f'Frame_{num_frames}_xCenter_Prediction'].iloc[i]
        y_pred = df[f'Frame_{num_frames}_yCenter_Prediction'].iloc[i]
        x_gt = df[f'Frame_{num_frames}_xCenter_Actual_Value'].iloc[i]
        y_gt = df[f'Frame_{num_frames}_yCenter_Actual_Value'].iloc[i]

        distance = euclidean_distance(x_pred, y_pred, x_gt, y_gt)
        total_distance += distance

    fde = total_distance / num_samples
    return fde

def average_absolute_heading_error(df, num_frames):
    total_abs_heading_error = 0.0
    num_samples = df.shape[0]

    for i in range(num_samples):
        for frame_num in range(1, num_frames + 1):
            if f'Frame_{frame_num}_heading_Prediction' not in df.columns:
                break  # Skip if the column does not exist

            heading_pred = df[f'Frame_{frame_num}_heading_Prediction'].iloc[i]
            heading_gt = df[f'Frame_{frame_num}_heading_Actual_Value'].iloc[i]

            abs_heading_error = np.abs(heading_pred - heading_gt)
            total_abs_heading_error += abs_heading_error

    ahe = total_abs_heading_error / (num_samples * num_frames)
    return ahe

# Step 3: Calculate the number of frames and evaluate the performance metrics
num_frames = num_output_frames
ade_value = average_displacement_error(df, num_frames)
fde_value = average_final_displacement_error(df, num_frames)
ahe_value = average_absolute_heading_error(df, num_frames)

# Step 4: Print the evaluation results
print("Average Displacement Error (ADE):", ade_value)
print("Average Final Displacement Error (FDE):", fde_value)
print("Average Absolute Heading Error (AHE):", ahe_value) 

##############################################################################
# Plotting the Loss Curve
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


###########################################################################

# Initialize empty lists to store ADE values and time points
ade_values = []
fde_values = []
time_points = []

# Iterate over different values of num_frames
for num_frames in range(1, num_output_frames + 1):
    ade_value = average_displacement_error(df, num_frames)
    ade_values.append(ade_value)
    fde_value = average_final_displacement_error(df, num_frames)
    fde_values.append(fde_value)
    
    # Calculate time for this num_frames value
    time_points.append(num_frames * (skip_width + 1) / 25)  # Assuming 25 frames per second

# Create a plot
plt.figure(figsize=(10, 6))
plt.plot(time_points, ade_values, marker='o', linestyle='-', label='ADE')
plt.plot(time_points, fde_values, marker='x', linestyle='-', label='FDE')
plt.title('ADE and FDE vs Time')
plt.xlabel('Time (seconds)')
plt.ylabel('Average Displacement Error (ADE)')
plt.grid(True)
plt.legend()
plt.savefig('LSTM ADE and FDE vs Time.png', dpi=300)
plt.show()
