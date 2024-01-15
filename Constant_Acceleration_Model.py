import time
start_time = time.time()
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import math
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import mean_squared_error


recording_id = "25"
# Create a Pandas Dataframe
data_track = pd.read_csv('data_processing/dataset/data/' + recording_id + '_tracks.csv')
data_meta = pd.read_csv('data_processing/dataset/data/' + recording_id + '_tracksMeta.csv')

actor = 'car'
new_data_track = data_meta[data_meta['class'] == actor]
new_data_track_id = new_data_track['trackId']

data_track_filtered = data_track[data_track['trackId'].isin(new_data_track_id)]

track_data_raw = data_track_filtered
track_meta_data_raw = data_meta
################################################################################
class DataPreprocessor:
    def __init__(self):
        self.tracks_data_down_sampled = pd.DataFrame()
        
    def downsample(self, tracks_data, skip_width):
        tracks_data_downsampled = tracks_data.iloc[skip_width::skip_width+1]
        tracks_data_downsampled.reset_index(drop=True, inplace=True)
        return tracks_data_downsampled
        
########################################################################################
pre_process_obj = DataPreprocessor()

skip_width = 0
tracks_data_down_sampled = pre_process_obj.downsample(track_data_raw, skip_width)

# Extracting ground truth data as Dataframe
data = tracks_data_down_sampled.iloc[:, 4:6]

################### Acceleration model ######################################
class ConstantAcceleration(nn.Module):
    def __init__(self, future_sequence_length):
        super().__init__()
        self.future_sequence_length = future_sequence_length
        self.layers = nn.Identity()
    
    def forward(self, x):
        # Create a Numpy Array from Dataframe
        positions = (x.iloc[:, 4:6]).values
        velocities = (x.iloc[:, 9:11]).values
        accelerations = (x.iloc[:, 11:13]).values
        predictions = []
        
        #### Done using previous Predicted Position Value as base value in formula for next prediction
        pred_position1 = positions[0] + velocities[0] * ((skip_width+1)/25) + 0.5 * accelerations[0] * ((skip_width+1)/25)**2
        predictions.append((pred_position1).tolist())
        
        velocity = velocities[0] + accelerations[0]
        
        pred_positions_list = [pred_position1]
        
        for t in range(self.future_sequence_length-1):
            pred_positions = pred_positions_list[-1] + velocity * ((skip_width+1)/25) + 0.5 * accelerations[0] * ((skip_width+1)/25)**2
            velocity = velocity + accelerations[0] * ((skip_width+1)/25)
            
            # To get current predicted position for next iteration
            pred_positions_list.append(pred_positions)
            
            predictions.append((pred_positions).tolist())
            
            # Converting list into a PyTorch tensor
            predictions1 = torch.tensor(predictions)
        return predictions1
        
    def loss_function(self, y, y_hat):
        loss = F.mse_loss(y_hat, y)
        return loss

########################## Calling the function #########################
future_sequence_length = 25

acc = ConstantAcceleration(future_sequence_length)
predictions = acc.forward(tracks_data_down_sampled)
predictions_x = predictions[: , 0]
predictions_y = predictions[: , 1]

# Extract DF vals as Numpy array and convert it into a Pytorch tensor
data_x = torch.tensor((data.values)[:future_sequence_length, 0])#[0]
data_y = torch.tensor((data.values)[:future_sequence_length, 1])#[1]

elapsed_time1 = time.time() - start_time
print(f"Elapsed Time before saving predictions is {elapsed_time1}")

error_x = acc.loss_function(data_x, predictions_x)
error_y = acc.loss_function(data_y, predictions_y)
#########################################################################

datax_np = data_x.numpy()
datay_np = data_y.numpy()
predictions_x_np = predictions_x.numpy()
predictions_y_np = predictions_y.numpy()
distance = np.sqrt((datax_np - predictions_x_np)**2 + (datay_np - predictions_y_np)**2)
print("Deviation in distance for", actor ,"is", distance[-1], 'meters')

# Create a DF with the data
df = pd.DataFrame({
    'xCenter': datax_np,
    'yCenter': datay_np,
    'xCenter_predicted': predictions_x_np,
    'yCenter_predicted': predictions_y_np,
    'Deviation in Distance': distance
})

# Save the DataFrame to an Excel file
file_path = 'Prediction and Actual Values Acc Model.xlsx'
df.to_excel(file_path, index=False)


average_distance_errors = []
for t in range(future_sequence_length):
    subset_distance = distance[:t + 1]
    average_error = np.mean(subset_distance)
    average_distance_errors.append(average_error)

average_distance_error = np.mean(distance)

# Print the result
print('Average distance error is', average_distance_error, 'meters')

mse_total = np.sqrt(error_x**2 + error_y**2)

# Print the MSE values
print(f"Total Mean Squared Error (MSE): {mse_total} meters")

############################################################################
time_range = np.linspace(1/25, future_sequence_length/25, num=future_sequence_length)

plt.figure(figsize=(10, 6))
plt.plot(time_range, distance, marker='o', linestyle='-', label='Deviated Distance')
plt.plot(time_range, average_distance_errors, marker='x', linestyle='-', label='Average Distance Error')
plt.xlabel('Time [seconds]')
plt.ylabel('Deviation in Distance [m]')
plt.title('Distance deviation and ADE vs Time')
plt.grid(True)
plt.legend()
plt.savefig('Acc Distance deviation and ADE vs Time.png', dpi=300)
plt.show()

