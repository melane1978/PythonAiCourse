# Save this code in a new Python file, for example, crime_prediction_2024.py

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from crimeMl import data, X_train, CrimePredictor, CrimeDataset

# Prediction for 2024
# Define the number of samples and features for the dummy data
num_samples_2024 = 45 # You can adjust this as needed
num_features = X_train.shape[1]  # Number of features in your training data

# Generate random dummy data for the year 2024 with the same structure as your training data
# You may need to adjust this based on the actual structure of your training data
X_2024_dummy = np.random.rand(num_samples_2024, num_features)  # Generate random features
# You can also generate specific values or patterns based on your understanding of the data

# Print the first few rows of the real data sample
real_data_sample = data.head()
print("Sample of Real Data:")
print(real_data_sample)

# Optionally, you can display the first few rows of the dummy data to verify the structure
print("Dummy Data for 2024:")
print(X_2024_dummy[:5])  # Display the first 5 rows as an example

# Define a function to generate dummy data for the year 2024
# Define a function to generate dummy data for the year 2024
def generate_dummy_data(grouped_data, num_regions=33):  # Assuming there are 33 regions
    # Calculate historical averages for each feature
    historical_averages = grouped_data.groupby('Crime_Type').mean()
    
    # Initialize an empty list to store the dummy data
    dummy_data_list = []
    
    # Generate dummy data for each crime type
    for crime_type, avg_values in historical_averages.iterrows():
        # Generate dummy data for each month in 2024
        for month in range(1, 13):
            # Generate random region indices 
            region_indices = np.random.choice(num_regions, size=num_regions, replace=False)
            # Adjust values based on historical averages (you can modify this based on your data)
            dummy_values = {
                'Year': 2024,
                'Crime_Type': crime_type,
                'Antal_lag_1': np.random.normal(avg_values['Antal_lag_1'], 10),
                'Antal_lag_2': np.random.normal(avg_values['Antal_lag_2'], 10),
                'Antal_lag_3': np.random.normal(avg_values['Antal_lag_3'], 10),
                'Antal_lag_4': np.random.normal(avg_values['Antal_lag_4'], 10),
                'Antal_lag_5': np.random.normal(avg_values['Antal_lag_5'], 10),
                'Antal_lag_6': np.random.normal(avg_values['Antal_lag_6'], 10),
                'Antal_lag_7': np.random.normal(avg_values['Antal_lag_7'], 10),
                'Antal_lag_8': np.random.normal(avg_values['Antal_lag_8'], 10),
                'Antal_lag_9': np.random.normal(avg_values['Antal_lag_9'], 10),
                'Antal_lag_10': np.random.normal(avg_values['Antal_lag_10'], 10),
                'Antal_lag_11': np.random.normal(avg_values['Antal_lag_11'], 10),
                'Antal_lag_12': np.random.normal(avg_values['Antal_lag_12'], 10),
            }
            # Add region data to the dummy values
            for i, region_index in enumerate(region_indices):
                dummy_values[f'Region_{region_index+1}'] = np.random.randint(0, 2)  # Randomly assign 0 or 1 for region presence
            # Append dummy values to the dummy data list
            dummy_data_list.append(dummy_values)
    
    # Create a DataFrame from the list of dummy data
    dummy_data = pd.DataFrame(dummy_data_list)
    
    return dummy_data

# Generate dummy data for 2024
dummy_data_2024 = generate_dummy_data(data)
print(dummy_data_2024)
print(data)

# Prepare the dataset for 2024 dummy data
X_2024 = dummy_data_2024.drop(['Year', 'Crime_Type'], axis=1).values.astype(float)

# # Create a CrimeDataset instance for 2024 dummy data
# dataset_2024 = CrimeDataset(X_2024, np.zeros(len(X_2024)))  # Target values are not available for 2024 yet, so zeros are used

# # Define a DataLoader for the dataset
# dataloader_2024 = DataLoader(dataset_2024, batch_size=64, shuffle=False)

# # Load the trained model
# model = CrimePredictor(input_size=X_2024.shape[1])  # Update input_size to match the number of features in X_2024
# model.load_state_dict(torch.load("./CrimeML/TestingML/saved_models/crime_prediction_model.pth"))
# model.eval()

# # Make predictions for 2024
# predictions_2024 = []
# with torch.no_grad():
#     for inputs, _ in dataloader_2024:
#         outputs = model(inputs)
#         predictions_2024.extend(outputs.numpy())

# # Optionally, you can save the predictions to a DataFrame
# predictions_df = pd.DataFrame({
#     'Crime_Type': dummy_data_2024['Crime_Type'],
#     'Predicted_Antal_lag_1': predictions_2024  # Adjust this accordingly if you have multiple lag columns
# })

# # Print or save the predictions as needed
# print(predictions_df.head())
