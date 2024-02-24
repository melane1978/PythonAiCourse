# Save this code in a new Python file, for example, crime_prediction_2024.py
import subprocess
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from crimeMl import data, X_train, CrimePredictor, CrimeDataset, grouped_data


num_samples_2024 = 46
num_features = X_train.shape[1] 
X_2024_dummy = np.random.rand(num_samples_2024, num_features)  # Generate random features
real_data_sample = data.head()
# print(X_2024_dummy[:5])  # Display the first 5 rows as an example

def generate_dummy_data(grouped_data, num_regions=33):
    # Calculate historical averages for each feature
    historical_averages = grouped_data.groupby('Crime').mean()

    # Initialize an empty list to store the dummy data
    dummy_data_list = []

    # Generate dummy data for each crime type
    for Crime, avg_values in historical_averages.iterrows():
        # Generate dummy data for each month in 2024
        for month in range(1, 13):
            # Generate random region indices
            region_indices = np.random.choice(num_regions, size=num_regions, replace=False)
            # Assign values to lagged features, rounding only Antal_lag_* columns
            dummy_values = {
                'Year': 2024,
                'Month': month,
                'Crime': Crime,
                'Antal_lag_1': round(np.random.normal(avg_values['Antal_lag_1'], 10), 1),
                'Antal_lag_2': round(np.random.normal(avg_values['Antal_lag_2'], 10), 1),
                'Antal_lag_3': round(np.random.normal(avg_values['Antal_lag_3'], 10), 1),
                'Antal_lag_4': round(np.random.normal(avg_values['Antal_lag_4'], 10), 1),
                'Antal_lag_5': round(np.random.normal(avg_values['Antal_lag_5'], 10), 1),
                'Antal_lag_6': round(np.random.normal(avg_values['Antal_lag_6'], 10), 1),
                'Antal_lag_7': round(np.random.normal(avg_values['Antal_lag_7'], 10), 1),
                'Antal_lag_8': round(np.random.normal(avg_values['Antal_lag_8'], 10), 1),
                'Antal_lag_9': round(np.random.normal(avg_values['Antal_lag_9'], 10), 1),
                'Antal_lag_10': round(np.random.normal(avg_values['Antal_lag_10'], 10), 1),
                'Antal_lag_11': round(np.random.normal(avg_values['Antal_lag_11'], 10), 1),
                'Antal_lag_12': round(np.random.normal(avg_values['Antal_lag_12'], 10), 1),
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

# print("dummy_data_2024")
#print(dummy_data_2024)
#print(grouped_data)

# Prepare the dataset for 2024 dummy data
X_2024 = dummy_data_2024.drop(['Year', 'Crime'], axis=1).values.astype(float)

# # Create a CrimeDataset instance for 2024 dummy data
dataset_2024 = CrimeDataset(X_2024, np.zeros(len(X_2024)))  # Target values are not available for 2024 yet, so zeros are used
print(dataset_2024)

# # Define a DataLoader for the dataset
dataloader_2024 = DataLoader(dataset_2024, batch_size=64, shuffle=False)


# Check the number of features in  training data (X_train)
num_features_training = X_train.shape[1]
print("Number of features in training data:", num_features_training)

# Check the number of features in  input data for 2024 (X_2024)
num_features_2024 = X_2024.shape[1]
print("Number of features in 2024 input data:", num_features_2024)

model = CrimePredictor(input_size=X_2024.shape[1])  # Update input_size to match the number of features in X_2024
print(model)
model.load_state_dict(torch.load("./CrimeML/TestingML/saved_models/crime_prediction_model.pth"))
model.eval()

#Make predictions for 2024
#Make predictions for 2024
predictions_2024 = []
with torch.no_grad():
    for inputs, _ in dataloader_2024:
        outputs = model(inputs)
        predictions_2024.extend(outputs.numpy())

# Convert predictions_2024 to a numpy array
predictions_2024 = np.array(predictions_2024)

# Extract predicted lag columns names
predicted_lag_columns = [f'Predicted_Antal_lag_{i}' for i in range(1, 13)]
# Define the number of regions based on the dataset you provided
num_regions = 33


# Mapping dictionary for month names
month_mapping = {
    1: 'January',
    2: 'February',
    3: 'March',
    4: 'April',
    5: 'May',
    6: 'June',
    7: 'July',
    8: 'August',
    9: 'September',
    10: 'October',
    11: 'November',
    12: 'December'
}

# Mapping dictionary for crime types
Crime_mapping = {
    1: '3 kap. Brott mot liv och hälsa',
    2: '3-7 kap. Brott mot person',
    3: '4 kap. Brott mot frihet och frid',
    4: '5 kap. Ärekränkningsbrott',
    5: '6 kap. Sexualbrott',
    6: '7 kap. Brott mot familj',
    7: 'därav misshandel inkl. grov'
}

# Define the list of municipalities
municipalities = ['Botkyrka kommun', 'Danderyd kommun', 'Haninge kommun', 'Huddinge kommun',
              'Järfälla kommun', 'Knivsta kommun', 'Lidingö kommun', 'Nacka kommun',
              'Norrtälje kommun', 'Salem kommun', 'Sigtuna kommun', 'Sollentuna kommun',
              'Solna kommun', 'Bromma (Sthlm)', 'Enskede - Årsta - Vantör (Sthlm)',
              'Farsta (Sthlm)', 'Hägersten-Älvsjö (Sthlm)', 'Hässelby - Vällingby (Sthlm)',
              'Kungsholmen (Sthlm)', 'Norrmalm (Sthlm)', 'Rinkeby-Kista (Sthlm)',
              'Skarpnäck (Sthlm)', 'Skärholmen (Sthlm)', 'Spånga - Tensta (Sthlm)',
              'Sundbyberg kommun', 'Södertälje kommun', 'Tyresö kommun', 'Täby kommun',
              'Upplands Väsby kommun', 'Vallentuna kommun', 'Vaxholm kommun',
              'Värmdö kommun', 'Österåker kommun']

# Assuming predictions_df is the DataFrame containing the predictions
predictions_df = pd.DataFrame({
    'Year': dummy_data_2024['Year'],
    'Month': dummy_data_2024['Month'].map(month_mapping),  # Map month numbers to month names
    'Crime': dummy_data_2024['Crime'].map(Crime_mapping)
})

# Add predicted lag columns to the DataFrame
for i, col in enumerate(predicted_lag_columns):
    predictions_df[col] = predictions_2024[:, 0]  # Use the correct index for predictions_2024

# Create a DataFrame to hold the reshaped data
reshaped_data = []

# Populate the reshaped DataFrame
for region in municipalities:
    for index, row in predictions_df.iterrows():
        for lag_col in predicted_lag_columns:
            reshaped_data.append({
                'Region': region,
                'Month': row['Month'],
                'Crime': row['Crime'],
                'Antal': row[lag_col]
            })

# Convert reshaped data to DataFrame
reshaped_df = pd.DataFrame(reshaped_data)

# Sort the reshaped DataFrame by month and region
reshaped_df = reshaped_df.sort_values(by=['Month', 'Region']).reset_index(drop=True)

# Drop duplicate rows from the reshaped DataFrame
reshaped_df = reshaped_df.drop_duplicates()

# Print or save the predictions as needed
print(reshaped_df.head())

# Save the reshaped DataFrame to a CSV file
reshaped_df.to_csv('./CrimeML/data/predictions_2024_reshaped.csv', index=False)


# Run the plot script

try:
    # Run another Python file
    subprocess.run(["python", "./CrimeML/TestingML/plot.py"])
except Exception as e:
    print(f"An error occurred: {e}")

