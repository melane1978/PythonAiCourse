import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Lets load my data, i had do redo it because i notice it was many duplicate values in the first file
data = pd.read_csv('./CrimeML/data/crimedata.csv')

# separate columns, combine them into a datetime column, easier to have on column that two
data['Date'] = pd.to_datetime(data[['Year', 'Month']].assign(day=1))


# Data preprocessing
scaled_data = pd.DataFrame()
scaled_dfs = []

# Here i took help from chatgp, because did'nt really know how to break down all my data and make features of it the right way,
# because MATH is not my strong side.
#Scalers are used to scale or normalize the data, ensuring that each feature contributes equally to the analysis 
#and preventing features with larger scales from dominating the model.

# Iterate over each group of data (grouped by crime type and region)
for (crime_type, region), group in data.groupby(['Crime', 'Region']):
    # Initialize a MinMaxScaler for the current group
    scaler = MinMaxScaler(feature_range=(-1, 1))
    
    # Scale the 'Antal' values of the current group
    scaled_values = scaler.fit_transform(group['Antal'].values.reshape(-1, 1))
    
    # Create a column name based on crime type and region
    column_name = f'{crime_type}_Region_{region}'
     
    # Create a DataFrame with scaled values and set column name
    scaled_df = pd.DataFrame(scaled_values.flatten(), columns=[column_name])
    
    # Append the scaled DataFrame to the list
    scaled_dfs.append(scaled_df)

    # Concatenate all scaled DataFrames along axis=1 to form the final DataFrame
    scaled_data = pd.concat(scaled_dfs, axis=1)

#print(scaled_data)

# Convert our scaled data to pytorch sensor, 
crime_tensors = {column_name: torch.Tensor(scaled_data[column_name].values).view(-1, 1)
                 for column_name in scaled_data.columns}

#print(crime_tensors)

# # Define a simple neural network model
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Time to train my model for each crime type and region
models = {}
epoch = 10000

for column_name, crime_tensor in crime_tensors.items():
    input_data = crime_tensor[:-1].view(-1, 1)
    target_data = crime_tensor[1:].view(-1, 1)
    model = NeuralNetwork(input_size=1, hidden_size=64, output_size=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epoch):
        optimizer.zero_grad()
        output = model(input_data)
        loss = criterion(output, target_data)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f'Crime Type and Region: {column_name}, Epoch [{epoch+1}/{1000}], Loss: {loss.item():.4f}')
    models[column_name] = model


# Start with my prediction for all months in 2024
scalers = {} 
for (crime_type, region), group in data.groupby(['Crime', 'Region']):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    # Fit the scaler to the 'Antal' values of the current group
    scaler.fit(group['Antal'].values.reshape(-1, 1))
    # Save the scaler in the scalers dictionary
    scalers[f'{crime_type}_Region_{region}'] = scaler

predictions = {}
for column_name, model in models.items():
    predicted_values = []
    with torch.no_grad():
        test_input = crime_tensors[column_name][-1].reshape(1, 1)
        for _ in range(12):
            predicted = model(test_input).item()
            predicted_values.append(predicted)
            test_input = torch.tensor([[predicted]])
    # Inverse transform the predicted values using the appropriate scaler
    predictions[column_name] = scalers[column_name].inverse_transform(np.array(predicted_values).reshape(-1, 1))

# Save predictions to a CSV file, so i can use it for my predict2024v1 file.
predictions_data = []

for column_name, predicted_values in predictions.items():
    crime_type, region = column_name.split('_Region_')
    for month, crime_count in zip(range(1, 13), predicted_values):
        predictions_data.append({
            'Crime_Type': crime_type,
            'Region': region,
            'Month': month,
            'Year': 2024,
            'Predicted_Crime': crime_count[0]
        })

predictions_df = pd.DataFrame(predictions_data)
predictions_df.to_csv('./CrimeML/data/predictions_2024.csv', index=False)

# Save trained models to files
for column_name, model in models.items():
    torch.save(model.state_dict(), f'./CrimeML/TestingML/saved_models/{column_name}_model.pth')

