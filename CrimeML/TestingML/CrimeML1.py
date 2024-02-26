import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the data
data = pd.read_csv('./CrimeML/data/crimedata.csv')

# Assuming 'Month' and 'Year' are separate columns, combine them into a datetime column
data['Date'] = pd.to_datetime(data[['Year', 'Month']].assign(day=1))

# Data preprocessing
scalers = {}  # Dictionary to store scalers for each crime type and region
scaled_data = pd.DataFrame()

for (crime_type, region), group in data.groupby(['Crime', 'Region']):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_values = scaler.fit_transform(group['Antal'].values.reshape(-1, 1))
    column_name = f'{crime_type}_Region_{region}'
    scaled_data[column_name] = scaled_values.flatten()
    scalers[column_name] = scaler

# Convert data to PyTorch tensors
crime_tensors = {column_name: torch.Tensor(scaled_data[column_name].values).view(-1, 1)
                 for column_name in scaled_data.columns}

# Define a simple neural network model
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Train the model for each crime type and region
models = {}
for column_name, crime_tensor in crime_tensors.items():
    input_data = crime_tensor[:-1].view(-1, 1)
    target_data = crime_tensor[1:].view(-1, 1)
    model = SimpleNN(input_size=1, hidden_size=64, output_size=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(1000):
        optimizer.zero_grad()
        output = model(input_data)
        loss = criterion(output, target_data)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f'Crime Type and Region: {column_name}, Epoch [{epoch+1}/1000], Loss: {loss.item():.4f}')
    models[column_name] = model

# Prediction for all months in 2024
predictions = {}
for column_name, model in models.items():
    predicted_values = []
    with torch.no_grad():
        test_input = crime_tensors[column_name][-1].reshape(1, 1)
        for _ in range(12):  # Predict for 12 months
            predicted = model(test_input).item()
            predicted_values.append(predicted)
            test_input = torch.tensor([[predicted]])
    predictions[column_name] = scalers[column_name].inverse_transform(np.array(predicted_values).reshape(-1, 1))

# Display predicted crime for all months in 2024 for each crime type and region
# for column_name, predicted_values in predictions.items():
#     crime_type, region = column_name.split('_Region_')
#     print(f'Crime Type: {crime_type}, Region: {region}')
#     for month, crime_count in zip(range(1, 13), predicted_values):
#         print(f'  Predicted crime for month {month} in 2024:', crime_count[0])


# Save predictions to a CSV file
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



