import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np


# Load your dataset
data = pd.read_csv("./CrimeML/data/crimedata.csv")

# One-hot encoding for region column
data = pd.get_dummies(data, columns=['Region'])

# Aggregating data by Year and Crime Type
grouped_data = data.groupby(['Year', 'Crime']).sum().reset_index()

# Data Splitting
train_data = grouped_data[grouped_data['Year'] < 2023]  # Keep data until 2023 for training
test_data = grouped_data[grouped_data['Year'] == 2023]  # Use data from 2023 for testing


#Training and Evaluation for each Antal_Lag_* column
for lag_index in range(1, 13):
    lag_column = 'Antal_lag_' + str(lag_index)
   
    # # Selecting features and target variable
    X_train = train_data.drop(['Year', 'Crime', lag_column], axis=1).values.astype(float)
    y_train = train_data[lag_column].values.astype(float)
    

    X_test = test_data.drop(['Year', 'Crime', lag_column], axis=1).values.astype(float)
    y_test = test_data[lag_column].values.astype(float)

print("How many features do i have in my train data")
print(X_train.shape[1])


# Define dataset and dataloader
class CrimeDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = CrimeDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)



# Define the neural network architecture
class CrimePredictor(nn.Module):
    def __init__(self, input_size):
        super(CrimePredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Initialize model, loss function, and optimizer
model = CrimePredictor(input_size=X_train.shape[1])
print(model)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
num_epochs = 200000
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_dataset)
    print(f'Epoch: {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

# Evaluation
model.eval()
with torch.no_grad():
    y_pred = model(torch.tensor(X_test, dtype=torch.float32))
mse = mean_squared_error(y_test, y_pred.numpy())
print("Mean Squared Error:", mse)

# Save the trained model
torch.save(model.state_dict(), "./CrimeML/TestingML/saved_models/crime_prediction_model.pth")