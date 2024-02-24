import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data_2023 = pd.read_csv('./CrimeML/data/data2023.csv')
predictions_2024 = pd.read_csv('./CrimeML/data/predictions_2024_reshaped.csv')

# Get unique regions
regions = data_2023['Region'].unique()

while True:
    # Prompt user to select a region
    print("Select a region to compare:")
    for i, region in enumerate(regions):
        print(f"{i+1}. {region}")
    region_index = int(input("Enter the index of the region (Enter 0 to exit): ")) - 1
    
    if region_index == -1:
        print("Exiting the script...")
        break
    
    selected_region = regions[region_index]

    # Get unique crime types for the selected region
    Crimes = data_2023[data_2023['Region'] == selected_region]['Crime'].unique()

    # Prompt user to select a crime type
    print("\nSelect a crime type to compare:")
    for i, Crime in enumerate(Crimes):
        print(f"{i+1}. {Crime}")
    Crime_index = int(input("Enter the index of the crime type (Enter 0 to select a different region): ")) - 1
    
    if Crime_index == -1:
        continue
    
    selected_Crime = Crimes[Crime_index]

    # Filter data for comparison
    data_2023_compare = data_2023[(data_2023['Region'] == selected_region) & (data_2023['Crime'] == selected_Crime)]
    predictions_2024_compare = predictions_2024[(predictions_2024['Region'] == selected_region) & (predictions_2024['Crime'] == selected_Crime)]

    # Plotting
    plt.figure(figsize=(12, 8))

    # Plot data from 2023
    plt.scatter(data_2023_compare['Month'], data_2023_compare['Antal'], label='2023 Data')

    # Plot predicted data for 2024
    plt.scatter(predictions_2024_compare['Month'], predictions_2024_compare['Antal'], label='2024 Predictions')

    # Adding labels and title
    plt.xlabel('Month')
    plt.ylabel('Antal')
    plt.title(f'Comparison of Data 2023 and Predictions 2024 for {selected_region}, {selected_Crime}')
    plt.legend()

    # Show plot
    plt.show()
