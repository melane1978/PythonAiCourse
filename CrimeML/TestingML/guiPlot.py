import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import ttk

# Load the data from 2023 and compare with predicted data
data_2023 = pd.read_csv('./CrimeML/data/data2023.csv')
predictions_2024 = pd.read_csv('./CrimeML/data/predictions_2024_reshaped.csv')

# Remove duplicates of region for the menu below
regions = sorted(data_2023['Region'].unique())
crimes = sorted(data_2023['Crime'].unique())

#  plot data for the selected region and crime type
def plot_comparison():
    region_index = region_combobox.current()
    selected_region = regions[region_index]

    crime_index = crime_combobox.current()
    selected_crime = crimes[crime_index]

    data_2023_compare = data_2023[(data_2023['Region'] == selected_region) & (data_2023['Crime'] == selected_crime)]
    predictions_2024_compare = predictions_2024[(predictions_2024['Region'] == selected_region) & (predictions_2024['Crime'] == selected_crime)]

    plt.figure(figsize=(12, 8))
    plt.scatter(data_2023_compare['Month'], data_2023_compare['Antal'], label='2023 Data')
    plt.scatter(predictions_2024_compare['Month'], predictions_2024_compare['Predicted_Crime'], label='2024 Predictions')
    plt.yticks(np.arange(0, 525, 25))
    plt.xlabel('Month')
    plt.ylabel('Antal')
    plt.title(f'Comparison of Data 2023 and Predictions 2024 for {selected_region}, {selected_crime}')
    plt.legend()
    plt.show()

# Create the main window
root = tk.Tk()
root.title("Crime Data Analysis")

# Create a frame to hold the widgets
frame = ttk.Frame(root, padding="20")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# Menu for selecting region
region_label = ttk.Label(frame, text="Select a region:")
region_label.grid(row=0, column=0, sticky=tk.W)
region_combobox = ttk.Combobox(frame, values=regions, state="readonly")
region_combobox.grid(row=0, column=1)

# update crime options based on the selected region
def update_crimes(event):
    selected_region = region_combobox.get()
    crimes_for_region = sorted(data_2023[data_2023['Region'] == selected_region]['Crime'].unique()) 
    crime_combobox.config(values=crimes_for_region)

# Bind the function to the combobox selection event
region_combobox.bind("<<ComboboxSelected>>", update_crimes)

# Menu for selecting crime type
crime_label = ttk.Label(frame, text="Select a crime type:")
crime_label.grid(row=1, column=0, sticky=tk.W)
crime_combobox = ttk.Combobox(frame, values=crimes, state="readonly")
crime_combobox.grid(row=1, column=1)

# A button 
confirm_button = ttk.Button(frame, text="Plot", command=plot_comparison)
confirm_button.grid(row=2, column=1, pady=10)

root.mainloop()
