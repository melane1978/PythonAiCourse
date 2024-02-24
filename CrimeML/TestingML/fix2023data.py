import pandas as pd

# Mapping dictionary for months
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

# Load the 2023 data
data_2023 = pd.read_csv('./CrimeML/data/data2023.csv')

# Replace 'Month' values with month names
data_2023['Month'] = data_2023['Month'].map(month_mapping)

# Remove duplicates
data_2023.drop_duplicates(inplace=True)

# Save the modified DataFrame to a new CSV file
data_2023.to_csv('./CrimeML/data/data2023.csv', index=False)
