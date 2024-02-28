import pandas as pd

# These Values i got from my Jupyter file
# Mapping dictionary for month names
month_mapping = {
    "Jan": 'January',
    "Feb": 'February',
    "Mar": 'March',
    "Apr": 'April',
    "May": 'May',
    "Jun": 'June',
    "Jul": 'July',
    "Aug": 'August',
    "Sep": 'September',
    "Oct": 'October',
    "Nov": 'November',
    "Dec": 'December'
}

# Mapping dictionary for crime types
crime_mapping = {
    1: '3 kap. Brott mot liv och hälsa',
    2: '3-7 kap. Brott mot person',
    3: '4 kap. Brott mot frihet och frid',
    4: '5 kap. Ärekränkningsbrott',
    5: '6 kap. Sexualbrott',
    6: '7 kap. Brott mot familj',
    7: 'därav misshandel inkl. grov'
}

# Mapping dictionary for region codes to names
region_mapping = {
    1: 'Botkyrka kommun', 
    2: 'Danderyd kommun', 
    3: 'Haninge kommun', 
    4: 'Huddinge kommun',
    5: 'Järfälla kommun', 
    6: 'Knivsta kommun', 
    7: 'Lidingö kommun', 
    8: 'Nacka kommun',
    9: 'Norrtälje kommun', 
    10: 'Salem kommun', 
    11: 'Sigtuna kommun', 
    12: 'Sollentuna kommun',
    13: 'Solna kommun', 
    14: 'Bromma (Sthlm)', 
    15: 'Enskede - Årsta - Vantör (Sthlm)', 
    16: 'Farsta (Sthlm)',
    17: 'Hägersten-Älvsjö (Sthlm)', 
    18: 'Hässelby - Vällingby (Sthlm)', 
    19: 'Kungsholmen (Sthlm)', 
    20: 'Norrmalm (Sthlm)', 
    21: 'Rinkeby-Kista (Sthlm)',
    22: 'Skarpnäck (Sthlm)', 
    23: 'Skärholmen (Sthlm)', 
    24: 'Spånga - Tensta (Sthlm)', 
    25: 'Sundbyberg kommun', 
    26: 'Södertälje kommun', 
    27: 'Tyresö kommun', 
    28: 'Täby kommun',
    29: 'Upplands Väsby kommun', 
    30: 'Vallentuna kommun', 
    31: 'Vaxholm kommun',
    32: 'Värmdö kommun', 
    33: 'Österåker kommun'
}

# Read predictions data from the provided CSV file
predictions_df = pd.read_csv('./CrimeML/data/data2023.csv')

# Create a DataFrame to hold the reshaped data
reshaped_data = []

# Replace all int numbers to words and append it to a new list
for index, row in predictions_df.iterrows():
    region_name = region_mapping[int(row['Region'])]
    reshaped_data.append({
        'Region': region_name,
        'Month': month_mapping[row['Month']],
        'Crime': crime_mapping[row['Crime']],
        'Antal': round(row['Antal'])  # Round and convert to integer
    })


# Convert reshaped data to DataFrame
reshaped_df = pd.DataFrame(reshaped_data)
# Convert the 'Month' column to categorical with the specified order
reshaped_df['Month'] = pd.Categorical(reshaped_df['Month'], categories=month_mapping.values(), ordered=True)

# Sort the reshaped DataFrame by month and region
reshaped_df = reshaped_df.sort_values(by=['Region', 'Month']).reset_index(drop=True)
reshaped_df = reshaped_df.drop_duplicates()

reshaped_df.to_csv('./CrimeML/data/data2023.csv', index=False)
