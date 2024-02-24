import pandas as pd

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

# Mapping dictionary for crime types
crime_type_mapping = {
    0: '3 kap. Brott mot liv och hälsa',
    1: '3-7 kap. Brott mot person',
    2: '4 kap. Brott mot frihet och frid',
    3: '5 kap. Ärekränkningsbrott',
    4: '6 kap. Sexualbrott',
    5: '7 kap. Brott mot familj',
    6: 'därav misshandel inkl. grov'
}

# Load the 2023 data
data_2023 = pd.read_csv('./CrimeML/data/2023.csv')

# Replace 'Region' values with municipality names
data_2023['Region'] = data_2023['Region'].apply(lambda x: municipalities[int(x)])

# Replace 'Month' values with month names
data_2023['Month'] = data_2023['Month'].map(month_mapping)

# Replace 'Crime' values with crime type names
data_2023['Crime'] = data_2023['Crime'].map(crime_type_mapping)

# Remove duplicates
data_2023.drop_duplicates(inplace=True)

# Save the modified DataFrame to a new CSV file
data_2023.to_csv('./CrimeML/data/data2023.csv', index=False)
