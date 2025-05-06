# preprocess_data.py

import pandas as pd
import numpy as np
from scipy.stats import percentileofscore
from sklearn.preprocessing import StandardScaler
import pickle

# Load the dataset
df = pd.read_csv("data.csv")

# List of columns to clean and convert
columns_to_clean = ['Income', 'Upward mobility', 'Life Expectancy']

# Replace non-numeric characters and convert to float
for col in columns_to_clean:
    df[col] = pd.to_numeric(df[col].replace(r'[^0-9.]', '', regex=True), errors='coerce')

# Fill NaN values with the mean of the column, after ensuring the column is numeric
for col in columns_to_clean:
    df[col] = df[col].fillna(df[col].mean())

# Calculate population percentiles for each county

df['population_percentile'] = df['Population'].apply(lambda x: percentileofscore(df['Population'], x))

# List of features to normalize (racial demographics)
racial_features = ['% Black', 
                   '% American Indian or Alaska Native', '% Asian',
                   '% Native Hawaiian or Other Pacific Islander', '% Hispanic', '% Non-Hispanic White']

# Other features to consider (including non-racial features)
other_features = ['Population', '% Rural']
features = racial_features + other_features

# Convert features to numeric
for feature in features:
    df[feature] = pd.to_numeric(df[feature], errors='coerce').fillna(0)

# Calculate population percentiles if not already present
if 'population_percentile' not in df.columns:
    df['population_percentile'] = pd.qcut(df['Population'], q=100, labels=False)


# Scale the features in the DataFrame
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[features])

# Save the scaler to a file for future use in the Dash app
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Define custom weights for features
race_weights ={'% Black': 50, '% American Indian or Alaska Native': 50, '% Asian': 50,
                   '% Native Hawaiian or Other Pacific Islander': 50, '% Hispanic': 50, '% Non-Hispanic White': 10}
non_race_weights = {'Population': 1, '% Rural': 50}

# Increase weight for any racial category greater than 30%
for col in racial_features:
    if df[col].mean() > 25:
        race_weights[col] *= 2  # Double the weight

all_weights = np.array([race_weights.get(col, non_race_weights.get(col, 1)) for col in features])

# Save the weights for future use in the Dash app
with open('weights.pkl', 'wb') as f:
    pickle.dump(all_weights, f)

# Sort the states and counties alphabetically
sorted_states = sorted(df['State'].unique())
sorted_counties = sorted(df[['County', 'State']].sort_values(by='County')['County'].unique())

df['More Info'] = ''


# Save the preprocessed data to a CSV (optional for later use)
df.to_csv('preprocessed_data.csv', index=False)

print("Preprocessing complete and saved.")
