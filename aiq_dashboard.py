import numpy as np
import pandas as pd
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.express as px
import gspread
from gspread_dataframe import set_with_dataframe
from scipy.stats import percentileofscore
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import plotly.graph_objects as go
import requests
import json

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

# Define the features to consider (including the non-racial features)
other_features = ['Population', '% Rural']
features = racial_features + other_features

# Convert features to numeric
for feature in features:
    df[feature] = pd.to_numeric(df[feature], errors='coerce').fillna(0)

# Calculate population percentiles if not already present
if 'population_percentile' not in df.columns:
    df['population_percentile'] = pd.qcut(df['Population'], q=100, labels=False)

# ------------------------------------------------------
# Create the Dash app
app = dash.Dash(__name__)
server = app.server  # This is necessary for gunicorn to recognize the server

# GeoJSON data for counties
with open('geojson-counties-fips.json', 'r') as file:
    counties = json.load(file)

# Layout for the dashboard
app.layout = html.Div([
    html.H1("County Comparison Dashboard"),
    
    # Dropdown for selecting state and county
    dcc.Dropdown(
        id='state-dropdown',
        options=[{'label': state, 'value': state} for state in df['State'].unique()],
        value='South Dakota',  # default value
        placeholder="Select a State"
    ),
    
    dcc.Dropdown(
        id='county-dropdown',
        options=[],
        value='Oglala Lakota',  # default value
        placeholder="Select a County"
    ),

    html.Br(),

    html.Div(id='county-data-container'),
    html.Div(id='table-container'),

    html.Br(),
    
    # Dropdown for selecting the variable to display in the bar chart
    dcc.Dropdown(
        id='variable-dropdown',
        options=[
            {'label': 'Income', 'value': 'Income'},
            {'label': 'Life Expectancy', 'value': 'Life Expectancy'},
            {'label': 'Upward Mobility', 'value': 'Upward mobility'}
        ],
        value='Income',  # default value
        placeholder="Select a variable to compare",
        style={'width': '50%', 'padding': '10px'}
    ),

    html.Br(),
      # Main content section with bar chart and map
    html.Div([
        # Left column: Bar chart
        html.Div([
            dcc.Graph(id='income-comparison-bar-chart')
        ], style={'width': '48%', 'display': 'inline-block'}),  # Bar chart on the left

        # Right column: Choropleth map
        html.Div([
            dcc.Graph(id='choropleth-map')
        ], style={'width': '48%', 'display': 'inline-block', 'padding-left': '2%'})  # Map on the right
    ], style={'display': 'flex', 'flex-direction': 'row'})
])

# Callback to update the county dropdown based on the state selected
@app.callback(
    Output('county-dropdown', 'options'),
    Output('county-dropdown', 'value'),
    Input('state-dropdown', 'value')
)
def update_county_dropdown(state):
    counties = df[df['State'] == state]['County'].unique()
    return [{'label': county, 'value': county} for county in counties], counties[0]

# Callback to display the output based on the selected county and state
@app.callback(
    [Output('county-data-container', 'children'),
        Output('table-container', 'children'),
        Output('income-comparison-bar-chart', 'figure'),
        Output('choropleth-map', 'figure')],
    [Input('state-dropdown', 'value'),
    Input('county-dropdown', 'value'),
    Input('variable-dropdown', 'value')]
)
def display_output(state_input, county_input,variable_input):
    # Filter the dataframe to match the user input
    selected_row = df[(df['State'] == state_input) & (df['County'] == county_input)]
    
    # Check if the county exists
    if selected_row.empty:
        return html.Div([html.H3(f"No data found for {county_input}, {state_input}.")])
    
     # Get the selected row index and its features
     # Get the selected row index and its features
    index = selected_row.index[0]
    selected_features = selected_row[features].values.flatten()

    # Scale the features in the DataFrame
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[features])

    # Scale the selected row's features
    selected_scaled = scaler.transform([selected_features]).flatten()

    # Define custom weights for features
    race_weights = {
        '% Black': 50,
        '% American Indian or Alaska Native': 50,
        '% Asian': 50,
        '% Native Hawaiian or Other Pacific Islander': 50,
        '% Hispanic': 50,
        '% Non-Hispanic White': 10,
    }
    non_race_weights = {'Population': 1, '% Rural': 50}
    all_weights = np.array([race_weights.get(col, non_race_weights.get(col, 1)) for col in features])

     # Increase weight for any racial category greater than 30%
    for col in racial_features:
        if selected_row[col].values[0] > 30:
            race_weights[col] = race_weights.get(col, 100) * 5  # Increase the weight by 1.5 times

    # Compute weighted Euclidean distances
    distances = np.linalg.norm((df_scaled - selected_scaled) * all_weights, axis=1)

    # Get k nearest neighbors
    k = 150
    indices = np.argsort(distances)[:k]
    similar_counties = df.iloc[indices]

    # Filter by population percentiles
    selected_percentile = selected_row['population_percentile'].values[0]
    percentile_min = selected_percentile - 3
    percentile_max = selected_percentile + 3
    similar_counties = similar_counties[
        (similar_counties['population_percentile'] >= percentile_min) &
        (similar_counties['population_percentile'] <= percentile_max)
    ]

    # Sort by the largest combined difference
    ranked_counties = similar_counties.sort_values(by='Income', ascending=False)
    ranked_counties = ranked_counties[ranked_counties.index != index]

     # change top_10_counties to point to ranked_counties if  you want the difference rankings
    top_10_counties = ranked_counties.head(10)
    # output = top_10_counties  # No need to concatenate with selected_row
    output = top_10_counties

    # Define display columns
    display_columns = ['State', 'County', 'Population', 'Income',
                       '% Rural', '% Black', '% American Indian or Alaska Native', 
                       '% Asian', '% Native Hawaiian or Other Pacific Islander', '% Hispanic']

    # Display the output
    output[display_columns]

    # Create the table for the selected county (1 row table)
    selected_county_table = html.Div([
        html.H3(f"Selected County: {county_input}, {state_input}"),
        dash.dash_table.DataTable(
            columns=[{'name': col, 'id': col} for col in display_columns],
            data=selected_row[display_columns].to_dict('records')
        )
    ])

    # Create DataTable for similar counties
    similar_counties_table = html.Div([
        html.H3(f"Similar Counties to {county_input}, {state_input}"),
        dash.dash_table.DataTable(
            columns=[{'name': col, 'id': col} for col in display_columns],
            data=top_10_counties[display_columns].to_dict('records')
        )
    ])

    # Create the bar chart for variable comparison
    bar_fig = px.bar(
        top_10_counties,
        x='County',
        y=variable_input,
        title=f"Similar Counties' {variable_input} Comparison"
    )

    # Create the choropleth map
    choropleth_fig = px.choropleth(
        top_10_counties,
        geojson=counties,
        locations='FIPS',
        color=variable_input,
        color_continuous_scale="Viridis",
        hover_name='County',
        hover_data={'County': True, variable_input: True},
        scope="usa",
        title=f"Choropleth Map for {variable_input}"
    )

    choropleth_fig.update_geos(fitbounds="locations")

    return selected_county_table, similar_counties_table, bar_fig, choropleth_fig


if __name__ == '__main__':
    app.run_server(debug=True)
