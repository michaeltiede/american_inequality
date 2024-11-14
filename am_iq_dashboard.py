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

df = pd.read_csv("data.csv")
# List of columns to clean and convert
columns_to_clean = ['Income', 'Unemployment', 'Upward mobility', 'Life expectancy']

# Remove non-numeric characters and convert to numeric, coercing errors to NaN
df[columns_to_clean] = df[columns_to_clean].replace(r'[^0-9.]', '', regex=True)

# Convert columns to numeric
df[columns_to_clean] = df[columns_to_clean].apply(pd.to_numeric, errors='coerce')

# Fill NaN values with the mean of each column
df[columns_to_clean] = df[columns_to_clean].apply(lambda col: col.fillna(col.mean()))

# Calculate population percentiles for each county
df['population_percentile'] = df['Population'].apply(lambda x: percentileofscore(df['Population'], x))

# Define features
features = ['Population','% Rural','Percent_White','% Non-Hispanic White',
'% Less than 18 Years of Age','% 65 and Over','% Black',
'% American Indian or Alaska Native','% Asian','% Native Hawaiian or Other Pacific Islander',
'% Hispanic','% Not Proficient in English','% Female','% foreign born',
'less_than_hs_edu','hs_edu','some_college_or_assoc_edu','bachelors_or_higher_edu']

# Convert the features to float
for feature in features:
    df[feature] = pd.to_numeric(df[feature], errors='coerce').fillna(0)


# ------------------------------------------------------
# Create the Dash app
app = dash.Dash(__name__)

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
            {'label': 'Life Expectancy', 'value': 'Life expectancy'},
            {'label': 'Upward Mobility', 'value': 'Upward mobility'}
        ],
        value='Income',  # default value
        placeholder="Select a variable to compare"
    ),

    html.Br(),
    # Display the results in a table
    # Display the selected county's data
    dcc.Graph(id='income-comparison-bar-chart')  # Add graph for visualization
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
        Output('income-comparison-bar-chart', 'figure')],
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
    
    # Get the index of the selected county
    index = selected_row.index[0]
    
    # Extract the features for the selected county
    selected_features = selected_row[features].values.flatten()
    
    # Filter the dataframe for outliers (for training the model)
    df_filtered = df
    # Scale the features for the filtered DataFrame (for model training)
    scaler = StandardScaler()
    df_scaled = df_filtered[features].copy()
    df_scaled[features] = scaler.fit_transform(df_scaled[features])

    # Scale the selected county's features as well
    selected_scaled = scaler.transform([selected_features])

    # Manually adjust the weights of the features
    weights = np.array([1, 7, 7, 7, 2.5, 3, 3, 7, 3.5, 1, 2, 1, 2.5, 3, 10, 1, 1, 1])

    # Compute weighted distance
    distances = np.linalg.norm((df_scaled[features] - selected_scaled) * weights, axis=1)

    # Get the indices of the k nearest neighbors (e.g., 20 similar counties)
    k = 30
    indices = distances.argsort()[:k]

    # Filter similar counties
    similar_counties = df_filtered.iloc[indices]
    
    # Get the population percentile of the selected county
    selected_percentile = selected_row['population_percentile'].values[0]

    # Define the range of percentiles to consider (within 5 percentile points)
    percentile_min = selected_percentile - 3
    percentile_max = selected_percentile + 3
    similar_counties = similar_counties[(similar_counties['population_percentile'] >= percentile_min) & 
                                        (similar_counties['population_percentile'] <= percentile_max)]

    # Calculate differences in outcome variables
    similar_counties['Income_diff'] = similar_counties['Income'] - selected_row['Income'].values[0]
    similar_counties['Life_expectancy_diff'] = selected_row['Life expectancy'].values[0] - similar_counties['Life expectancy']
    similar_counties['Upward_mobility_diff'] = similar_counties['Upward mobility'] - selected_row['Upward mobility'].values[0]

    # Scale the differences
    similar_counties[['Income_diff', 'Life_expectancy_diff', 'Upward_mobility_diff']] = scaler.fit_transform(
        similar_counties[['Income_diff', 'Life_expectancy_diff', 'Upward_mobility_diff']]
    )

    # Rank counties by largest positive differences
    similar_counties['Combined_diff'] = (
        similar_counties['Income_diff'] + similar_counties['Upward_mobility_diff']
    )

    ranked_counties = similar_counties.sort_values(by='Combined_diff', ascending=False)
    top_10_counties = ranked_counties.head(10)
    # Concatenate the selected county with its ranked neighbors for output
    output = pd.concat([top_10_counties])
    output = output.drop(index=selected_row.index)




    # Display relevant columns
    display_columns = [
        'State', 'County', 'Population','Income', 'Life expectancy','Upward mobility'
    ]

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
        html.H3(f"Top Similar Counties to {county_input}, {state_input}"),
        dash.dash_table.DataTable(
            columns=[{'name': col, 'id': col} for col in display_columns],
            data=top_10_counties[display_columns].to_dict('records')
        )
    ])

     # Create bar chart comparing income
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=top_10_counties['County'],
        y=top_10_counties[variable_input],
        name=variable_input,
        marker=dict(color='blue')
    ))
 
    fig.update_layout(
        barmode='group',
        title=f"{variable_input} Comparison: {county_input}, {state_input} vs. Similar Counties",
        xaxis_title="County",
        yaxis_title=variable_input,
        showlegend=True
    )

    return selected_county_table ,similar_counties_table, fig

if __name__ == '__main__':
    app.run_server(debug=True)
