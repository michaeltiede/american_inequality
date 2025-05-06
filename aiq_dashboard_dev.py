# app.py

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import pickle
import json
from preprocessed_data import df, features, np  # Import preprocessed data and features


# Link Function
def make_compare_link(original_fips, compare_fips):
    original_fips = str(original_fips).zfill(5)
    compare_fips = str(compare_fips).zfill(5)
    return f"https://www.countyhealthrankings.org/health-data/compare-counties?year=2025&compareCounties={original_fips},{compare_fips}"

# Load the pre-trained scaler and weights
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('weights.pkl', 'rb') as f:
    all_weights = pickle.load(f)

# ------------------------------------------------------
# Create the Dash app
app = dash.Dash(__name__)
server = app.server  # This is necessary for gunicorn to recognize the server

# GeoJSON data for counties
with open('geojson-counties-fips.json', 'r') as file:
    counties = json.load(file)

# Layout for the dashboard
app.layout = html.Div([
    # Container for the image and main title
    html.Div([
        html.Img(src='https://raw.githubusercontent.com/michaeltiede/american_inequality/main/AIP_logo.png', style={'height': '100px', 'width': 'auto', 'marginRight': '20px'}),
        html.Div([
            html.H1(
                "The American Inequality Project: County Comparison Dashboard",
                style={'textAlign': 'center', 'marginTop': '20px', 'color': '#4A4A4A', 'fontFamily': 'Inter, sans-serif'}
            ),
        ], style={'flex': '1', 'display': 'flex', 'justifyContent': 'center'})
    ], style={'display': 'flex', 'alignItems': 'center'}),

    # Subtitle and Link (Right-Aligned Below Header)
    html.Div([
        html.A(
            "AmericanInequality.io",
            href="https://www.americaninequality.io/",
            target="_blank",
            style={'color': 'blue', 'textDecoration': 'underline', 'fontSize': '16px', 'marginRight': 'auto'}
        ),
        html.Span("Developed by Michael Tiede", style={'marginLeft': 'auto', 'fontSize': '10px'})
    ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'marginTop': '10px', 'marginLeft': '10px', 'marginRight': '10px', 'color': 'black','marginBottom': '10px'}),

    # Dropdown for selecting state and county
    dcc.Dropdown(
        id='state-dropdown',
        options=[{'label': state, 'value': state} for state in df['State'].unique()],
        value='California',  # default value
        placeholder="Select a State",
        style={'padding': '10px', 'borderRadius': '5px', 'backgroundColor': '#FFF0E1', 'color': '#4A4A4A', 'fontFamily': 'Inter, sans-serif', 'fontWeight': 'bold'}
    ),

    dcc.Dropdown(
        id='county-dropdown',
        options=[],
        value='Imperial',  # default value
        placeholder="Select a County",
        style={'padding': '10px', 'borderRadius': '5px', 'backgroundColor': '#FFF0E1', 'color': '#4A4A4A', 'fontFamily': 'Inter, sans-serif', 'fontWeight': 'bold'}
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
        style={'width': '50%', 'padding': '10px', 'borderRadius': '5px', 'backgroundColor': '#FFF0E1', 'color': '#4A4A4A', 'fontFamily': 'Inter, sans-serif', 'fontWeight': 'bold'}
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
], style={'backgroundColor': '#F7E1C4', 'padding': '20px', 'fontFamily': 'Inter, sans-serif'})

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
def display_output(state_input, county_input, variable_input):
    # Filter the dataframe to match the user input
    selected_row = df[(df['State'] == state_input) & (df['County'] == county_input)]

    # Check if the county exists
    if selected_row.empty:
        return html.Div([html.H3(f"No data found for {county_input}, {state_input}.")])

    # Get the selected row index and its features
    index = selected_row.index[0]
    selected_features = selected_row[features].values.flatten()

    # Scale the features in the DataFrame using the preloaded scaler
    df_scaled = scaler.transform(df[features])

    # Scale the selected row's features
    selected_scaled = scaler.transform([selected_features]).flatten()

    # Compute weighted Euclidean distances
    distances = np.linalg.norm((df_scaled - selected_scaled) * all_weights, axis=1)

    # Get k nearest neighbors
    k = 200
    indices = np.argsort(distances)[:k]
    similar_counties = df.iloc[indices]

    # Apply state filter to exclude counties from the same state
    similar_counties = similar_counties[similar_counties['State'] != state_input]

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

    top_10_counties = ranked_counties.head(10)

    # Generate Compare Links for each row
    original_fips = selected_row['FIPS'].values[0]

    # Create a column called "More Info" with markdown-style links
    top_10_counties['More Info'] = top_10_counties['FIPS'].apply(
    lambda compare_fips: f"[Link]({make_compare_link(original_fips, compare_fips)})"
    )

    # Define display columns
    display_columns = ['State', 'County', 'Population', 'Income',
                       '% Rural', '% Black', '% American Indian or Alaska Native', 
                       '% Asian', '% Native Hawaiian or Other Pacific Islander', '% Hispanic','More Info']

    # Display the output
    output = top_10_counties[display_columns]

    # Create the table for the selected county (1 row table)
    selected_county_table = html.Div([
        html.H3(f"Selected County: {county_input}, {state_input}"),
        dash.dash_table.DataTable(
            columns=[
                {'name': col, 'id': col, 'presentation': 'markdown'} if col == 'More Info' else {'name': col, 'id': col}
                for col in display_columns
            ],
            data=selected_row.to_dict('records'),

            style_header={
                'backgroundColor': '#FFF0E1',
                'border': '1px solid #8B4513',
                'fontFamily': 'Inter, sans-serif',
                'fontWeight': 'bold' 
            },
            style_cell={
                'backgroundColor': '#FFF0E1',
                'border': '1px solid #8B4513',
                'fontFamily': 'Inter, sans-serif'
            }
        )
    ])

    # Create DataTable for similar counties
    similar_counties_table = html.Div([
        html.H3(f"Similar Counties to {county_input}, {state_input}"),
        dash.dash_table.DataTable(
            columns=[
                {'name': col, 'id': col, 'presentation': 'markdown'} if col == 'More Info' else {'name': col, 'id': col}
             for col in display_columns  # assuming 'More Info' is in display_columns
            ],
         data=top_10_counties.assign(
                **{'More Info': top_10_counties['More Info'].apply(lambda url: f"[Link]({url})")}
            ).to_dict('records'),
            style_header={
                'backgroundColor': '#FFF0E1',
               'border': '1px solid #8B4513',
               'fontFamily': 'Inter, sans-serif',
              'fontWeight': 'bold' 
            },
            style_cell={
                'backgroundColor': '#FFF0E1',
             'border': '1px solid #8B4513',
             'fontFamily': 'Inter, sans-serif'
            }
        )
    ])


    # Create the bar chart for variable comparison
    bar_fig = px.bar(
        top_10_counties,
        x='County',
        y=variable_input,
        color=variable_input,
        color_continuous_scale="RdBu"
    )
    bar_fig.update_layout(
        plot_bgcolor='#FFF0E1',
        paper_bgcolor='#FFF0E1',
        font={'family': 'Inter, sans-serif'}
    )

    # Create the choropleth map
    choropleth_fig = go.Figure(go.Choropleth(
        z=top_10_counties[variable_input],
        hoverinfo='location+z',
        locationmode="USA-states",
        locations=top_10_counties['FIPS'],
        colorscale="RdBu",
        colorbar_title=variable_input
    ))
    choropleth_fig.update_layout(
        geo=dict(showcoastlines=True, coastlinecolor="Black"),
        plot_bgcolor='#FFF0E1',
        paper_bgcolor='#FFF0E1',
        font={'family': 'Inter, sans-serif'}
    )

    return selected_county_table, similar_counties_table, bar_fig, choropleth_fig


if __name__ == '__main__':
    app.run(debug=True)
