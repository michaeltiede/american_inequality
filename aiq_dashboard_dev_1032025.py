# app.py

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash.dash_table
import plotly.express as px
import pandas as pd
import pickle
import numpy as np
import json
from preprocessed_data import df, features, sorted_states, sorted_counties, racial_features, get_pool_and_scaled

# -------------------------------
# Helper Function
# -------------------------------
def make_compare_link(original_fips, compare_fips):
    original_fips = str(original_fips).zfill(5)
    compare_fips = str(compare_fips).zfill(5)
    return f"https://www.countyhealthrankings.org/health-data/compare-counties?year=2025&compareCounties={original_fips},{compare_fips}"

# -------------------------------
# Load scaler and weights
# -------------------------------
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('weights.pkl', 'rb') as f:
    all_weights = pickle.load(f)

# -------------------------------
# Create Dash App
# -------------------------------
app = dash.Dash(__name__)
server = app.server  # For deployment

# Load GeoJSON
with open('geojson-counties-fips.json', 'r') as file:
    counties = json.load(file)

# -------------------------------
# Layout
# -------------------------------
app.layout = html.Div([
    # Header
    html.Div([
        html.Img(src='https://raw.githubusercontent.com/michaeltiede/american_inequality/main/AIP_logo.png',
                 style={'height': '100px', 'width': 'auto', 'marginRight': '20px'}),
        html.Div([
            html.H1("The American Inequality Project: County Comparison Dashboard",
                    style={'textAlign': 'center', 'marginTop': '20px', 'color': '#000000',
                           'fontFamily': 'Inter, sans-serif'})
        ], style={'flex': '1', 'display': 'flex', 'justifyContent': 'center'})
    ], style={'display': 'flex', 'alignItems': 'center'}),

    # Subtitle
    html.Div([
        html.A("AmericanInequality.substack.com",
               href="https://www.americaninequality.substack.com/",
               target="_blank",
               style={'color': 'blue', 'textDecoration': 'underline', 'fontSize': '16px', 'marginRight': 'auto'}),
        html.Span("Developed by Michael Tiede", style={'marginLeft': 'auto', 'fontSize': '10px'})
    ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center',
              'margin': '10px', 'color': 'black'}),

    # State and County dropdowns
    dcc.Dropdown(
        id='state-dropdown',
        options=[{'label': state, 'value': state} for state in sorted_states],
        value='Alabama',
        placeholder="Select a State",
        style={'padding': '10px', 'borderRadius': '5px', 'backgroundColor': '#FFF0E1',
               'color': '#4A4A4A', 'fontFamily': 'Inter, sans-serif', 'fontWeight': 'bold'}
    ),
    dcc.Dropdown(
        id='county-dropdown',
        options=[{'label': county, 'value': county} for county in sorted_counties],
        placeholder="Select a County",
        style={'padding': '10px', 'borderRadius': '5px', 'backgroundColor': '#FFF0E1',
               'color': '#4A4A4A', 'fontFamily': 'Inter, sans-serif', 'fontWeight': 'bold'}
    ),
    html.Br(),

    html.Div(id='county-data-container'),
    html.Div(id='table-container'),
    html.Br(),

    # Variable dropdown
    dcc.Dropdown(
        id='variable-dropdown',
        options=[
            {'label': 'Income', 'value': 'Income'},
            {'label': 'Life Expectancy', 'value': 'Life Expectancy'},
            {'label': 'Upward Mobility', 'value': 'Upward mobility'}
        ],
        value='Income',
        placeholder="Select a variable to compare",
        style={'width': '50%', 'padding': '10px', 'borderRadius': '5px',
               'backgroundColor': '#FFF0E1', 'color': '#4A4A4A',
               'fontFamily': 'Inter, sans-serif', 'fontWeight': 'bold'}
    ),
    html.Br(),

    # Bar chart and choropleth map
    html.Div([
        html.Div([dcc.Graph(id='income-comparison-bar-chart')],
                 style={'width': '48%', 'display': 'inline-block'}),
        html.Div([dcc.Graph(id='choropleth-map')],
                 style={'width': '48%', 'display': 'inline-block', 'padding-left': '2%'})
    ], style={'display': 'flex', 'flex-direction': 'row'})
], style={'backgroundColor': '#FFFFFF', 'padding': '20px', 'fontFamily': 'Inter, sans-serif'})

# -------------------------------
# Callbacks
# -------------------------------
@app.callback(
    Output('county-dropdown', 'options'),
    Output('county-dropdown', 'value'),
    Input('state-dropdown', 'value')
)
def update_county_dropdown(state):
    counties = df[df['State'] == state]['County'].unique()
    return [{'label': c, 'value': c} for c in counties], counties[0]


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
    # Filter selected county
    selected_row = df[(df['State'] == state_input) & (df['County'] == county_input)]
    if selected_row.empty:
        return html.Div([html.H3(f"No data found for {county_input}, {state_input}.")]), None, None, None

    index = selected_row.index[0]

    # -------------------------------
    # Scale features properly (DataFrame)
    # -------------------------------
    selected_scaled = scaler.transform(selected_row[features]).flatten()
    df_scaled = scaler.transform(df[features])

    # -------------------------------
    # Get weighted pool
    # -------------------------------
    population_value = selected_row['Population'].values[0]
    df_pool, df_scaled_pool, weights = get_pool_and_scaled(
        population_value,
        df,
        scaler,
        features,
        racial_features,
        all_weights
    )

    # Compute distances
    distances = np.linalg.norm((df_scaled_pool - selected_scaled) * weights, axis=1)

    # Get k nearest neighbors
    k = 200
    indices = np.argsort(distances)[:k]
    similar_counties = df_pool.iloc[indices]
    similar_counties = similar_counties[similar_counties['State'] != state_input]

    # Apply percentile filter for small counties
    if population_value <= 700_000:
        selected_percentile = selected_row['population_percentile'].values[0]
        percentile_min = selected_percentile - 3
        percentile_max = selected_percentile + 3
        similar_counties = similar_counties[
            (similar_counties['population_percentile'] >= percentile_min) &
            (similar_counties['population_percentile'] <= percentile_max)
        ]

    # Rank by variable
    ranked_counties = similar_counties.sort_values(by='Income', ascending=False)
    ranked_counties = ranked_counties[ranked_counties.index != index]
    top_10_counties = ranked_counties.head(10)

    # Add links
    original_fips = selected_row['FIPS'].values[0]
    top_10_counties['More Info'] = top_10_counties['FIPS'].apply(
        lambda fips: f"[Link]({make_compare_link(original_fips, fips)})"
    )

    # Display columns
    display_columns = ['State', 'County', 'Population', 'Income','Top 1 Industry String','Top 2 Industry String',
                       '% Rural', '% Black', '% American Indian or Alaska Native',
                       '% Asian', '% Native Hawaiian or Other Pacific Islander', '% Hispanic', 'More Info']

    # Selected county table
    selected_county_table = html.Div([
        html.H3(f"Selected County: {county_input}, {state_input}"),
        dash.dash_table.DataTable(
            columns=[{'name': col, 'id': col, 'presentation': 'markdown'} if col == 'More Info'
                     else {'name': col, 'id': col} for col in display_columns],
            data=selected_row.to_dict('records'),
            style_header={'backgroundColor': '#FFF0E1', 'border': '1px solid #8B4513',
                          'borderRadius': '5px', 'fontFamily': 'Inter, sans-serif', 'fontWeight': 'bold'},
            style_cell={'backgroundColor': '#FFF0E1', 'border': '1px solid #8B4513',
                        'borderRadius': '5px', 'fontFamily': 'Inter, sans-serif'}
        )
    ])

    # Similar counties table
    similar_counties_table = html.Div([
        html.H3(f"Similar Counties to {county_input}, {state_input}"),
        dash.dash_table.DataTable(
            columns=[{'name': col, 'id': col, 'presentation': 'markdown'} if col == 'More Info'
                     else {'name': col, 'id': col} for col in display_columns],
            data=top_10_counties.to_dict('records'),
            style_header={'backgroundColor': '#FFF0E1', 'border': '1px solid #8B4513',
                          'borderRadius': '5px', 'fontFamily': 'Inter, sans-serif', 'fontWeight': 'bold'},
            style_cell={'backgroundColor': '#FFF0E1', 'border': '1px solid #8B4513',
                        'fontFamily': 'Inter, sans-serif'}
        )
    ])

    # Bar chart
    bar_fig = px.bar(top_10_counties, x='County', y=variable_input, color=variable_input, color_continuous_scale="RdBu")
    bar_fig.update_layout(plot_bgcolor='#FFF0E1', paper_bgcolor='#FFF0E1',
                          title={'text': f"{variable_input} Compared to Similar Counties",
                                 'font': {'size': 20, 'color': '#4A4A4A', 'family': 'Inter, sans-serif', 'weight': 'bold'}})

    # Choropleth map
    choropleth_fig = px.choropleth(top_10_counties, geojson=counties, locations='FIPS', color=variable_input,
                                   color_continuous_scale="RdBu", hover_name='County', hover_data={'County': True, variable_input: True},
                                   scope="usa")
    choropleth_fig.update_layout(plot_bgcolor='#FFF0E1', paper_bgcolor='#FFF0E1',
                                 title={'text': f"Choropleth Map for {variable_input}",
                                        'font': {'size': 20, 'color': '#4A4A4A', 'family': 'Inter, sans-serif', 'weight': 'bold'}})
    choropleth_fig.update_geos(fitbounds="locations")

    return selected_county_table, similar_counties_table, bar_fig, choropleth_fig

# -------------------------------
# Run the App
# -------------------------------
if __name__ == '__main__':
    app.run(debug=True)
