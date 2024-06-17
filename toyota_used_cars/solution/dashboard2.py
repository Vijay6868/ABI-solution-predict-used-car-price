import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import joblib
import numpy as np

# Load the data
results = pd.read_csv('predicted_prices.csv')
dummies_columns = joblib.load('dummies_columns.pkl')
model = joblib.load('random_forest_model.pkl')
current_year = 2024

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# App layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Car Price Prediction Dashboard and Portal"), className="mb-2")
    ]),
    dbc.Row([
        dbc.Col(html.Div([
            dcc.Dropdown(id='car-model-dropdown', 
                         options=[{'label': model, 'value': model} for model in dummies_columns],
                         value=dummies_columns[0]),
            dcc.Input(id='car-year', type='number', placeholder='Enter Year', min=1980, max=current_year),
            dcc.Input(id='car-mileage', type='number', placeholder='Enter Mileage'),
            html.Button('Predict Price', id='predict-button', n_clicks=0),
            html.Div(id='prediction-output')
        ]), width=6)
    ]),
    dbc.Row([
        dbc.Col(html.Div([
            dcc.Graph(id='price-distribution-graph')
        ]), width=12)
    ]),
    dbc.Row([
        dbc.Col(html.H4(f"Car Inventory Summary"), className="mt-4")
    ]),
    dbc.Row([
        dbc.Col(html.Div(id='car-inventory-summary'), width=12)
    ])
])

# Callback to update prediction output
@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    State('car-model-dropdown', 'value'),
    State('car-year', 'value'),
    State('car-mileage', 'value')
)
def update_prediction(n_clicks, selected_model, year, mileage):
    if n_clicks > 0 and year is not None and mileage is not None:
        car_age = current_year - year
        input_data = [0] * len(dummies_columns)
        if selected_model in dummies_columns:
            input_data[dummies_columns.index(selected_model)] = 1
        input_data = [car_age, mileage] + input_data
        
        # Make prediction
        predicted_price = model.predict([input_data])[0]
        
        return f"The predicted price for the {selected_model} is ${predicted_price:.2f}"
    return ""

# Callback to update graph and car inventory summary based on selected car model
@app.callback(
    [Output('price-distribution-graph', 'figure'),
     Output('car-inventory-summary', 'children')],
    [Input('car-model-dropdown', 'value')]
)
def update_graph_and_summary(selected_model):
    filtered_df = results[results[selected_model] == 1]

    if filtered_df.empty:
        return {}, html.P(f"No data available for model {selected_model}")

    # Price distribution graph
    price_distribution_fig = px.histogram(filtered_df, x='Predicted Price', title=f'Price Distribution for {selected_model}')
    
    # Car inventory summary
    inventory_summary = html.Ul([
        html.Li(f"Total Cars: {len(filtered_df)}"),
        html.Li(f"Average Predicted Price: ${filtered_df['Predicted Price'].mean():.2f}"),
        html.Li(f"Average Car Age: {filtered_df['Car Age'].mean():.2f} years"),
        html.Li(f"Average Mileage: {filtered_df['Mileage'].mean():.2f} miles")
    ])
    
    return price_distribution_fig, inventory_summary

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
