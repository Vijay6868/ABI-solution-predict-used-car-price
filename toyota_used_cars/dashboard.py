import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px

# Load the data
results = pd.read_csv('predicted_prices.csv')

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# App layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Car Price Prediction Dashboard"), className="mb-2")
    ]),
    dbc.Row([
        dbc.Col(html.Div([
            dcc.Dropdown(id='car-model-dropdown', 
                         options=[{'label': model, 'value': model} for model in results.columns if model not in ['Predicted Price', 'Actual Price', 'Car Age']],
                         value=results.columns[0]),
            dcc.Graph(id='price-prediction-graph')
        ]), width=6),
        dbc.Col(html.Div([
            dcc.Graph(id='age-price-scatter-plot')
        ]), width=6)
    ]),
    dbc.Row([
        dbc.Col(html.Div([
            dcc.Graph(id='prediction-vs-actual-graph')
        ]), width=12)
    ])
])

# Callback to update graphs based on selected car model
@app.callback(
    Output('price-prediction-graph', 'figure'),
    Input('car-model-dropdown', 'value')
)
def update_price_prediction_graph(selected_model):
    filtered_df = results[results[selected_model] == 1]
    fig = px.bar(filtered_df, x=filtered_df.index, y='Predicted Price', title=f'Predicted Prices for {selected_model}')
    return fig

# Callback to update age vs price scatter plot
@app.callback(
    Output('age-price-scatter-plot', 'figure'),
    Input('car-model-dropdown', 'value')
)
def update_age_price_scatter_plot(selected_model):
    filtered_df = results[results[selected_model] == 1]
    fig = px.scatter(filtered_df, x='Car Age', y='Predicted Price', title=f'Car Age vs Predicted Price for {selected_model}')
    return fig

# Callback to update prediction vs actual price graph
@app.callback(
    Output('prediction-vs-actual-graph', 'figure'),
    Input('car-model-dropdown', 'value')
)
def update_prediction_vs_actual_graph(selected_model):
    filtered_df = results[results[selected_model] == 1]
    fig = px.scatter(filtered_df, x='Actual Price', y='Predicted Price', title=f'Actual vs Predicted Prices for {selected_model}')
    fig.add_shape(
        type="line", line=dict(dash="dash"), 
        x0=min(filtered_df['Actual Price']), y0=min(filtered_df['Actual Price']), 
        x1=max(filtered_df['Actual Price']), y1=max(filtered_df['Actual Price'])
    )
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
