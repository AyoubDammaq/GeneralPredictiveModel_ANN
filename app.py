import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import json
from plotting import plot_given_vs_predicted, plot_training_metrics

# Initialize the Dash application
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Load JSON data from files or other sources 
try:
    with open('history.json', 'r') as file:
        history_str = json.load(file)
        history_json = json.loads(history_str)
except FileNotFoundError:
    history_str = None
    history_json = None

def load_json_file(file_path):
    try:
        with open(file_path, 'r') as file:
            file_str = file.read()
            file_json = json.loads(file_str)
            return file_json
    except FileNotFoundError:
        return None

metrics_values_json = load_json_file('metrics_values.json')
X_test_json = load_json_file('X_test.json')
Y_test_json = load_json_file('Y_test.json')
predictions_json = load_json_file('predictions.json')





# Create the layout of the application
app.layout = html.Div([
    html.H1("Training Metrics"),
    dcc.Store(id='history', data=history_json),
    dcc.Store(id='metrics-values', data=metrics_values_json),
    dcc.Graph(id='mae-graph'),
    dcc.Graph(id='mse-graph'),
    dcc.Graph(id='all-metrics-graph'),
    html.Table(id='metrics-table'),
    html.H1("Plots from plot given vs predicted"),
    dcc.Graph(id='given-predicted-plot'),
])

# Define the logic of the application
@app.callback(
    [Output('mae-graph', 'figure'),
     Output('mse-graph', 'figure'),
     Output('all-metrics-graph', 'figure'),
     Output('metrics-table', 'children'),
     Output('given-predicted-plot', 'figure')], 
    [Input('history', 'data'), Input('metrics-values', 'data')]
)
def update_metrics_plots(history_json, metrics_values_json):

    if history_json is None or metrics_values_json is None or X_test_json is None or Y_test_json is None or predictions_json is None:
        raise dash.exceptions.PreventUpdate("JSON data not loaded or empty")

    try:
        # Convert JSON data to pandas dataframes
        history = pd.DataFrame(history_json)
        metrics_values = pd.DataFrame.from_dict(metrics_values_json, orient='index').T
        X_test = pd.DataFrame(X_test_json)
        Y_test = pd.DataFrame(Y_test_json)  
        predictions = pd.DataFrame(predictions_json) 
    except Exception as e:
        raise dash.exceptions.PreventUpdate(f"Error loading JSON data: {str(e)}")
    
    mae_fig, mse_fig, all_metrics_fig, metrics_table = plot_training_metrics(history, metrics_values)

    fig = plot_given_vs_predicted(X_test, Y_test, predictions, 1)

    return mae_fig, mse_fig, all_metrics_fig, metrics_table, fig



# Run the application
if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
