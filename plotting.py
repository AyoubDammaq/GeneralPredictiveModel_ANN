import matplotlib.pyplot as plt
import random
import plotly.graph_objs as go
from dash import html
from error import r_squared_score
from sklearn.metrics import r2_score



def plot_training_metrics(history, metrics_values):
    # Create MAE graph
    mae_fig = go.Figure()
    mae_fig.add_trace(go.Scatter(x=history.index, y=history['mae'], mode='lines', name='Train'))
    mae_fig.add_trace(go.Scatter(x=history.index, y=history['val_mae'], mode='lines', name='Validation'))
    mae_fig.update_layout(title="Evolution of Mean Absolute Error (MAE) during Training",
                          xaxis_title='Epoch',
                          yaxis_title='MAE',
                          showlegend=True,
                          legend=dict(x=0, y=1),
                          xaxis=dict(showgrid=True),  
                          yaxis=dict(showgrid=True))

    # Create MSE graph
    mse_fig = go.Figure()
    mse_fig.add_trace(go.Scatter(x=history.index, y=history['loss'], mode='lines', name='Train'))
    mse_fig.add_trace(go.Scatter(x=history.index, y=history['val_loss'], mode='lines', name='Validation'))
    mse_fig.update_layout(title='Evolution of Mean Squared Error (MSE) during Training',
                          xaxis_title='Epoch',
                          yaxis_title='MSE',
                          showlegend=True,
                          legend=dict(x=0, y=1),
                          xaxis=dict(showgrid=True),  
                          yaxis=dict(showgrid=True))

    # Create all metrics graph
    all_metrics_fig = go.Figure()
    all_metrics_fig.add_trace(go.Scatter(x=history.index, y=history['mae'], mode='lines', name='MAE'))
    all_metrics_fig.add_trace(go.Scatter(x=history.index, y=history['mse'], mode='lines', name='MSE'))
    all_metrics_fig.update_layout(title='All Metrics',
                                  xaxis_title='Epochs',
                                  yaxis_title='Value',
                                  showlegend=True,
                                  legend=dict(x=0, y=1),
                                  xaxis=dict(showgrid=True),  
                                  yaxis=dict(showgrid=True))

    # Define a custom CSS style for the table
    table_style = {
        'font-family': 'Arial, sans-serif',
        'border-collapse': 'collapse',
        'width': '100%',
    }

    header_cell_style = {
        'border': '1px solid #dddddd',
        'text-align': 'center',
        'padding': '8px',
        'background-color': '#f2f2f2',
        'color': '#333333',
    }

    data_cell_style = {
        'border': '1px solid #dddddd',
        'text-align': 'center',
        'padding': '8px',
    }

    # Create the table
    metrics_table = html.Table(
        style=table_style,
        children=[
            html.Thead(
                html.Tr([
                    html.Th(col, style=header_cell_style) for col in metrics_values.columns
                ])
            ),
            html.Tbody([
                html.Tr([
                    html.Td(f'{metrics_values[col][0]:.2f}', style=data_cell_style)
                    for col in metrics_values.columns
                ])
            ])
        ]
    )

    return mae_fig, mse_fig, all_metrics_fig, metrics_table


def plot_given_vs_predicted(X_test, Y_test, predictions, num_samples_to_plot):
    # Calculate the number of rows and columns for the subplot grid
    num_rows = (num_samples_to_plot + 1) // 2
    num_columns = 2

    # Create subplots
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_columns, figsize=(12, 10))

    # Flatten the axes for easier indexing
    axes = axes.flatten()

    # Initialize a list to store the Plotly traces
    plot_traces = []

    # Loop through the number of samples
    for i in range(num_samples_to_plot):
        # Choose a random index for validation
        validation_index = random.randint(0, len(Y_test) - 1)

        # Data for the specific case
        donnees_du_cas = Y_test.iloc[validation_index]

        # Assign each value to a separate variable
        P0, T1, P1, T2, P2, T3, P3, T4, P4, T5, P5, T6 = donnees_du_cas

        # Extracted points
        points = [
            (0, -T6), (P5, -T5), (P4, -T4), (P3, -T3),
            (P2, -T2), (P1, -T1), (P0, 0), (P1, T1),
            (P2, T2), (P3, T3), (P4, T4), (P5, T5), (0, T6)
        ]

        # Separate x and y coordinates
        x_values, y_values = zip(*points)

        # Predicted values for the specific case
        predicted_values = predictions.loc[validation_index]

        PP0, PT1, PP1, PT2, PP2, PT3, PP3, PT4, PP4, PT5, PP5, PT6 = predicted_values

        ppoints = [
            (0, -PT6), (PP5, -PT5), (PP4, -PT4), (PP3, -PT3),
            (PP2, -PT2), (PP1, -PT1), (PP0, 0), (PP1, PT1),
            (PP2, PT2), (PP3, PT3), (PP4, PT4), (PP5, PT5), (0, PT6)
        ]

        # Separate x and y coordinates for predicted values
        px_values, py_values = zip(*ppoints)

        # Calculate the error for the specific case
        r_squared_score_calcul = r2_score(donnees_du_cas, predicted_values)

        # Plot the given points and predicted values
        axes[i].plot(y_values, x_values, marker='', linestyle='-', label='Given Points')
        axes[i].plot(py_values, px_values, marker='', linestyle='--', label='Predicted Values')

        # Add labels and title
        axes[i].set_xlabel('Time[ps]')
        axes[i].set_ylabel('Power[mW]')
        axes[i].set_title(f'Sample {i+1} - R-squared score: {r_squared_score_calcul:.4f}')

        # Display legend
        axes[i].legend()

        # Add text annotations for the values of g0 and Esat
        g01_value = X_test.iloc[validation_index]['g01']
        Esat1_value = X_test.iloc[validation_index]['Esat1']
        g02_value = X_test.iloc[validation_index]['g02']
        Esat2_value = X_test.iloc[validation_index]['Esat2']
        axes[i].text(0.5, 0.5, f'g01={g01_value}, Esat1={Esat1_value}, g02={g02_value}, Esat2={Esat2_value}',
                     transform=axes[i].transAxes, ha='center', va='center', fontsize=10, color='blue')

        # Convert the Matplotlib plot to a Plotly trace
        plotly_trace = go.Scatter(
            x=y_values, y=x_values,
            mode='lines', name='Given Points',
            line=dict(color='blue', dash='solid')
        )
        plot_traces.append(plotly_trace)

        plotly_pred_trace = go.Scatter(
            x=py_values, y=px_values,
            mode='lines', name='Predicted Values',
            line=dict(color='red', dash='dash')
        )
        plot_traces.append(plotly_pred_trace)

    # Adjust layout
    plt.tight_layout()

    # Convert Matplotlib figure to Plotly figure
    fig = go.Figure(data=plot_traces)

    return fig