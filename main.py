import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from sklearn import preprocessing
from createModel import create_and_train_model
from evaluateModel import evaluate_model
from plotting import plot_training_metrics, plot_given_vs_predicted
from error import calculate_errors_all, calculate_squared_errors, r_squared_score
from app import app as dash_app
import numpy as np
import json


def main():
    # Load data from Excel files
    data = pd.read_excel('C:/Users/USER/Desktop/2 INFO 01/PFA2/coding/predictiveModelGeneral/bd.xlsx')

    # Split data into input variables (X) and target variables (Y)
    X = data[['g01', 'Esat1',  'g02', 'Esat2']]
    X[['Esat1','Esat2']] *= 10**12
    Y = data[['P0', 'T1', 'P1', 'T2', 'P2', 'T3', 'P3', 'T4', 'P4', 'T5', 'P5', 'T6']]

    # Data Preprocessing
    scaler = StandardScaler()
    norm = MinMaxScaler()
    #X = scaler.fit_transform(X)
    #X = preprocessing.normalize(X)
    #Y = preprocessing.normalize(Y)

    # Split data into training, validation, and test sets
    #X_train, temp_X, Y_train, temp_Y = train_test_split(X, Y, test_size=0.3, random_state=42)
    #X_val, X_test, Y_val, Y_test = train_test_split(temp_X, temp_Y, test_size=0.5, random_state=42)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create and train the model
    model, history, time = create_and_train_model(X_train, Y_train, 4, 200, 55)
    print('time of training : ', time)
    # Evaluate the model
    predictions, loss, mae, rmse, r_squared, adj_r_squared = evaluate_model(model, X_test, Y_test)


    # Create a dictionary with the metrics values
    metrics_values = {
        'Loss': loss,
        'MAE': mae,
        'RMSE': rmse,
        'R-squared': r_squared,
        'Adj R-squared': adj_r_squared
    }


    # Calculate squared errors for each sample independently
    squared_errors_for_samples = r_squared_score(Y_test, predictions)
    print('squared_errors: ', np.mean(squared_errors_for_samples))
    
    # Convert history dataframe to dictionary
    history_dict = history.history

    # Convert predictions to a list of dictionaries
    labels = ['PP0', 'PT1', 'PP1', 'PT2', 'PP2', 'PT3', 'PP3', 'PT4', 'PP4', 'PT5', 'PP5', 'PT6']
    predictions_list = predictions.tolist()

    predictions_dict = [{label: value for label, value in zip(labels, sublist)} for sublist in predictions_list]

    

    # Convert NumPy arrays to lists
    X_test_list = X_test.tolist()
    Y_test_list = Y_test.values.tolist()

    # Create dictionaries for X_test and Y_test data
    labels = ['g01', 'Esat1', 'g02', 'Esat2']
    X_test_dict = [{label: sample[i] for i, label in enumerate(labels)} for sample in X_test]
    labels = ['P0', 'T1', 'P1', 'T2', 'P2', 'T3', 'P3', 'T4', 'P4', 'T5', 'P5', 'T6']
    Y_test_dict = [{label: value for label, value in zip(labels, sublist)} for sublist in Y_test_list]


    # Serialize history to JSON
    history_json = json.dumps(history_dict)
    metrics_values_json = json.dumps(metrics_values)
    predictions_json = json.dumps(predictions_dict)

    # Save the JSON data to files (optional)
    with open('history.json', 'w') as f:
        json.dump(history_json, f)
    with open('metrics_values.json', 'w') as f:
        f.write(metrics_values_json)
    with open('predictions.json', 'w') as f:
        json.dump(predictions_dict, f)
    with open('X_test.json', 'w') as file:
        json.dump(X_test_dict, file)
    with open('Y_test.json', 'w') as file:
        json.dump(Y_test_dict, file)

    # Call the Dash app to display the interface
    dash_app.run_server(debug=True)


    

if __name__ == "__main__":
    main()