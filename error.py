import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.optimizers import Adam
import random
from tensorflow.keras.optimizers import schedules
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout


# Fonction pour calculer les erreurs pour une seule prédiction
def calculate_errors_single(true_values, predicted_values):
    mae = mean_absolute_error(true_values, predicted_values)
    mse = mean_squared_error(true_values, predicted_values)
    rmse = np.sqrt(mse)
    r2 = r2_score(true_values, predicted_values)
    return mae, rmse, r2

# Fonction pour calculer les erreurs pour toutes les prédictions
def calculate_errors_all(true_data, predicted_data):
    errors = []
    for i in range(len(true_data)):
        true_values = true_data.iloc[i]
        predicted_values = predicted_data[i]
        errors.append(calculate_errors_single(true_values, predicted_values))
    return errors

def calculate_squared_errors(Y_actual, predictions):
    # Calculate squared errors for each output variable
    squared_errors = (Y_actual - predictions)**2

    # Sum of squared errors for each sample
    sum_squared_errors = squared_errors.sum(axis=1)

    # Sum of squared values of the actual values for each sample
    sum_squared_actual_values = (Y_actual**2).sum(axis=1)

    # Calculate the squared errors for each sample independently
    squared_errors_for_samples = sum_squared_errors / sum_squared_actual_values

    return squared_errors_for_samples


def r_squared_score(actual, predicted):
    """
    Calculate the R-squared score.
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    ss_total = np.sum((actual) ** 2, axis=0)
    ss_residual = np.sum((actual - predicted) ** 2, axis=0)
    r2 = 1 - (ss_residual / ss_total)
    return r2