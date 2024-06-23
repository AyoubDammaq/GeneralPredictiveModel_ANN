from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def evaluate_model(model, X_test_scaled, Y_test):
    # Model Evaluation
    predictions = model.predict(X_test_scaled)

    # Calculate evaluation metrics
    loss = model.evaluate(X_test_scaled, Y_test)[0]
    mae = mean_absolute_error(Y_test, predictions)
    rmse = np.sqrt(loss)
    r2 = r2_score(Y_test, predictions)

    # Adjusted R-squared calculation (if necessary)
    n = len(Y_test)
    p = X_test_scaled.shape[1]  # number of explanatory variables
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    # Display results
    print(f'Loss: {loss}')
    print(f'MAE: {mae}')
    print(f'RMSE: {rmse}')
    print(f'R-squared: {r2}')
    print(f'Adj R-squared: {adj_r2}')

    return predictions, loss, mae, rmse, r2, adj_r2