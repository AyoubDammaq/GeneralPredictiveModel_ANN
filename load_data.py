import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(x_path, y_path, test_size=0.3, random_state=42):
    # Load data from Excel files
    datax = pd.read_excel(x_path)
    datay = pd.read_excel(y_path)

    # Split data into input variables (X) and target variables (Y)
    X = datax[['g0', 'Esat']]
    Y = datay[['P0', 'T1', 'P1', 'T2', 'P2', 'T3', 'P3', 'T4', 'P4', 'T5', 'P5', 'T6']]

    # Split data into training, validation, and test sets
    X_train, temp_X, Y_train, temp_Y = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    X_val, X_test, Y_val, Y_test = train_test_split(temp_X, temp_Y, test_size=0.5, random_state=random_state)

    # Data Preprocessing
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, Y_train, X_val_scaled, Y_val, X_test_scaled, Y_test



