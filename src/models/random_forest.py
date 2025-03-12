import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from .feature import create_features

def random_forest(df, year, ccy, model_output_filepath):

    df["datetime"] = pd.to_datetime(df["datetime"])
    df['datetime_original'] = df['datetime']  # Add a backup column

    # Step 2: Feature Engineering
    # Create lagged features
    lags = [1, 2, 3]
    for lag in lags:
        df[f'lag_{lag}'] = df['price'].shift(lag)

    # Include `ticks_moved` as a feature
    df['ticks_moved_lag_1'] = df['ticks_moved'].shift(1)  # Add lagged `ticks_moved`

    # Fill missing values (introduced by lags)
    df.dropna(inplace=True)

    # Step 3: Define Features and Target
    features = [col for col in df.columns if col.startswith('lag') or 'ticks_moved' in col]
    df['target'] = (df['directions'] > 0).astype(int)  # Binary target: 1 for up, 0 for down
    """"
    The purpose of this line is to create a binary target variable that represents two possible outcomes for each row in the data:

    1 (Up): If the value in the directions column is positive, implying the price or movement is upward.
    0 (Down): If the value in the directions column is zero or negative, implying the price or movement is flat or downward.
    This target column will then be used as the dependent variable (label) for a machine learning model, such as a Random Forest Classifier, to predict future price movements.

    1. Expression: df['directions'] > 0
    This part checks whether the value in the column directions is greater than 0 for each row in the DataFrame.
    The result of this operation is a Boolean Series (True or False) where:
    True if the value of directions is greater than 0.
    False if the value of directions is less than or equal to 0.
    2. Method: .astype(int)
    The .astype(int) method converts the Boolean values (True and False) into integers:
    True becomes 1
    False becomes 0
    This conversion is done because most machine learning algorithms work with numerical data, and a binary classification target is typically represented as 0 or 1.
    3. Assignment: df['target'] = ...
    The result of the Boolean expression (df['directions'] > 0) and its conversion to integers (astype(int)) is assigned to a new column named target in the DataFrame df.

    """
    X = df[features]
    y = df['target']

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-Test Split: Split the data into training and testing sets -> 80:20 split
    # X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, shuffle=False)

    # Step 4: Define Ratios for Splits
    train_ratio = 0.6  # 60% for training
    test_ratio = 0.2   # 20% for testing
    backtest_ratio = 0.2  # 20% for backtesting

    # Step 5: Calculate Split Indices
    total_rows = len(X_scaled)

    # Calculate the end index for training, testing, and backtesting splits
    train_end = int(total_rows * train_ratio)  # End of training data
    test_end = train_end + int(total_rows * test_ratio)  # End of testing data (start of backtesting)

    # Ensure backtesting starts immediately after testing
    backtest_start = test_end

    # Split the data
    X_train, y_train = X_scaled[:train_end], y[:train_end]
    X_test, y_test = X_scaled[train_end:test_end], y[train_end:test_end]
    X_backtest, y_backtest = X_scaled[backtest_start:], y[backtest_start:]

    # Step 5: Train Random Forest Classifier
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)

    """""
    1. RandomForestClassifier
    The RandomForestClassifier is a machine learning model provided by the scikit-learn library. It is used for classification tasks, and it is based on the Random Forest algorithm. The Random Forest algorithm works by:

    Creating multiple decision trees during the training phase.
    Combining their outputs (e.g., majority voting) to make the final prediction.
    This ensemble method reduces overfitting and increases accuracy compared to a single decision tree.

    2. random_state=42
    Purpose: The random_state parameter is used to control the randomness in the model's training process.
    Why Set It?:
    The Random Forest model has random components (e.g., choosing subsets of data and features for building decision trees).
    By setting random_state=42, you make the results reproducible—i.e., running the code multiple times will produce the same model and predictions. (The choice of 42 is arbitrary; any integer can be used.)
    If random_state is not specified, the results may vary between runs due to randomness.

    3. Initialization Only
    At this step, the RandomForestClassifier model is only being initialized. The model parameters (e.g., number of trees, max depth, etc.) are being set to their default values, except for the random_state.
    The model has not yet been trained or fitted to the data. Training happens later when the fit method is called (e.g., rf_model.fit(X_train, y_train)).

    """""

    # Step 6: Evaluate Model
    y_pred_test = rf_model.predict(X_test)
    y_pred_backtest = rf_model.predict(X_backtest)

    accuracy = accuracy_score(y_test, y_pred_test)
    conf_matrix = confusion_matrix(y_test, y_pred_test)
    classification_report_text = classification_report(y_test, y_pred_test)

    # Display Results
    results = {
        "Accuracy": accuracy,
        "Confusion Matrix": conf_matrix,
        "Classification Report": classification_report_text
    }

    # Show data used for training and testing
    # Add a column to retain the original tick numbers and timestamps
    df['tick_number'] = df.index  # Preserve the original row index as tick_number

    # Retrieve indices for training, testing, and backtesting data
    train_indices = df.iloc[:train_end].index
    test_indices = df.iloc[train_end:test_end].index
    backtest_indices = df.iloc[test_end:].index

    # Extract training, testing, and backtesting data for display
    training_data = df.loc[train_indices, ['tick_number', 'datetime']]
    testing_data = df.loc[test_indices, ['tick_number', 'datetime']]
    backtesting_data = df.loc[backtest_indices]

    # Print training, testing, and backtesting data samples
    print("Training Data (Tick Numbers and Timestamps):")
    print(training_data.head())  # Display first 5 rows used for training

    print("\nTesting Data (Tick Numbers and Timestamps):")
    print(testing_data.head())  # Display first 5 rows used for testing

    print("\nBacktesting Data (Tick Numbers and Timestamps):")
    print(backtesting_data.head())  # Display first 5 rows used for backtesting


    # Results Summary
    print("\nModel Evaluation Results:")
    print(results)

    # Convert the results dictionary into a pandas DataFrame for saving to CSV
    results_df = pd.DataFrame.from_dict(
        {key: [value] if not isinstance(value, list) else value for key, value in results.items()}
    )

    # Save the results DataFrame to CSV
    results_df.to_csv(rf'{model_output_filepath}/{year}_{ccy}_all_random_forest_results.csv', index=False)
    df.to_csv(rf'{model_output_filepath}/{year}_{ccy}_all_random_forest_dataframe.csv')
    print(f"Model output saved to {model_output_filepath}")
    
    return backtesting_data, X_backtest, y_backtest, y_pred_backtest

def random_forest_session(df, year, ccy, session, model_output_filepath):
    # 1) Basic date/time preparation
    df['datetime'] = pd.to_datetime(df['datetime'], errors='raise')
    df['datetime_original'] = df['datetime']

    # 2) Call your new feature-engineering function
    df, features = create_features(df)

    # 3) Define your target
    df['target'] = (df['directions'].shift(-1) > 0).astype(int)

    # 4) Remove timezone
    df['datetime'] = df['datetime'].apply(lambda x: x.replace(tzinfo=None))

    # 5) Set 'datetime' as index
    df.set_index('datetime', inplace=True)
    df.index = pd.to_datetime(df.index, errors='raise')

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("The DataFrame index is not a DatetimeIndex.")

    # 6) Split by date
    df['date'] = df.index.date
    unique_dates = df['date'].unique()
    print(f"Unique session days: {len(unique_dates)}")

    train_days = int(len(unique_dates) * 0.6)
    test_days = int(len(unique_dates) * 0.2)
    backtest_days = len(unique_dates) - (train_days + test_days)

    train_dates = unique_dates[:train_days]
    test_dates = unique_dates[train_days:train_days + test_days]
    backtest_dates = unique_dates[train_days + test_days:]

    # Filter data
    train_data = df[df['date'].isin(train_dates)]
    test_data = df[df['date'].isin(test_dates)]
    backtest_data = df[df['date'].isin(backtest_dates)]

    # 7) Build X, y for each set
    X_train = train_data[features]
    y_train = train_data['target']
    X_test = test_data[features]
    y_test = test_data['target']
    X_backtest = backtest_data[features]
    y_backtest = backtest_data['target']

    # 8) Normalize features (fit on train, apply to others)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_backtest_scaled = scaler.transform(X_backtest)

    # 9) Train & Evaluate
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train_scaled, y_train)

    y_pred_test = rf_model.predict(X_test_scaled)
    y_pred_backtest = rf_model.predict(X_backtest_scaled)

    accuracy = accuracy_score(y_test, y_pred_test)
    conf_matrix = confusion_matrix(y_test, y_pred_test)
    classification_report_text = classification_report(y_test, y_pred_test)

    results = {
        "Accuracy": accuracy,
        "Confusion Matrix": conf_matrix,
        "Classification Report": classification_report_text
    }
    print(results)

    # 10) Save results
    results_df = pd.DataFrame.from_dict(
        {key: [value] if not isinstance(value, list) else value for key, value in results.items()}
    )
    results_df.to_csv(rf'{model_output_filepath}/{year}_{ccy}_{session}_random_forest_results.csv', index=False)
    df.to_csv(rf'{model_output_filepath}/{year}_{ccy}_{session}_random_forest_dataframe.csv')
    print(f"Model output saved to {model_output_filepath}")
    
    print(backtest_data.columns)

    return backtest_data, X_backtest, y_backtest, y_pred_backtest

def random_forest_session_old(df, year, ccy, session, model_output_filepath):
    # Ensure 'datetime' column is in datetime format
    df['datetime'] = pd.to_datetime(df['datetime'], errors='raise')
    # Keep a copy of the 'datetime' column before setting it as the index
    df['datetime_original'] = df['datetime']  # Add a backup column
    # Run through each line to remove the timezone info

    # Feature Engineering
    # Create lagged features
    lags = [1, 2, 3]
    for lag in lags:
        df[f'lag_{lag}'] = df['renko_price'].shift(lag)

    # Include `ticks_moved` as a feature
    df['ticks_moved_lag_1'] = df['ticks_moved'].shift(1)  # Add lagged `ticks_moved`

    # ─────────────────────────────────────────
    #       NEW FEATURES
    # ─────────────────────────────────────────
    # 1) 10-bar moving average of renko_price
    df['ma_10'] = df['renko_price'].rolling(window=10).mean()
    # 2) 5-bar max/min of renko_price
    df['max_5'] = df['renko_price'].rolling(window=5).max()
    df['min_5'] = df['renko_price'].rolling(window=5).min()
    # 3) Distance from MA10 and sign of renko_price
    df['distance_from_ma10'] = df['renko_price'] - df['ma_10']
    df['sign_of_distance'] = np.where(df['distance_from_ma10'] >= 0, 1, -1)

    # Fill missing values (introduced by lags)
    df.dropna(inplace=True)

    # Step 3: Define Features and Target
    features = [
        col for col in df.columns
        if (
            col.startswith('lag') or 
            'ticks_moved' in col or
            col in ['ma_10','max_5','min_5','distance_from_ma10','sign_of_distance']
        )
    ]
    df['target'] = (df['directions'].shift(-1) > 0).astype(int)


    df['datetime'] = df['datetime'].apply(lambda x: x.replace(tzinfo=None))
    # Set 'datetime' as the index
    df.set_index('datetime', inplace=True)
    # Convert the index to DatetimeIndex explicitly
    df.index = pd.to_datetime(df.index, errors='raise')


    # Verify that the index is a DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("The DataFrame index is not a DatetimeIndex. Ensure 'datetime' is properly converted.")

    # Extract the date from the index
    df['date'] = df.index.date
    unique_dates = df['date'].unique()  # Get unique session days
    print(f"Unique session days: {len(unique_dates)}")

    # Calculate train-test-backtest split based on unique session days
    train_days = int(len(unique_dates) * 0.6)
    test_days = int(len(unique_dates) * 0.2)
    backtest_days = len(unique_dates) - (train_days + test_days)

    # Split session days into train, test, and backtest
    train_dates = unique_dates[:train_days]
    test_dates = unique_dates[train_days:train_days + test_days]
    backtest_dates = unique_dates[train_days + test_days:]
    """"
    The : slicing syntax is used to extract a portion of the list.
    The part before : (in this case, nothing) means "start at the beginning of the list."
    The part after : (train_days) means "stop slicing at the index train_days (but not including it)."
    """
    print(f"Training days: {len(train_dates)}, Testing days: {len(test_dates)}, Backtesting days: {len(backtest_dates)}")

    # Filter the DataFrame for each split
    train_data = df[df['date'].isin(train_dates)]
    test_data = df[df['date'].isin(test_dates)]
    backtest_data = df[df['date'].isin(backtest_dates)]

    # Feature Engineering for Each Split
    X_train = train_data[features]
    y_train = train_data['target']

    X_test = test_data[features]
    y_test = test_data['target']

    X_backtest = backtest_data[features]
    y_backtest = backtest_data['target']

    # Normalize features (fit only on training data, apply to others)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_backtest_scaled = scaler.transform(X_backtest)

    # Step 4: Train Random Forest Classifier
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train_scaled, y_train)

    # Step 5: Evaluate Model
    y_pred_test = rf_model.predict(X_test_scaled)
    y_pred_backtest = rf_model.predict(X_backtest_scaled)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred_test)
    conf_matrix = confusion_matrix(y_test, y_pred_test)
    classification_report_text = classification_report(y_test, y_pred_test)

    # Display Results
    results = {
        "Accuracy": accuracy,
        "Confusion Matrix": conf_matrix,
        "Classification Report": classification_report_text
    }

    print(results)
    # Convert the results dictionary into a pandas DataFrame for saving to CSV
    results_df = pd.DataFrame.from_dict(
        {key: [value] if not isinstance(value, list) else value for key, value in results.items()}
    )
    # Save the results DataFrame to CSV
    results_df.to_csv(rf'{model_output_filepath}/{year}_{ccy}_{session}_random_forest_results.csv', index=False)
    df.to_csv(rf'{model_output_filepath}/{year}_{ccy}_{session}_random_forest_dataframe.csv')
    print(f"Model output saved to {model_output_filepath}")
    
    print(backtest_data.columns)

    return backtest_data, X_backtest, y_backtest, y_pred_backtest

def run_backtest(model_output_filepath, year, ccy, session, df, X_test, y_test, y_pred, initial_balance=10000, lot_size=10000):
    """
    Run a simple backtest using model predictions.

    Parameters:
    - df: DataFrame, the original data with features and target.
    - X_test: array, the test feature set. Provides the inputs for the model to generate predictions; indirectly referenced for metadata.
    - y_test: array, the actual target values for the test set. Used to evaluate whether the model's predictions (y_pred) are correct or not.
    - y_pred: array, the predicted target values for the test set. Drives the simulated trades and determines the backtesting outcomes.
    - initial_balance: float, the starting capital for backtesting.
    - lot_size: float, the trade size in units.

    Returns:
    - balance_history: list of balances over time.
    - trade_log: list of trades with details.
    """
    # Initialize variables
    balance = initial_balance
    balance_history = [balance]
    trade_log = []
    session_times = {
        "asian": datetime.time(19, 0),     # 7:00 PM
        "asian_morning": datetime.time(19, 0),     # 7:00 PM
        "london": datetime.time(3, 0),            # 3:00 AM
        "london_morning": datetime.time(3, 0),            # 3:00 AM
        "london_afternoon": datetime.time(7, 0),  # 7:00 AM
        "ny": datetime.time(8, 0),  # 8:00 AM
        "ny_morning": datetime.time(8, 0),  # 8:00 AM
        "ny_evening": datetime.time(12, 0),  # 8:00 AM
    }
    
    # Pick the session start time; default to 19:00 if not in dictionary
    session_start = session_times.get(session, datetime.time(19, 0))

    for i in range(len(y_pred)):
        current_idx = len(df) - len(X_test) + i
        current_dt = df.iloc[current_idx]['datetime_original']
        actual = y_test.iloc[i]
        predicted = y_pred[i]
        actual_open_price = df.iloc[len(df) - len(X_test) + i]['actual_openprice']
        previous_idx = len(df) - len(X_test) + i - 1  # the prior row
        if previous_idx >= 0:
            prev_open_price = df.iloc[previous_idx]['actual_openprice']
            fx_return = (actual_open_price - prev_open_price) / prev_open_price
        else:
            fx_return = 0  # Or NaN, etc.
        
        if i > 0:  # We have a previous row
            prev_dt = df.iloc[previous_idx]['datetime_original']
            
            # Check if we crossed from hour <19 to hour >=19 on the same day
            # OR we jumped to a new day whose time is >= 19:00.
            crossed_same_day = (
                prev_dt.date() == current_dt.date() 
                and prev_dt.time() < session_start 
                and current_dt.time() >= session_start
            )
            new_day_after_19 = (
                prev_dt.date() != current_dt.date() 
                and current_dt.time() >= session_start
            )

            if crossed_same_day or new_day_after_19:
                fx_return = 0

        # Simulate a trade based on predicted direction
        if predicted == 1:  # Buy signal
            trade_profit = fx_return * lot_size
        else:  # Sell signal
            trade_profit = -fx_return * lot_size

        # Update balance
        balance += trade_profit
        balance_history.append(balance)

        # Log trade
        trade_log.append({
            'tick': df.iloc[len(df) - len(X_test) + i]['tick_number'],
            'datetime': df.iloc[len(df) - len(X_test) + i]['datetime_original'],
            'renko_price': df.iloc[len(df) - len(X_test) + i]['renko_price'],
            'actual_open_price': actual_open_price,
            'predicted': predicted,
            'actual': actual,
            'fx_return':fx_return,
            'profit': trade_profit,
            'balance': balance
        })

    print(f"Test set size: {len(X_test)}")
    # Convert trade_log to a DataFrame for easier analysis
    trade_log_df = pd.DataFrame(trade_log)
    print(trade_log_df)
    trade_log_df.to_csv(rf'{model_output_filepath}/{year}_{ccy}_{session}_backtesting.csv')

    # Plot balance history
    plt.figure(figsize=(10, 6))
    plt.plot(balance_history, label='Account Balance')
    plt.title(f'Backtest Balance History ({year}_{ccy}_{session})')
    plt.xlabel('Trades')
    plt.ylabel('Balance')
    plt.legend()
    plt.show()
    
    return balance_history, trade_log
