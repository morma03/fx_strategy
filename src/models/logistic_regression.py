import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from .feature import create_features
from sklearn.linear_model import LogisticRegression

def logistic_regression_session(df, year, ccy, session, model_output_filepath):
    # 1) Basic date/time preparation
    df['datetime'] = pd.to_datetime(df['datetime'], errors='raise')
    df['datetime_original'] = df['datetime']

    # 2) Call your feature-engineering function (same as before)
    df, features = create_features(df)

    # 3) Define your target
    df['target'] = (df['directions'].shift(-1) > 0).astype(int)

    # 4) Remove timezone if present
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

    # Filter data for train/test/backtest
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

    # 8) Normalize features (fit on train, apply to test & backtest)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_backtest_scaled = scaler.transform(X_backtest)

    # 9) Train & Evaluate logistic model
    log_model = LogisticRegression(random_state=42)
    log_model.fit(X_train_scaled, y_train)

    y_pred_test = log_model.predict(X_test_scaled)
    y_pred_backtest = log_model.predict(X_backtest_scaled)

    # Evaluate model
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
    results_df.to_csv(rf'{model_output_filepath}/{year}_{ccy}_{session}_logistic_regression_results.csv', index=False)
    df.to_csv(rf'{model_output_filepath}/{year}_{ccy}_{session}_logistic_regression_dataframe.csv')
    print(f"Model output saved to {model_output_filepath}")

    print(backtest_data.columns)

    # Return the backtest subset & predictions for further analysis
    return backtest_data, X_backtest, y_backtest, y_pred_backtest
