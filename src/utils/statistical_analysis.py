import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

def correlation_analysis_resample(target_filepath, feature_filepath, target_name, feature_name, year, output_filepath, resample_interval=None):
    column_names = ['datetime', 'open', 'high', 'low', 'close', 'volume']
    feature_data = pd.read_csv(feature_filepath, names=column_names, header=None, delimiter=';', encoding='utf-8')
    target_data = pd.read_csv(target_filepath, names=column_names, header=None, delimiter=';', encoding='utf-8')

    print(feature_data.head())
    print(target_data.head())

    feature_data['datetime'] = pd.to_datetime(feature_data['datetime'])
    target_data['datetime'] = pd.to_datetime(target_data['datetime'])

    # Resample the data if a resample interval is provided
    if resample_interval:
        feature_data.set_index('datetime', inplace=True)
        target_data.set_index('datetime', inplace=True)

        feature_data = feature_data.resample(resample_interval).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        target_data = target_data.resample(resample_interval).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        feature_data.reset_index(inplace=True)
        target_data.reset_index(inplace=True)

    aligned_data = pd.merge(feature_data, target_data, on='datetime', suffixes=(f'_{feature_name}', f'_{target_name}'))
    print(aligned_data.head())
    aligned_data.to_csv(os.path.join(output_filepath, f'{year}_aligned_{feature_name}_{target_name}_{resample_interval}.csv'), index=False)

    aligned_data[f'return_{feature_name}'] = aligned_data[f'open_{feature_name}'].pct_change()
    aligned_data[f'return_{target_name}'] = aligned_data[f'open_{target_name}'].pct_change()

    aligned_data[f'return_{target_name}_t+1'] = aligned_data[f'return_{target_name}'].shift(-1)
    aligned_data = aligned_data.dropna()

    correlation = aligned_data[f'return_{feature_name}'].corr(aligned_data[f'return_{target_name}_t+1'])
    print(correlation)
    return correlation

# Define a function for statistical analysis
def statistical_analysis(df, session=None, day=None, result_file_path=None):
    """
    Perform statistical analysis on a subset of the time series based on session and day.
    Save the result to a CSV file with the format 'fx_return_{session}_{day}.csv'.
   
    Parameters:
        df (DataFrame): The input time series DataFrame with a datetime index.
        session (str): The trading session or sub-session to filter by ('Asian', 'London', 'NY', etc.).
        day (str): The day of the week to filter by ('Monday', 'Tuesday', etc.).
        result_file_path (str): The directory path where the result CSV file should be saved.

    Returns:
        stats (DataFrame): A DataFrame containing statistical analysis (mean, std, etc.).
        Plot: A plot of the filtered time series.
    """
   
    # Ensure 'datetime' is in datetime format
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')

    # Check for NaT values in the 'datetime' column
    if df['datetime'].isna().sum() > 0:
        print(f"Found {df['datetime'].isna().sum()} missing datetime values (NaT).")
        # Option 1: Fill missing NaT values (forward fill)
        df.fillna({'datetime': df['datetime'].ffill()}, inplace=True)
        # Option 2: Drop rows with NaT in 'datetime'
        # df = df.dropna(subset=['datetime'])

    # Convert the index to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)

    # Filter by day of the week if specified
    if day:
        df = df[df.index.day_name() == day]

    # Filter by trading session if specified
    if session:
        df = filter_by_session_nyc(df, session)

    # Calculate basic statistics (mean, std, etc.)
    stats = pd.DataFrame({
    'Statistic': ['count', 'mean', 'std', 'min', '25%', '50% (median)', '75%', 'max', '5th percentile', '95th percentile', 'autocorr (lag-1)'],
    'fx_return': [df['fx_return'].count(),
                  df['fx_return'].mean(),
                  df['fx_return'].std(),
                  df['fx_return'].min(),
                  df['fx_return'].quantile(0.25),
                  df['fx_return'].median(),
                  df['fx_return'].quantile(0.75),
                  df['fx_return'].max(),
                  df['fx_return'].quantile(0.05),
                  df['fx_return'].quantile(0.95),
                  df['fx_return'].autocorr(lag=1)]
    })

    # Displaying the corrected statistics table
    print(stats)

    # Plot the time series
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df['fx_return'], label='FX Return')
    plt.title(f'Time Series for {session} Session on {day}')
    plt.xlabel('Datetime')
    plt.ylabel('FX Return')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Save the results to a CSV file if a result_file_path is provided
    if result_file_path:
        # Clean up the session and day strings for the filename
        session_str = session.replace(" ", "").lower() if session else "all_sessions"
        day_str = day.lower() if day else "all_days"
        
        # Create the filename based on session and day
        file_name = f"fx_return_{session_str}_{day_str}.csv"
        
        # Combine the file path and filename
        full_path = os.path.join(result_file_path, file_name)
        
        # Save the filtered DataFrame to a CSV file
        df.to_csv(full_path, encoding='utf-8-sig')
        print(f"Results saved to {full_path}")

    return stats, df

def filter_by_session_nyc(df, session, output_file=None):
    """
    Filters the DataFrame by the given trading session or sub-session using NYC time as a base.
   
    Trading session/sub-session times (NYC time):
    - asian: 19:00 (prev day) - 04:00
    - asian_morning: 19:00 (prev day) - 01:00
    - london: 03:00 - 12:00
    - london_morning: 03:00 - 07:00
    - london_afternoon: 07:00 - 12:00
    - ny: 08:00 - 17:00
    - ny_morning: 08:00 - 12:00
    - ny_evening: 12:00 - 17:00
   
    Parameters:
        df (DataFrame): The input time series DataFrame (with NYC timezone-aware timestamps).
        session (str): The trading session or sub-session ('asian', 'london_morning', etc.).
        output_file (str): Path to save the filtered DataFrame as a CSV file. If None, no file is saved.

    Returns:
        DataFrame: Filtered DataFrame for the session.
    """
    print(f"Function called with session: {session}", flush=True)

    # Define session times
    session_times = {
        'asian': ('19:00', '04:00'),
        'asian_morning': ('19:00', '01:00'),
        'london': ('03:00', '12:00'),
        'london_morning': ('03:00', '07:00'),
        'london_afternoon': ('07:00', '12:00'),
        'ny': ('08:00', '17:00'),
        'ny_morning': ('08:00', '12:00'),
        'ny_evening': ('12:00', '17:00')
    }

    # Validate session input
    if session not in session_times:
        raise ValueError("Invalid session. Choose from the predefined sessions.")

    start_time, end_time = session_times[session]

    try:
        # Special handling for sessions crossing midnight
        if session in ['asian', 'asian_morning']:
            # Debug: After start time
            print(f"Filtering rows after start time {start_time}...", flush=True)
            after_start = df.between_time(start_time, '23:59:59')
            if after_start.empty:
                print(f"No rows found after {start_time}.", flush=True)
            else:
                print(f"Rows after {start_time}:\n{after_start}", flush=True)

            # Debug: Before end time
            print(f"Filtering rows before end time {end_time}...", flush=True)
            before_end = df.between_time('00:00', end_time)
            if before_end.empty:
                print(f"No rows found before {end_time}.", flush=True)
            else:
                print(f"Rows before {end_time}:\n{before_end}", flush=True)

            # Combine the two
            print("Combining rows...", flush=True)
            filtered = pd.concat([after_start, before_end])
            print(f"Combined rows:\n{filtered}", flush=True)
        else:
            # General case for sessions not crossing midnight
            print(f"Filtering rows for {session} ({start_time} to {end_time})...", flush=True)
            filtered = df.between_time(start_time, end_time)
            filtered = filtered[filtered.index.time < pd.Timestamp(end_time).time()]
            print(f"Filtered rows:\n{filtered}", flush=True)

        # Save to CSV if requested
        if output_file:
            filtered.to_csv(output_file)
            print(f"Filtered data saved to {output_file}", flush=True)

        return filtered

    except Exception as e:
        print(f"An error occurred: {e}", flush=True)
        return None
