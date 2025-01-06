
import pandas as pd
import os

def convert_to_every_30min(csv_filepath, csv_filename):
    # Define the column names for the dataset
    column_names = ['datetime', 'open', 'high', 'low', 'close', 'volume']
    full_path = os.path.join(csv_filepath, csv_filename)
    df = pd.read_csv(full_path, names=column_names, header=None, delimiter=';', encoding='utf-8')
 
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Ensure that the 'datetime' column exists
    if 'datetime' in df.columns:
        # Set the 'datetime' column as the index for resampling
        df.set_index('datetime', inplace=True)

        # Resample the DataFrame to keep only every 30 minutes (12:00, 12:30, 13:00, etc.)
        df = df.resample('30min').first()  # Overwriting df with resampled data
        # df = df.resample('1h').first()

        # Reset the index to bring 'datetime' back as a column if needed
        df.reset_index(inplace=True)

        # Display the resampled DataFrame with the new headers
        print("CSV file successfully read and resampled to 30-minute intervals!")
        print(df.head())  # Display the first few rows
       
        return df
    else:
        print("Error: 'datetime' column not found in DataFrame")

def resample_fx_return_by_every_hour(df, full_path, aggregation_method='sum'):
    # Ensure datetime is parsed correctly
    if 'datetime' not in df.columns:
        raise KeyError("'datetime' column not found in the DataFrame")

    # Ensure datetime column is in datetime format
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Set the datetime column as the index
    df.set_index('datetime', inplace=True)

    # Reset the index to ensure datetime column is accessible
    reshaped_df = df.reset_index()

    # Create separate date and time columns
    reshaped_df['date'] = reshaped_df['datetime'].dt.date
    reshaped_df['time'] = reshaped_df['datetime'].dt.time

    # Filter to include only numeric columns for aggregation
    numeric_cols = reshaped_df.select_dtypes(include='number')

    # Handle duplicate date-time entries by aggregating the numeric columns
    reshaped_df = reshaped_df[['date', 'time']].join(numeric_cols)

    if aggregation_method == 'sum':
        reshaped_df = reshaped_df.groupby(['date', 'time']).sum().reset_index()
    elif aggregation_method == 'mean':
        reshaped_df = reshaped_df.groupby(['date', 'time']).mean().reset_index()

    # Pivot table with date as index and time as columns
    pivot_df = reshaped_df.pivot(index='date', columns='time', values='fx_return')

    # Save the output to a new CSV file    
    pivot_df.to_csv(full_path)


def clean_data_get_fx_return(df, cleaned_file_path):
    # Ensure that the 'datetime' column is in the correct format and set it as the index
    if 'datetime' in df.columns:
        # Convert 'datetime' column to a datetime object
        print("Converting 'datetime' column to datetime object...")
        df['datetime'] = pd.to_datetime(df['datetime'])

        # Set 'datetime' as the index
        print("Setting 'datetime' as index...")
        df.set_index('datetime', inplace=True)

    # Display the first few rows to check the structure
    print("First few rows of the data:")
    print(df.head())

    # Check for duplicate timestamps
    print("Checking for duplicate timestamps...")
    duplicates = df.index.duplicated(keep=False)

    # Display the duplicated entries (optional)
    if duplicates.any():
        print("Duplicate timestamps found:")
        print(df[df.index.duplicated(keep=False)])

    # Remove duplicates and keep the first occurrence
    print("Removing duplicates and keeping the first occurrence...")
    df = df[~df.index.duplicated(keep='first')]

    # Check if the index is already timezone-aware
    if df.index.tz is None:
        # Localize to 'America/New_York' timezone and convert to UTC
        print("Localizing to 'America/New_York' timezone and converting to UTC...")
        df = df.tz_localize('America/New_York', ambiguous='NaT', nonexistent='shift_forward').tz_convert('UTC')
    else:
        # Convert directly if it's already timezone-aware
        print("Index is already timezone-aware, converting to UTC...")
        df = df.tz_convert('UTC')

    # Check for missing values in the datetime index
    print("Checking for missing values...")
    missing_data = df.isnull().sum()
    print("Missing data per column:")
    print(missing_data)

    # Fill missing data (forward fill or back fill as per your requirements)
    print("Filling missing data...")
    df.ffill(inplace=True) # Forward fill missing values
    # df.fillna(method='ffill', inplace=True)  

    # Optional: Clean up data types and round values if necessary
    print("Rounding numerical columns...")
    df = df.round(6)  # Round numerical columns for consistency

    # Calculate percentage returns and scale them by multiplying by 100
    df['fx_return'] = df['open'].pct_change() * 100
    
    # Add a new column for the day of the week
    print("Adding 'day_of_week' column...")
    df['day_of_week'] = df.index.day_name()

    # Save the cleaned DataFrame to a new CSV    
    df.to_csv(cleaned_file_path)
    print(f"Cleaned FX data saved to {cleaned_file_path}")
    return df