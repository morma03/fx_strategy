import holidays
import pandas as pd

def get_global_holidays(year, country_codes):
    """
    Fetch public holidays for multiple countries and return a combined list of holiday dates and names.
    """
    all_holidays = []
    
    for country_code in country_codes:
        try:
            print(f"Attempting to fetch holidays for {country_code}...")
            
            # Dynamically get the holidays class for the country using the `holidays` library
            country_holidays = getattr(holidays, country_code)(years=year)
            
            # Add (date, name) tuples to the list
            for holiday_date, holiday_name in country_holidays.items():
                all_holidays.append((holiday_date, holiday_name))
        
        except AttributeError as e:
            print(f"Country code '{country_code}' is not supported by the 'holidays' library. Error: {e}")
    
    # Sort holidays by date
    all_holidays.sort(key=lambda x: x[0])
    
    print(f"All holidays fetched for {country_codes}: {all_holidays}\n")
    return all_holidays

def add_holidays_to_fx_data(df, cleaned_file_path, year, country_codes):
    """
    Adds public holidays from specified countries to the FX data and saves the updated DataFrame.
    """
    
    # Ensure 'datetime' column is a pandas datetime index
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
    
    # Debug step: Check if the datetime column is correctly set
    print("\nDataFrame index after conversion to datetime:")
    print(df.index)

    # Initialize an empty dictionary to hold holiday date mappings for each country
    holiday_names_dict = {}
    
    # Loop through each country code and fetch holidays
    for country_code in country_codes:
        print(f"\nFetching holidays for {country_code}...")
        
        # Fetch holidays for the current country
        country_holidays = get_global_holidays(year, [country_code])
        
        if not country_holidays:
            print(f"No holidays found for {country_code} in {year}.")
        
        # Create a dictionary mapping holiday dates to holiday names
        holiday_names = {pd.Timestamp(holiday_date).normalize(): holiday_name 
                         for holiday_date, holiday_name in country_holidays}
        
        # Store the names for this country in the dictionary
        holiday_names_dict[country_code] = holiday_names
        
        # Debug step: Print the fetched holiday names
        print(f"Holidays for {country_code}: {holiday_names}")
    
    # Add new columns for each country, mapping holiday names and flags
    for country_code in country_codes:
        country_holiday_name_col = f"{country_code}_holiday_name"
        country_holiday_flag_col = f"{country_code}_holiday"
        
        # Normalize the FX data index (datetime) to date level for mapping purposes
        df_normalized_dates = df.index.normalize()
        
        # Debug step: Check the unique normalized dates in the DataFrame
        print(f"\nNormalized dates in the DataFrame:\n{df_normalized_dates.unique()}")
        
        # Map holiday names to FX data by normalized date
        df[country_holiday_name_col] = df_normalized_dates.map(holiday_names_dict[country_code])
        
        # Debug step: Check the mapping for specific dates
        specific_dates = ['2023-01-01', '2023-01-02']
        print(f"\nHoliday mapping for specific dates ({country_code}):")
        for date in specific_dates:
            if date in df.index:
                print(f"{date}: {df.loc[date][country_holiday_name_col]}")
        
        # Add boolean columns indicating whether a row is a holiday
        df[country_holiday_flag_col] = df[country_holiday_name_col].notna()
    
    # Debug step: Print the updated DataFrame to verify changes
    print("\nUpdated DataFrame with holidays added:")
    print(df.head(10))
    
    # Save the updated DataFrame back to the CSV file
    df.to_csv(cleaned_file_path, encoding='utf-8-sig')    
    return df

def remove_friday(input_filepath, output_filepath):
    data = pd.read_csv(input_filepath)

    # Filter out rows where 'day_of_week' is 'Friday'
    filtered_data = data[data['day_of_week'] != 'Friday']
    filtered_data.to_csv(output_filepath, index=False)

    print("Rows with 'Friday' in the 'day_of_week' column have been removed.")
    print(f"Filtered data saved to: {output_filepath}")

def extract_weekday(input_filepath, output_filepath, weekday):
    data = pd.read_csv(input_filepath)

    # Filter out rows where 'day_of_week' matches the specified weekday
    filtered_data = data[data['day_of_week'] == weekday]
    filtered_data.to_csv(output_filepath, index=False)

    print(f"Rows with '{weekday}' in the 'day_of_week' column have been extracted.")
    print(f"Filtered data saved to: {output_filepath}")

def filter_by_session_nyc(file_path, session, output_file=None):
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
    print(f"Filtering session: {session}")

    # Define the column names for the dataset
    column_names = ['datetime', 'open', 'high', 'low', 'close', 'volume']
    df = pd.read_csv(file_path, names=column_names, header=None, delimiter=';', encoding='utf-8')

    # Convert the 'datetime' column to a datetime object
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index("datetime", inplace=True)
    df.index = df.index.tz_localize('America/New_York')

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

    if session not in session_times:
        raise ValueError("Invalid session. Choose from the predefined sessions.")

    start_time, end_time = session_times[session]

    # Initialize empty DataFrame
    filtered = pd.DataFrame()
    
    if session in ['asian', 'asian_morning']:
        try:
            # Split the DataFrame by date for precision
            df['date'] = df.index.date
            df['time'] = df.index.time

            print("Filtering previous day's rows...")
            prev_day_filter = (df['time'] >= pd.Timestamp(start_time).time())
            prev_day_rows = df[prev_day_filter]
            print(f"Previous day rows:\n{prev_day_rows}")

            print("Filtering next day's rows...")
            next_day_filter = (df['time'] < pd.Timestamp(end_time).time())
            next_day_rows = df[next_day_filter]
            print(f"Next day rows:\n{next_day_rows}")

            # Combine the results
            print("Combining results...")
            filtered = pd.concat([prev_day_rows, next_day_rows]).sort_index()
            print(f"Filtered DataFrame:\n{filtered}")

        except Exception as e:
            print(f"Error during filtering: {e}")
    else:
        filtered = df.between_time(start_time, end_time)
        filtered = filtered[filtered.index.time < pd.Timestamp(end_time).time()]
        print(f"Filtered rows for {session}:\n{filtered}", flush=True)
    
    if output_file:
        filtered.to_csv(output_file)
        print(f"Filtered data saved to {output_file}")


    return filtered