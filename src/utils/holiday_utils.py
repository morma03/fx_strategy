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