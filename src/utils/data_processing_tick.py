import pandas as pd
import os

def generate_renko(data, brick_size):
    renko_data = []
    actual_openprice = []
    renko_timestamps = []
    ticks_moved = []  # To store the number of ticks moved per brick
    directions = []
    fx_returns = []
    open_prices = data['open'].values
    timestamps = data.index.values
    current_price = open_prices[0]

    for i in range(1, len(open_prices)):
        price = open_prices[i]
        # Calculate total ticks moved for this price update
        total_ticks = int(abs(price - current_price) // brick_size)  # Calculate number of full bricks
        while abs(price - current_price) >= brick_size:
            if price > current_price:
                current_price += brick_size
                renko_data.append(current_price)
                actual_openprice.append(price)
                renko_timestamps.append(timestamps[i])
                ticks_moved.append(total_ticks)
                directions.append("+1")                
            elif price < current_price:
                current_price -= brick_size
                renko_data.append(current_price)
                actual_openprice.append(price)
                renko_timestamps.append(timestamps[i])
                ticks_moved.append(total_ticks)
                directions.append("-1")
    
    # Convert actual_openprice to a pandas Series to calculate pct_change
    actual_openprice_series = pd.Series(actual_openprice)
    fx_returns = actual_openprice_series.pct_change() * 100  # Percentage change
    return renko_data, actual_openprice, renko_timestamps, ticks_moved, directions, fx_returns



# Function to convert to tick-like data
def convert_to_tick(csv_filepath, csv_filename, output_file_path, output_filename, brick_size):

    # Define the column names for the dataset
    column_names = ['datetime', 'open', 'high', 'low', 'close', 'volume']
    csv_fullpath = os.path.join(csv_filepath, csv_filename)
    df = pd.read_csv(csv_fullpath, names=column_names, header=None, delimiter=';', encoding='utf-8')

    # Convert the 'datetime' column to a datetime object
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Ensure that the 'datetime' column exists
    if 'datetime' in df.columns:
        # Set the 'datetime' column as the index for resampling
        df.set_index('datetime', inplace=True)
        
        # Set your desired brick size
        renko_bricks, actual_openprice, renko_timestamps, ticks_moved, directions, fx_returns = generate_renko(df, brick_size)

        # Create a DataFrame with tick-like data
        tick_like_data = pd.DataFrame({
            'tick_number': range(1, len(renko_bricks) + 1),
            'datetime': renko_timestamps,
            'price': renko_bricks,
            'actual_openprice': actual_openprice,
            'ticks_moved': ticks_moved,
            'directions': directions,
            'fx_return': fx_returns
        })

        # Save to CSV
        # Combine the file path and filename
        full_path = os.path.join(output_file_path, output_filename)
        tick_like_data.to_csv(full_path, index=False)
        print("Tick-like data has been saved to tick_like_data.csv")
        print(tick_like_data.head())  # Display the first few rows
       
        return tick_like_data

    else:
        print("Error: 'datetime' column not found in DataFrame")


def extract_significant_prices(csv_filepath, csv_filename, output_file_path, output_filename, brick_size):

    # Define the column names for the dataset
    column_names = ['datetime', 'open', 'high', 'low', 'close', 'volume']
    csv_fullpath = os.path.join(csv_filepath, csv_filename)
    df = pd.read_csv(csv_fullpath, names=column_names, header=None, delimiter=';', encoding='utf-8')
    # Convert the 'datetime' column to a datetime object
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Ensure that the 'datetime' column exists
    if 'datetime' in df.columns:
        # Set the 'datetime' column as the index for resampling
        df.set_index('datetime', inplace=True)

        significant_prices, significant_timestamps, directions, brick_sizes_moved, fx_returns = generate_price_moves_with_bricks(df, brick_size)

        # Create a DataFrame with tick-like data
        tick_like_data = pd.DataFrame({
            'tick_number': range(1, len(significant_prices) + 1),
            'datetime': significant_timestamps,
            'price': significant_prices,
            'ticks_moved': brick_sizes_moved,
            'directions': directions,
            'fx_return': fx_returns
        })

        # Save to CSV
        # Combine the file path and filename
        full_path = os.path.join(output_file_path, output_filename)
        tick_like_data.to_csv(full_path, index=False)
        print("Tick-like data has been saved to tick_like_data.csv")
        print(tick_like_data.head())  # Display the first few rows
       
        return tick_like_data

    else:
        print("Error: 'datetime' column not found in DataFrame")


def generate_price_moves_with_bricks(data, brick_size):
    significant_prices = []
    significant_timestamps = []
    directions = []
    brick_sizes_moved = []
    fx_returns = []
    
    open_prices = data['open'].values
    timestamps = data.index.values
    
    if len(open_prices) < 2:
        raise ValueError("Insufficient data to evaluate price moves.")
    
    for i in range(1, len(open_prices)):
        current_price = open_prices[i - 1]
        next_price = open_prices[i]
        
        # Check if the price move is significant
        price_move = abs(next_price - current_price)
        if price_move >= brick_size:
            significant_prices.append(next_price)
            significant_timestamps.append(timestamps[i])
            directions.append("+1" if next_price > current_price else "-1")
            brick_sizes_moved.append(price_move // brick_size)  # Number of brick sizes moved
    
    # Calculate percentage returns for significant prices
    if significant_prices:
        significant_price_series = pd.Series(significant_prices)
        fx_returns = significant_price_series.pct_change() * 100  # Percentage change
    
    return significant_prices, significant_timestamps, directions, brick_sizes_moved, fx_returns
