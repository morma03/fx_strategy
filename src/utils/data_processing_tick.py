import pandas as pd
import os

def generate_renko_base_first_price(data, brick_size):
    # Initialize variables
    renko_data = []
    actual_openprice = []
    renko_timestamps = []
    ticks_moved = []  # To store the number of ticks moved per brick
    directions = []
    fx_returns = []

    # Extract prices and timestamps
    open_prices = data['open'].values
    timestamps = data.index.values
    current_price = open_prices[0]

    for i in range(1, len(open_prices)):
        price = open_prices[i]
        timestamp = timestamps[i]

        # Calculate price difference
        price_diff = price - current_price
        total_ticks = int(abs(price_diff) // brick_size)

        # Only proceed if price difference is significant enough to form a brick
        if total_ticks > 0:
            for j in range(total_ticks):
                # Update current price in the direction of the move
                current_price += brick_size if price_diff > 0 else -brick_size

                # Only keep the final brick for this price movement
                if j == total_ticks - 1:
                    renko_data.append(current_price)
                    actual_openprice.append(price)
                    renko_timestamps.append(timestamp)
                    ticks_moved.append(total_ticks)
                    directions.append("+1" if price_diff > 0 else "-1")

    # Calculate percentage returns for the bricks
    if actual_openprice:  # Ensure the list is not empty
        actual_openprice_series = pd.Series(actual_openprice)
        fx_returns = actual_openprice_series.pct_change() * 100  # Percentage change

    return renko_data, actual_openprice, renko_timestamps, ticks_moved, directions, fx_returns


def generate_renko_base_first_price_keep_intermediate_brick_move(data, brick_size):
    # Initialize variables
    renko_data = []
    actual_openprice = []
    renko_timestamps = []
    ticks_moved = []  # To store the number of ticks moved per brick
    directions = []
    fx_returns = []

    # Extract prices and timestamps
    open_prices = data['open'].values
    timestamps = data.index.values
    current_price = open_prices[0]

    for i in range(1, len(open_prices)):
        price = open_prices[i]
        timestamp = timestamps[i]

        # Calculate price difference
        price_diff = price - current_price
        total_ticks = int(abs(price_diff) // brick_size)

        # Only proceed if price difference is significant enough to form a brick
        if total_ticks > 0:
            for j in range(total_ticks):
                # Update current price in the direction of the move
                current_price += brick_size if price_diff > 0 else -brick_size

                # Append Renko brick details
                renko_data.append(current_price)
                actual_openprice.append(price)
                renko_timestamps.append(timestamp if j == total_ticks - 1 else f"{timestamp}_brick_{j+1}")
                ticks_moved.append(total_ticks)
                directions.append("+1" if price_diff > 0 else "-1")

    # Calculate percentage returns for the bricks
    if actual_openprice:  # Ensure the list is not empty
        actual_openprice_series = pd.Series(actual_openprice)
        fx_returns = actual_openprice_series.pct_change() * 100  # Percentage change

    return renko_data, actual_openprice, renko_timestamps, ticks_moved, directions, fx_returns

def generate_renko_base_prev_price(data, brick_size):
    # Initialize variables
    renko_data = []
    actual_openprice = []
    renko_timestamps = []
    ticks_moved = []
    directions = []
    fx_returns = []

    # Extract prices and timestamps
    open_prices = data['open'].values
    timestamps = data.index.values

    # Start with the first price
    current_price = open_prices[0]

    for i in range(1, len(open_prices)):
        # Use the last Renko price as the base, or open_prices[0] for the first iteration
        base_price = renko_data[-1] if renko_data else open_prices[0]
        target_price = open_prices[i]
        timestamp = timestamps[i]

        # Calculate price difference
        price_diff = target_price - base_price
        total_ticks = int(abs(price_diff) // brick_size)

        # Form Renko bricks if significant price movement occurs
        for j in range(total_ticks):
            # Update current_price based on direction
            direction = 1 if price_diff > 0 else -1
            current_price = base_price + direction * brick_size * (j + 1)

            # Only keep the final brick for this price movement
            if j == total_ticks - 1:
                renko_data.append(current_price)
                actual_openprice.append(target_price)
                renko_timestamps.append(timestamp)
                ticks_moved.append(total_ticks)
                directions.append(direction)

    # Calculate percentage returns
    if actual_openprice:  # Ensure the list is not empty
        actual_openprice_series = pd.Series(actual_openprice)
        fx_returns = actual_openprice_series.pct_change() * 100  # Percentage change

    return renko_data, actual_openprice, renko_timestamps, ticks_moved, directions, fx_returns

def generate_renko_base_prev_price_keep_intermediate_brick_move(data, brick_size):
    # Initialize variables
    renko_data = []
    actual_openprice = []
    renko_timestamps = []
    ticks_moved = []
    directions = []
    fx_returns = []

    # Extract prices and timestamps
    open_prices = data['open'].values
    timestamps = data.index.values

    # Start with the first price
    current_price = open_prices[0]

    for i in range(1, len(open_prices)):
        # Use the last Renko price as the base, or open_prices[0] for the first iteration
        base_price = renko_data[-1] if renko_data else open_prices[0]
        target_price = open_prices[i]
        timestamp = timestamps[i]

        # Calculate price difference
        price_diff = target_price - base_price
        total_ticks = int(abs(price_diff) // brick_size)

        # Form Renko bricks if significant price movement occurs
        for j in range(total_ticks):
            # Update current_price based on direction
            direction = 1 if price_diff > 0 else -1
            current_price = base_price + direction * brick_size * (j + 1)

            # Append Renko brick details
            renko_data.append(current_price)
            actual_openprice.append(target_price)
            renko_timestamps.append(timestamp if j == total_ticks - 1 else f"{timestamp}_brick_{j+1}")
            ticks_moved.append(total_ticks)
            directions.append(direction)

    # Calculate percentage returns
    if actual_openprice:  # Ensure the list is not empty
        actual_openprice_series = pd.Series(actual_openprice)
        fx_returns = actual_openprice_series.pct_change() * 100  # Percentage change

    return renko_data, actual_openprice, renko_timestamps, ticks_moved, directions, fx_returns



# Function to convert to tick-like data
def convert_to_tick(csv_filepath, csv_filename, output_file_path, output_filename, brick_size, use_first_price_as_base):

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
        if use_first_price_as_base:
            renko_bricks, actual_openprice, renko_timestamps, ticks_moved, directions, fx_returns = generate_renko_base_first_price(df, brick_size)
        else:
            renko_bricks, actual_openprice, renko_timestamps, ticks_moved, directions, fx_returns = generate_renko_base_prev_price(df, brick_size)
        
        # Create a DataFrame with tick-like data
        tick_like_data = pd.DataFrame({
            'tick_number': range(1, len(renko_bricks) + 1),
            'datetime': renko_timestamps,
            'price': renko_bricks,
            'actual_openprice': actual_openprice,
            'ticks_moved': ticks_moved,
            'directions': directions,
            'fx_return': fx_returns,
        })
        # Add the day of the week to the DataFrame
        tick_like_data['day_of_week'] = pd.to_datetime(tick_like_data['datetime']).dt.day_name()

        # Save to CSV
        # Combine the file path and filename
        full_path = os.path.join(output_file_path, output_filename)
        tick_like_data.to_csv(full_path, index=False)
        print("Tick-like data has been saved to tick_like_data.csv")
        print(tick_like_data.head())  # Display the first few rows
       
        return tick_like_data

    else:
        print("Error: 'datetime' column not found in DataFrame")
