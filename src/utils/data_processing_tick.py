import pandas as pd
import os
import re
import glob

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

    # Calculate percentage returns
    """
    if actual_openprice:  # Ensure the list is not empty
        actual_openprice_series = pd.Series(actual_openprice)
        fx_returns = actual_openprice_series.pct_change() * 100  # Percentage change
    """

    return renko_data, actual_openprice, renko_timestamps, ticks_moved, directions #, fx_returns

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
    """
    if actual_openprice:  # Ensure the list is not empty
        actual_openprice_series = pd.Series(actual_openprice)
        fx_returns = actual_openprice_series.pct_change() * 100  # Percentage change
    """

    return renko_data, actual_openprice, renko_timestamps, ticks_moved, directions #, fx_returns


# Function to convert to tick-like data
def convert_to_tick(csv_filepath, output_file_path, output_filename, brick_size, use_first_price_as_base):

    df = pd.read_csv(csv_filepath)

    if 'datetime' not in df.columns:
        print("Error: 'datetime' column not found in DataFrame")
        return

    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce', utc=True)
    df['datetime'] = df['datetime'].dt.tz_convert('America/New_York')

    print(df['datetime'].dtype)
    print(df['datetime'].head())
    print(df['datetime'].isna().sum())

    # Now set index
    df.set_index('datetime', inplace=True)
        
    # Set your desired brick size
    if use_first_price_as_base:
        renko_bricks, actual_openprice, renko_timestamps, ticks_moved, directions = generate_renko_base_first_price(df, brick_size)
    else:
        renko_bricks, actual_openprice, renko_timestamps, ticks_moved, directions = generate_renko_base_prev_price(df, brick_size)

    
    # Fix: convert renko_timestamps from UTC → America/New_York
    # or if they are naive but represent UTC, you'd do tz_localize('UTC').tz_convert('America/New_York').
    # If they're already tz-aware in UTC, just tz_convert:
    renko_timestamps_local = (
        pd.to_datetime(renko_timestamps, utc=True)  # parse as UTC
        .tz_convert('America/New_York')           # convert to local
    )
    
    # Create a DataFrame with tick-like data
    tick_like_data = pd.DataFrame({
        'tick_number': range(1, len(renko_bricks) + 1),
        'datetime': renko_timestamps_local,
        'renko_price': renko_bricks,
        'actual_openprice': actual_openprice,
        'ticks_moved': ticks_moved,
        'directions': directions
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

def create_summary(data_dir, result_file):

    filename_pattern = re.compile(
        rf"(\d{{4}})_([a-zA-Z]{{6}})_(\w+(?:_\w+)*)_({result_file}|backtesting)\.csv"
    )

    summary_data = {}    
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))

    print("Files found:", csv_files)
    for filepath in csv_files:
        filename = os.path.basename(filepath)
        print("Checking:", filename)

        match = filename_pattern.match(filename)
        if not match:
            print("  --> Regex did NOT match. Skipping.")
            continue

        year, ccy, session, file_type = match.groups()
        print(f"  --> Regex matched: year={year}, ccy={ccy}, session={session}, file_type={file_type}")
        key = (year, ccy, session)

        if key not in summary_data:
            summary_data[key] = {
                "year": year,
                "ccy": ccy,
                "session": session,
                "accuracy": None,
                "backtest_start": None,
                "backtest_end": None,
                "closing_balance": None
            }
        
        df = pd.read_csv(filepath)
        print("  --> DataFrame columns:", df.columns.tolist())

        if file_type == result_file:
            # Make sure "Accuracy" actually exists in df.columns
            print("Columns are:", df.columns.tolist())
            if "Accuracy" in df.columns:
                summary_data[key]["accuracy"] = df["Accuracy"].iloc[0]
            else:
             print("No 'Accuracy' column found! Found these columns:", df.columns.tolist())
            # Possibly skip or store None

        else:  # backtesting
            # Make sure "datetime" and "balance" exist in df.columns
            summary_data[key]["backtest_start"] = df["datetime"].iloc[0]
            summary_data[key]["backtest_end"]   = df["datetime"].iloc[-1]
            summary_data[key]["closing_balance"] = df["balance"].iloc[-1]

    # After the loop, see what we collected
    print("\nSummary so far:")
    for k, v in summary_data.items():
        print(k, "->", v)

    output_df = pd.DataFrame(summary_data.values())
    output_file = os.path.join(data_dir, "final_summary.csv")
    output_df.to_csv(output_file, index=False)
    print("Final summary saved to:", output_file)
