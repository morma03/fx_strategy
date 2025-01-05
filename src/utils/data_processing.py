
import pandas as pd
 
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
