import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import seaborn as sns

def plot_daily_rolling_average(df, window_minutes):
    """
    This function calculates and plots the rolling average of the 'fx_return' column.
   
    Parameters:
    df (pandas.DataFrame): DataFrame containing 'datetime' and 'fx_return' columns.
    window_minutes (int): The rolling window size in minutes. For example, 10,080 for one week.
    """
    # Ensure 'fx_return' and 'datetime' exist in the DataFrame
    if 'fx_return' not in df.columns or 'datetime' not in df.columns:
        print("Error: 'fx_return' or 'datetime' column not found in DataFrame.")
        return
   
    # Drop NaN values in 'fx_return'
    df_clean = df.dropna(subset=['fx_return']).copy()  # Create a copy to avoid SettingWithCopyWarning
   
    # Convert 'datetime' column to datetime format if not already
    df_clean['datetime'] = pd.to_datetime(df_clean['datetime'])

    # Remove timezone information from the 'datetime' column
    df_clean['datetime'] = df_clean['datetime'].dt.tz_localize(None)

    # Set the datetime column as the index (necessary for time-based rolling)
    df_clean.set_index('datetime', inplace=True)

    # Calculate the rolling average using a numeric window in minutes (e.g., 10,080 for one week)
    rolling_avg = df_clean['fx_return'].rolling(window=window_minutes).mean()

    # Add the rolling average to the DataFrame
    df_clean["rolling_avg"] = rolling_avg

    # Plot the rolling average
    plt.figure(figsize=(10, 6))
    plt.plot(df_clean.index, rolling_avg, label=f'{window_minutes}-Minute Rolling Average', color='blue')

    # Add title and labels
    plt.title(f'{window_minutes}-Minute Rolling Average of FX Returns')
    plt.xlabel('Date')
    plt.ylabel('FX Return')
   
    # Show legend
    plt.legend()

    # Display the plot
    plt.show()
   
    return df_clean

def plot_boxplot_fx_return(df):
    """
    This function creates a box plot for the 'fx_return' column in the DataFrame
    to visualize the distribution and detect outliers, after handling missing values.
    """
    # Ensure 'fx_return' exists in the DataFrame
    if 'fx_return' not in df.columns:
        print("Error: 'fx_return' column not found in DataFrame.")
        return

    # Drop NaN values from the 'fx_return' column
    df_clean = df['fx_return'].dropna()

    # Check if there's enough data after dropping NaNs
    if df_clean.empty:
        print("Error: No valid data to plot after dropping NaN values.")
        return

    # Create the box plot for fx_return
    plt.figure(figsize=(8, 6))
    plt.boxplot(df_clean, vert=False, patch_artist=True, boxprops=dict(facecolor='lightblue'))
   
    # Add title and labels
    plt.title('Box Plot of FX Returns')
    plt.xlabel('FX Return')
   
    # Display the plot
    plt.show()
    
def plot_autocorrelation(df, lags=50):
    """
    This function plots the autocorrelation of the 'fx_return' column.
   
    Parameters:
    df (pandas.DataFrame): DataFrame containing 'datetime' and 'fx_return' columns.
    lags (int): Number of lags for which the autocorrelation will be calculated.
    """
    # Ensure 'fx_return' exists in the DataFrame
    if 'fx_return' not in df.columns:
        print("Error: 'fx_return' column not found in DataFrame.")
        return
   
    # Drop NaN values in 'fx_return'
    df_clean = df.dropna(subset=['fx_return']).copy()

    # Convert 'datetime' column to datetime format if not already
    df_clean['datetime'] = pd.to_datetime(df_clean['datetime'])
    df_clean.set_index('datetime', inplace=True)
   
    # Plot the autocorrelation using statsmodels' plot_acf function
    plt.figure(figsize=(10, 6))
    sm.graphics.tsa.plot_acf(df_clean['fx_return'], lags=lags, alpha=0.05)
   
    # Add title and labels
    plt.title('Autocorrelation of FX Returns')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
   
    # Display the plot
    plt.show()

def normalize_and_plot_distribution(df):
    """
    This function normalizes the 'fx_return' column using z-score standardization
    and plots its distribution.
   
    Parameters:
    df (pandas.DataFrame): DataFrame containing 'datetime' and 'fx_return' columns.
    """
    # Ensure 'fx_return' exists in the DataFrame
    if 'fx_return' not in df.columns:
        print("Error: 'fx_return' column not found in DataFrame.")
        return
   
    # Drop NaN values in 'fx_return'
    df_clean = df.dropna(subset=['fx_return']).copy()

    # Standardize the 'fx_return' column (z-score normalization)
    mean_return = df_clean['fx_return'].mean()
    std_return = df_clean['fx_return'].std()
    df_clean['fx_return_normalized'] = (df_clean['fx_return'] - mean_return) / std_return
   
    # Plot the distribution of the normalized returns
    plt.figure(figsize=(10, 6))
    sns.histplot(df_clean['fx_return_normalized'], bins=50, kde=True, color='blue')
   
    # Add title and labels
    plt.title('Distribution of Normalized FX Returns')
    plt.xlabel('Normalized FX Return (Z-Score)')
    plt.ylabel('Density')
   
    # Show the plot
    plt.show()