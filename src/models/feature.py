import numpy as np
import pymannkendall as mk

def create_features(df):
    """
    Take in the original DataFrame and add all feature columns:
      - lag_X
       - rolling mean/min/max
      - distance_from_ma10, sign_of_distance
      + Mann–Kendall test statistic for a chosen window
    Then drop NaNs introduced by shifting/rolling.

    Returns:
      (modified_df, features_list)
    """
    # ─────────────────────────────────────
    # 1) Create lagged features
    # ─────────────────────────────────────
    lags = [1, 2, 3]
    for lag in lags:
        df[f'lag_{lag}'] = df['renko_price'].shift(lag)

    df['ticks_moved_lag_1'] = df['ticks_moved'].shift(1)

    # ─────────────────────────────────────
    # 2) Rolling stats
    # ─────────────────────────────────────
    df['ma_10'] = df['renko_price'].rolling(window=10).mean()
    df['max_5'] = df['renko_price'].rolling(window=5).max()
    df['min_5'] = df['renko_price'].rolling(window=5).min()

    df['distance_from_ma10'] = df['renko_price'] - df['ma_10']
    df['sign_of_distance'] = np.where(df['distance_from_ma10'] >= 0, 1, -1)

    # ─────────────────────────────────────
    # 3) Mann–Kendall feature
    # ─────────────────────────────────────
    window_mk = 50  # choose your window size
    mk_values = []

    renko_prices = df['renko_price'].values  # to speed up indexing

    for i in range(len(renko_prices)):
        # If we don't have enough history, append NaN
        if i < window_mk:
            mk_values.append(np.nan)
        else:
            # Subset of 'renko_price' for the last 50 bars up to i
            subset = renko_prices[i - window_mk : i]
            # Run the Mann–Kendall test
            result = mk.original_test(subset)

            mk_values.append(result.s)
            # result has these attributes:
            #  - result.trend (string: 'increasing','decreasing','no trend')
            #  - result.h (True if significant)
            #  - result.p (p-value)
            #  - result.z (z-score)
            #  - result.T (the MK test statistic S)
            #  - result.slope (Theil–Sen slope)
            # etc.

    df['mk_stat_50'] = mk_values

    # ─────────────────────────────────────
    # 4) Drop NaNs introduced by shift/rolling/MK
    # ─────────────────────────────────────
    df.dropna(inplace=True)

    # ─────────────────────────────────────
    # 5) Create the feature list
    # ─────────────────────────────────────
    features_list = [
        col
        for col in df.columns
        if (
            col.startswith('lag')
            or 'ticks_moved' in col
            or col
            in [
                'ma_10',
                'max_5',
                'min_5',
                'distance_from_ma10',
                'sign_of_distance',
                'mk_stat_50',  # NEW
            ]
        )
    ]

    return df, features_list


def create_features_old(df):
    """
    Take in the original DataFrame and add all feature columns:
      - lag_X
      - ticks_moved_lag_1
      - rolling mean/min/max
      - distance_from_ma10, sign_of_distance
    Then drop NaNs introduced by shifting/rolling.
    
    Returns:
      (modified_df, features_list)
    """
    # -- Create lagged features --
    lags = [1, 2, 3]
    for lag in lags:
        df[f'lag_{lag}'] = df['renko_price'].shift(lag)

    # Add lagged ticks_moved
    df['ticks_moved_lag_1'] = df['ticks_moved'].shift(1)

    # -- NEW FEATURES --
    df['ma_10'] = df['renko_price'].rolling(window=10).mean()
    df['max_5'] = df['renko_price'].rolling(window=5).max()
    df['min_5'] = df['renko_price'].rolling(window=5).min()

    df['distance_from_ma10'] = df['renko_price'] - df['ma_10']
    df['sign_of_distance'] = np.where(df['distance_from_ma10'] >= 0, 1, -1)

    # Drop rows with NaNs from shift/rolling
    df.dropna(inplace=True)

    # Create the feature list
    features_list = [
        col for col in df.columns
        if (
            col.startswith('lag')
            or 'ticks_moved' in col
            or col in [
                'ma_10', 'max_5', 'min_5',
                'distance_from_ma10', 'sign_of_distance'
            ]
        )
    ]

    return df, features_list
