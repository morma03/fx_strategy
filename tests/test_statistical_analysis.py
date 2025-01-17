import sys
import os
import pandas as pd
import pandas as pd

sys.path.append(os.path.abspath(r'C:\Users\mmori\Documents\fx_strategy_project\src'))
from utils.statistical_analysis import filter_by_session_nyc
notebook_path = os.path.abspath(os.path.join(os.getcwd(), '../notebooks'))
output_file_path = rf'{notebook_path}\output\correlation_analysis'
output_session_filepath = rf'{notebook_path}\data\processed\session'
input_file_path = rf'{notebook_path}\data\processed'


year = "2023"
ccy = "gbpusd"
session = "asian_morning"
base_price = "prev"

file_path = rf'{input_file_path}/{year}_{ccy}_tick_{base_price}_price_as_base.csv'
output_filepath = rf'{output_session_filepath}/{year}_{ccy}_{session}_tick_{base_price}_price_as_base.csv'

# Load the dataset
df = pd.read_csv(file_path)
df["datetime"] = pd.to_datetime(df["datetime"])
df.set_index("datetime", inplace=True)
df.index = df.index.tz_localize('America/New_York')

filtered_df = filter_by_session_nyc(df, session, output_filepath)