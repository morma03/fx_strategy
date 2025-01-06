import sys
import os
sys.path.append(os.path.abspath(r'C:\Users\mmori\Documents\fx_strategy_project\src'))
from utils.data_processing_tick import convert_to_tick
from utils.data_processing_tick import extract_significant_prices

year = "2023"
ccy = "usdjpy"
brick_size = 0.1

csv_filepath = rf'C:\Users\mmori\Documents\fx_strategy_project\notebooks\data\ASCII\M1\{ccy.lower()}\{year}'
csv_filename = f'DAT_ASCII_{ccy.upper()}_M1_{year}.csv'

output_file_path = 'output'
output_filename = f'{year}_{ccy.lower()}_tick.csv'

convert_to_tick(csv_filepath, csv_filename, output_file_path, output_filename, brick_size) 