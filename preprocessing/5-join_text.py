import pandas as pd
import os

dataset_name = 'Sports_and_Outdoors'  

tsv_file_path = f'./processed/{dataset_name}/{dataset_name}_item_desc.tsv'
csv_file_path = f'./processed/{dataset_name}/image_summary_{dataset_name}.csv'
output_file_path = f'./processed/{dataset_name}/image_summary_description_{dataset_name}.csv'

os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

tsv_df = pd.read_csv(tsv_file_path, sep='\t')
csv_df = pd.read_csv(csv_file_path)

common_key = 'item_id' 

tsv_summary_map = dict(zip(tsv_df[common_key], tsv_df['summary']))

def append_summary(row):
    original_summary = row['summary']
    item_id = row[common_key]
    
    if item_id in tsv_summary_map:
        additional_summary = tsv_summary_map[item_id]
        return f"{original_summary} {additional_summary}"
    else:
        return original_summary

csv_df['summary'] = csv_df.apply(append_summary, axis=1)

csv_df.to_csv(output_file_path, index=False)