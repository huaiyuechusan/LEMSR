import pandas as pd
import os

def convert_to_inter(input_file, output_file):
    df = pd.read_csv(input_file, sep='\t')

    df['rating'] = 1.0

    df.columns = ['user_id:token', 'item_id:token', 'timestamp:float', 'rating:float']
    
    df = df[['user_id:token', 'item_id:token', 'rating:float', 'timestamp:float']]
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    df.to_csv(output_file, sep='\t', index=False, header=True)


dataset = "Video_Games"
path = f"./processed/{dataset}/"

input_file = path + f'{dataset}_u_i_pairs.tsv'
output_file =  path + f'{dataset}.inter'

convert_to_inter(input_file, output_file)


i_map_df = pd.read_csv(path + f'{dataset}_i_map.tsv', sep='\t')

i_map_df.rename(columns={'original': 'asin'}, inplace=True)
i_map_df.to_csv(path + 'RecBole_i_id_mapping.csv', sep='\t', index=False)

u_map_df = pd.read_csv(path + f'{dataset}_u_map.tsv', sep='\t')

u_map_df.rename(columns={'original': 'reviewerID'}, inplace=True)
u_map_df.to_csv(path + 'RecBole_u_id_mapping.csv', sep='\t', index=False)