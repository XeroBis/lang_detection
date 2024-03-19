import pandas as pd
import re


def get_data(exclude=None, include=None):
    pattern = r'\(([A-Z]{3})\)'

    # Fichier brut
    input_file_path = 'data/train.txt'
    output_file_path = 'data/clean.txt'

    with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
        # Read each line from the input file
        for line in input_file:
            modified_line = re.sub(pattern, r'\1\t', line)
            output_file.write(modified_line)
    df = pd.read_csv('data/clean.txt', sep='\t', header=None)
    df.columns = ['Lang', 'Text']
    df.to_csv('data/train.csv', index=False)
    if exclude is not None:
        df = df[~df['Lang'].isin(exclude)]
    elif include is not None:
        df = df[df['Lang'].isin(include)]
    return df