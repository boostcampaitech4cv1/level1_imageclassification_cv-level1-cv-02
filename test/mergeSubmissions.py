import os
import sys
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse


# Main
if __name__ == '__main__':
    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str,
                        default='./csv/')
    parser.add_argument('--csv_age_name', type=str,
                        default='submission_age.csv')
    parser.add_argument('--csv_gender_name', type=str,
                        default='submission_gender.csv')
    parser.add_argument('--csv_mask_name', type=str,
                        default='submission_mask.csv')
    parser.add_argument('--csv_result_name', type=str,
                        default="submission.csv")
    args = parser.parse_args('')

    # Load CSV
    df_age = pd.read_csv(os.path.join(args.csv_path, args.csv_age_name))
    df_gender = pd.read_csv(os.path.join(args.csv_path, args.csv_gender_name))
    df_mask = pd.read_csv(os.path.join(args.csv_path, args.csv_mask_name))

    # Merge Columns
    df_merge = pd.DataFrame()
    df_merge['ImageID'] = df_age['ImageID']
    df_merge['age'] = df_age['ans']
    df_merge['gender'] = df_gender['ans']
    df_merge['mask'] = df_mask['ans']

    print(df_merge.head())
