import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np
import random
import sys
import os
sys.path.append('../')  # import를 위해 경로추가
from utils import Utility as U


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
    args = parser.parse_args()

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
    df_merge['ans'] = [ 
        U.convertAgeGenderMaskToLabel(mask, gender, age)
        for mask, gender, age 
        in zip(df_merge['mask'], df_merge['gender'], df_merge['age']) 
        ]
    df_merge = df_merge.drop(['age', 'gender', 'mask'], axis=1)
    print(df_merge.head())
    print('done!')
    df_merge.to_csv(os.path.join(args.csv_path, args.csv_result_name), index=False)
