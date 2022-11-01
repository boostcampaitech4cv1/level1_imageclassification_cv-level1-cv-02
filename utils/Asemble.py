import os
import argparse
import numpy as np
import pandas as pd

def asemble(inputs):
    """
    mask / gender / age 로 구분한 뒤에 max 값을 선택 만약에 모두 같다면 첫번째 input 선택(best 스코어)
    """
    m, g, a = [0,0,0], [0,0], [0,0,0]
    for i, input in enumerate(inputs):
        mask, gender, age = input//6%3, input//3%2, input%3
        if i == 0:
            m[mask] += 1.1
            g[gender] += 1.1
            a[age] += 1.1
        else:
            m[mask] += 1
            g[gender] += 1
            a[age] += 1
    
    mask = np.argmax(m)
    gender = np.argmax(g)
    age = np.argmax(a)
    
    return mask*6 + gender*3 + age

def main(args):
    file_names = os.listdir(args.csv_path)
    df_main = pd.DataFrame()
    df_temp = pd.read_csv(os.path.join(args.csv_path,file_names[0]))
    df_main['ImageID'] = df_temp['ImageID']
    for idx, file_name in enumerate(file_names):
        df_temp = pd.read_csv(os.path.join(args.csv_path, file_name))
        df_main[f'ans{idx}'] = df_temp['ans']

    ans_list = []
    for i in range(len(df_main)):
        data = df_main.loc[i].values
        ans_list.append(asemble(data[1:]))
    
    df_temp['ans'] = ans_list
    df_temp.to_csv(os.path.join(args.save_path , args.save_name), index=False)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Data and model checkpoints directories
    parser.add_argument('--csv_path', type=str, default='../test/csv/asembly')
    parser.add_argument('--save_dir', type=str, default='../models/ckpt')
    parser.add_argument('--save_path', type=str, default='../test/csv')
    parser.add_argument('--save_name', type=str, default='asembly.csv')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    main(args)