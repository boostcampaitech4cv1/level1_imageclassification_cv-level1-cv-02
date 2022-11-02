import os
import numpy as np
import pandas as pd
import argparse
import sys
from tqdm import tqdm
sys.path.append('./') # import를 위해 경로추가
from utils import Utility as U

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_train', type=str, default="./data/train/")
    parser.add_argument('--path_eval', type=str, default="./data/eval/")
    parser.add_argument('--path_train_output', type=str, default="./data/train_i.csv")
    parser.add_argument('--path_eval_output', type=str, default="./data/eval_i.csv")
    args = parser.parse_args()
    df_train = pd.read_csv(os.path.join(args.path_train, 'train.csv'))
    images = []
    for path in df_train['path']:
        __path_folder = os.path.join(*[args.path_train, 'images', path])
        __path_image = [os.path.join(*[__path_folder, p])  for p in os.listdir(__path_folder) if p[0] != '.' ]
        images.append(__path_image)

    df_train['images'] = images

    df_train.head()

    df_eval = pd.read_csv(os.path.join(args.path_eval, 'info.csv'))
    images = [os.path.join(*[args.path_eval, 'images', p])
              for p in df_eval['ImageID']]
    df_eval['images'] = images

    df_eval.head()
    image_df_labels = ['id', 'gender', 'age', 'mask', 'path']
    image_df_rows = []
    for _id, (_gender, _age, _images) in enumerate(zip(df_train['gender'], df_train['age'], df_train['images'])):
        for  _path in _images:
            _mask = U.convertImagePathToMaskStatus(_path)
            image_df_rows.append(
                [_id, _gender, _age, _mask, _path])
    image_df = pd.DataFrame(image_df_rows, columns=image_df_labels)
    image_df['gender_class'] = [U.encodeGender(g.capitalize()) for g in image_df['gender']]
    image_df['age_class'] = [U.encodeAge(a) for a in image_df['age']]
    image_df['mask_class'] = [U.encodeMask(m) for m in image_df['mask']]
    print('total number of images :', image_df.size / image_df.columns.size)
    image_df.sample(5)

    image_df.to_csv(args.path_train_output, index=False)
    df_eval.to_csv(args.path_eval_output, index=False)

