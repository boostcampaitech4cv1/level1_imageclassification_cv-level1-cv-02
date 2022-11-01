import os
import sys
import tarfile
import cv2
import glob
import numpy as np
import pandas as pd
import random
import argparse
import matplotlib.pyplot as plt
import multiprocessing as mp
from tqdm import tqdm
sys.path.append('../')  # import를 위해 경로추가
from models import *
from utils import CustomDataset
from utils import Utility as U

def save_image(item):
    path_load, path_save, pred = item
    _img = cv2.imread(path_load)
    _img = cv2.resize(_img, dsize=(round(0.5*_img.shape[1]), round(0.5*_img.shape[0])))
    mask, gender, age = U.convertLabelToMaskGenderAge(pred)
    mask, gender, age = U.decodeMask(mask), U.decodeGender(gender), U.decodeAge(age)
    if (gender == 'Male'): # todo
        gender_color = (255,0,0)
    else:
        gender_color = (0, 0, 255)
    if (mask == 'Wear'):
        mask_color = (200,22,22)
    elif (mask == 'Incorrect'):
        mask_color = (22,200,200)
    else :
        mask_color = (22,22,200)
    age_color = (55,200,55)
    dy = 20
    font_scale = 0.7
    thickness_back = 3
    thickness_front = 2
    # for mask
    cv2.putText(_img, mask, (0, int(_img.shape[0]-(dy*0.5))),
                cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, color=(0), thickness=thickness_back, bottomLeftOrigin=False)
    cv2.putText(_img, mask, (0, int(_img.shape[0]-(dy*0.5))),
                cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, color=mask_color, thickness=thickness_front, bottomLeftOrigin=False)
    # for gender
    cv2.putText(_img, gender, (0, int(_img.shape[0]-(dy*1.5))),
                cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, color=(0), thickness=thickness_back, bottomLeftOrigin=False)
    cv2.putText(_img, gender, (0, int(_img.shape[0]-(dy*1.5))),
                cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, color=gender_color, thickness=thickness_front, bottomLeftOrigin=False)
    # for age
    cv2.putText(_img, age, (0, int(dy)),
                cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, color=(0), thickness=thickness_back, bottomLeftOrigin=False)
    cv2.putText(_img, age, (0, int(dy)),
                cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, color=age_color, thickness=thickness_front, bottomLeftOrigin=False)
    cv2.imwrite(path_save, _img)
    return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_scale', type=int, default=0.5)
    parser.add_argument('--path_ref', default='../')
    parser.add_argument('--path_image', default='data/eval/images/')
    parser.add_argument('--path_csv', type=str, default='test/csv/submission.csv')
    parser.add_argument('--path_save_folder', type=str, default="data/result_images/")
    parser.add_argument('--path_save_zip_name', type=str, default="data/result_images.tar")
    args = parser.parse_args()
    os.makedirs(os.path.join(args.path_ref, args.path_save_folder), exist_ok=True)
    df = pd.read_csv(os.path.join(args.path_ref, args.path_csv))
    print(df.head())
    pool = mp.Pool(mp.cpu_count() // 2)
    path_load = [os.path.join(args.path_ref, args.path_image, p) for p in df['ImageID'].values]
    path_save = [os.path.join(args.path_ref, args.path_save_folder, p) for p in df['ImageID'].values]
    
    print('Processing start...')
    _ = pool.map(save_image, zip(path_load, path_save, df['ans'].values))
    print('Processing done.')

    path_zip = os.path.join(args.path_ref, args.path_save_zip_name)
    print('Save zip file at ', path_zip)
    if(os.path.exists(path_zip)):
        os.remove(path_zip)
    with tarfile.open(path_zip, 'w') as mytar:
        for rp in path_save:
            mytar.add(rp)
