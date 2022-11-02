from sklearn import preprocessing
import sys
import os
import torchvision.models as models
from albumentations.pytorch.transforms import ToTensorV2
import albumentations as A
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
import random
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import glob
sys.path.append('../')  # import를 위해 경로추가
from models import *
from utils import CustomDataset
from utils import Utility as U

def inference(model, test_loader, device):
    model.to(device)
    model.eval()

    preds = []

    with torch.no_grad():
        for item in tqdm(iter(test_loader)):
            images = item['image'].to(device)
            pred = model(images)
            pred = pred.argmax(dim=-1)
            preds.extend(pred.cpu().numpy())
    print('Done.')
    return preds

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=41)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--path_ref', default='../')
    parser.add_argument('--path_csv', type=str, default='data/eval_i.csv')
    parser.add_argument('--path_save', type=str, default='test/csv/')
    parser.add_argument('--path_tar', type=str, default='models/checkpoint/')
    parser.add_argument('--save_name', type=str, default="submission_all_v8.csv")
    parser.add_argument('--tar_name', type=str, default='Weight_VIT_V8.KHS.tar')
    args = parser.parse_args()
    print(args)
    U.setSeedEverything(args.seed)

    df = pd.read_csv(os.path.join(args.path_ref, args.path_csv))
    print(df.head())

    # image_paths = [os.path.join(args.path_ref, p) for p in df['images'].values]
    tar = torch.load(os.path.join(args.path_ref, args.path_tar, args.tar_name))

    eval_transform = A.Compose([
    A.Resize(tar['args'].img_size, tar['args'].img_size),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0, always_apply=False, p=1.0),
                ToTensorV2()
                ])

    eval_dataset = CustomDataset(args.path_ref, df['images'].values, None, eval_transform)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = eval(tar['args'].target_model)
    model.load_state_dict(tar['model'])
    preds = inference(model, eval_loader, args.device)
    df_result = pd.DataFrame()
    df_result['ImageID'] = df['ImageID']
    df_result['ans'] = preds
    df_result.to_csv(os.path.join(args.path_ref, args.path_save, args.save_name), index=False)
    print('Saved at ', os.path.join(args.path_ref, args.path_save, args.save_name))

