import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
import random
import glob
import sys
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from sklearn.model_selection import train_test_split

sys.path.append('/opt/ml/mask/cv2_mask')
from utils.BaseDataset import MaskOnlyDataset, BaseAugmentation
from models.Backbone import ResnetBackBone
from utils.Utility import Args, encodeAge, setSeedEverything, calcF1Score, load_model

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train(data_dir, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # -- dataset
    dataset = MaskOnlyDataset(
        data_dir=data_dir,
    )

    # -- augmentation
    transform = BaseAugmentation(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )
    dataset.set_transform(transform)
    
    # -- data_loader
    train_dataset, val_dataset = dataset.split_dataset()
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=2)

    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=2)
    
    model = ResnetBackBone().to(device)
    loss_func = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None

    best_val_acc = 0
    best_val_loss = np.inf
    best_model = None

    train_loss_items = []
    train_acc_items = []
    val_loss_items = []
    val_acc_items = []

    for epoch in range(1, args.epochs+1):
        model.train()
        for data in tqdm(train_loader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outs = model(inputs)
            preds = torch.argmax(outs, dim=-1)
            train_loss = loss_func(outs, labels)

            train_loss.backward()
            optimizer.step()

            train_loss_items.append(train_loss.item())
            train_acc = (preds == labels).sum().item() / len(preds)
            train_acc_items.append(train_acc)

        with torch.no_grad():
            for val_data in val_loader:
                inputs, labels = val_data
                inputs = inputs.to(device)
                labels = labels.to(device)


                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)
                f1_score = calcF1Score(preds.detach().cpu(), labels.detach().cpu())

                val_loss = loss_func(outs, labels).item()
                val_acc = (labels == preds).sum().item() / len(labels)
                val_loss_items.append(val_loss)
                val_acc_items.append(val_acc)

                best_val_loss = min(best_val_loss, val_loss)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model = model
                
        
        print(f'EPOCH: {epoch}/{args.epochs} | train loss: {train_loss:.3f}, train acc: {train_acc:.3f} | val loss: {val_loss:.3f}, val acc: {val_acc:.3f} f1_score: {f1_score:.3f}')

    os.makedirs(f'{args.save_dir}', exist_ok=True)
    torch.save(best_model.state_dict(), f"{args.save_dir}/{args.epochs}_best.pth")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=dict, default={'train': '../data/train/images/', 'test': '../data/eval/images/'})
    parser.add_argument('--csv_path', type=dict, default={'train':'../data/train/train.csv', 'test': '../data/eval_i.csv'})
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 1)')
    parser.add_argument("--resize", type=list, default=[224, 224], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-3)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')
    parser.add_argument('--save_dir', type=str, default=os.environ.get('SM_SAVE_DIR', './models/ckpt'))
    parser.add_argument('--device', type=str)
    
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', './data/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, args)