import os
import sys
sys.path.append('../')  # import를 위해 경로추가
from utils import Utility as U
from utils import CustomDataset

import datetime
import argparse
from torch.utils.data import Dataset, DataLoader
from models import *
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torchvision.models as models
from sklearn import preprocessing


def validation(model, criterion, test_loader, device):
    model.eval()

    model_preds = []
    true_labels = []

    val_loss = []

    with torch.no_grad():
        for item in tqdm(test_loader):
            img = item['image']
            label = item['label']
            h = item['height']
            w = item['width']
            img, label = img.float().to(device), label.type(torch.FloatTensor).to(device)
            model_pred = model(img)

            loss = criterion(
                model_pred.view(-1), label)

            val_loss.append(loss.item())

            model_preds += model_pred.detach().cpu().numpy().tolist()
            true_labels += label.detach().cpu().numpy().tolist()

    y_preds_raw = np.array(model_preds).squeeze()
    y_preds = np.zeros(y_preds_raw.shape, np.int32)
    y_preds[y_preds_raw >= 0.5] = 1
    y_label = np.array(true_labels)
    val_acc = 0.
    for i in range(len(true_labels)):
        if (y_preds[i] == y_label[i]):
            val_acc += 1.
    val_acc = val_acc / len(true_labels)
    return np.mean(val_loss), val_acc


def train(model, optimizer, train_loader, test_loader, scheduler, args, datetime, path_model_weight):
    model.to(args.device)

    criterion = nn.BCEWithLogitsLoss().to(args.device)

    best_score = 0

    for epoch in range(1, args.epochs+1):
        model.train()
        train_loss = []
        for item in tqdm(train_loader):
            img = item['image']
            label = item['label']
            h = item['height']
            w = item['width']
            img, label = img.float().to(args.device), label.type(
                torch.FloatTensor).to(args.device)
            optimizer.zero_grad()

            model_pred = model(img)
            loss = criterion(model_pred.view(-1), label)

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        tr_loss = np.mean(train_loss)

        val_loss, val_acc = validation(
            model, criterion, test_loader, args.device)

        print(
            f'Epoch [{epoch}] / Train Loss : [{tr_loss:.5f}] / Val Loss : [{val_loss:.5f}] / Val Acc : [{val_acc:.5f}]')

        if scheduler is not None:
            scheduler.step()

        if best_score < val_acc:
            best_score = val_acc
            print(
                f' * New Best Model -> Epoch [{epoch}] / best_score : [{best_score:.5f}]')
            U.saveModel(model=model, optimizer=optimizer,
                        args=args, datetime=datetime, path=path_model_weight)
            print(" -> The model has been saved at " + path_model_weight)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="params")
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--validation_ratio', type=float, default=0.2)
    parser.add_argument('--step_size', type=int, default=5)
    parser.add_argument('--step_gamma', type=float, default=0.8)
    parser.add_argument('--lr', type=float, default=4e-3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=41)
    parser.add_argument('--img_size', type=int, default=244)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--csv_path', type=str, default='../train_i.csv')
    parser.add_argument('--save_path', type=str,
                        default="../models/checkpoint/")
    parser.add_argument('--save_name', type=str,
                        default="GenderWeight.tar")
    parser.add_argument('--target_model', type=str,
                        default="ResNext_GenderV0_KHS()")
    args = parser.parse_args()
    print(args)

    U.setSeedEverything(args.seed)

    target_model = eval(args.target_model)
    df = pd.read_csv(args.csv_path)
    train_df, val_df = U.splitDataset(
        df,  validation_ratio=args.validation_ratio, random_state=args.seed)
    train_img_paths = train_df['path'].values
    train_labels = train_df['gender_class'].values
    val_img_paths = val_df['path'].values
    val_labels = val_df['gender_class'].values
    train_transform = A.Compose([
                                # A.Resize(args.img_size, args.img_size),
                                A.RandomResizedCrop(
                                    args.img_size, args.img_size, scale=(0.8, 1.0)),
                                A.RandomBrightnessContrast(p=0.3),
                                A.RandomGamma(p=0.3),
                                A.RandomFog(),
                                A.RandomToneCurve(),
                                A.HorizontalFlip(p=0.5),
                                A.Normalize(mean=(0.485, 0.456, 0.406), std=(
                                    0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                                ToTensorV2()
                                ])

    test_transform = A.Compose([
        A.Resize(args.img_size, args.img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                    max_pixel_value=255.0, always_apply=False, p=1.0),
        ToTensorV2()
    ])
    train_dataset = CustomDataset(
        '.', train_img_paths, train_labels, train_transform)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    val_dataset = CustomDataset('.', val_img_paths, val_labels, test_transform)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = target_model.to(args.device)
    model.eval()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)

    # scheduler = None
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.step_gamma)

    # start time
    dt = datetime.datetime.now()

    train(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        test_loader=val_loader,
        scheduler=scheduler,
        args=args,
        datetime=dt,
        path_model_weight=os.path.join(args.save_path, args.save_name)
    )
