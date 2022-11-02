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


def validation(model, criterion, test_loader, args):
    model.eval()

    model_preds = []
    true_labels = []

    val_loss = []
    val_acc = []

    with torch.no_grad():
        for item in tqdm(test_loader):
            inputs = item['image']
            labels = item['label']
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)

            outs = model(inputs)
            preds = torch.argmax(outs, dim=-1)
            
            loss = criterion(outs, labels)
            acc = (labels == preds).sum().item()

            val_loss.append(loss.item())
            val_acc.append(acc / args.batch_size)

            model_preds += outs.argmax(1).detach().cpu().numpy().tolist()
            true_labels += labels.detach().cpu().numpy().tolist()

    val_f1 = U.calcF1Score(true_labels, model_preds)
    return np.mean(val_loss), np.mean(val_acc), val_f1


def train(model, optimizer, train_loader, test_loader, scheduler, args, datetime, path_model_weight):
    model.to(args.device)

    criterion = nn.CrossEntropyLoss()#.to(args.device)

    best_score = 0

    for epoch in range(1, args.epochs+1):
        model.train()
        train_loss = []
        train_acc = []
        for item in tqdm(train_loader):
            inputs = item['image']
            labels = item['label']
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            optimizer.zero_grad()

            outs = model(inputs)
            preds = torch.argmax(outs, dim=-1)
            loss = criterion(outs, labels)

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            train_acc.append((preds == labels).sum().item() / args.batch_size)

        tr_loss = np.mean(train_loss)
        tr_acc = np.mean(train_acc)

        val_loss, val_acc, f1 = validation(model, criterion, test_loader, args)

        print(f'Ep[{epoch}] / T.Loss[{tr_loss:.5f}] / T.Acc[{tr_acc:.5f}] / V.Loss[{val_loss:.5f}] / V.Acc[{val_acc:.5f}] / F1[{f1:.5f}]')

        if args.step_enable:
            scheduler.step()

        if best_score < f1:
            best_score = f1
            print(
                f' * New Best Model -> Epoch [{epoch}] / best_score : [{best_score:.5f}]')
            U.saveModel(model=model, optimizer=optimizer,
                        args=args, datetime=datetime, path=path_model_weight)
            print(" -> The model has been saved at " + path_model_weight)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="params")
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--validation_ratio', type=float, default=0.2)
    parser.add_argument('--step_enable', type=bool, default=True)
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--step_gamma', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=41)
    parser.add_argument('--img_size', type=int, default=384)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--csv_path', type=str, default='../data/train_i.csv')
    parser.add_argument('--save_path', type=str,
                        default="../models/checkpoint/")
    parser.add_argument('--save_name', type=str,
                        default="Weight_VIT_V1_KHS_SGD.tar")
    parser.add_argument('--target_model', type=str,
                        default="VIT_V1_KHS(False)")
    args = parser.parse_args()
    print(args)

    U.setSeedEverything(args.seed)

    target_model = eval(args.target_model)
    df = pd.read_csv(args.csv_path)
    train_df, val_df = U.splitDataset(
        df,  validation_ratio=args.validation_ratio, random_state=args.seed)
    train_img_paths = train_df['path'].values
    train_labels = [U.convertAgeGenderMaskToLabel(m, g, a) for m, g, a in zip(train_df['mask_class'].values, train_df['gender_class'].values, train_df['age_class'].values)]
    val_img_paths = val_df['path'].values
    val_labels = [U.convertAgeGenderMaskToLabel(m, g, a) for m, g, a in zip(val_df['mask_class'].values, val_df['gender_class'].values, val_df['age_class'].values)]
    train_transform = A.Compose([
                                A.Resize(args.img_size, args.img_size),
                                # A.RandomResizedCrop(
                                #     args.img_size, args.img_size, scale=(0.6, 1.0)),
                                # A.RandomBrightnessContrast(p=0.3),
                                # A.RandomGamma(p=0.3),
                                # # A.RandomFog(),
                                # A.RandomToneCurve(),
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
    # optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=args.lr, 
        weight_decay=5e-4)

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
