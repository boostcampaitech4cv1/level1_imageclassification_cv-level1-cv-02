import pandas as pd
import numpy as np
import os
import sys
sys.path.append('../')
import argparse
import json
from tqdm import tqdm

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from sklearn.model_selection import train_test_split

from utils.CustomDataset import CustomDataset
from models.ResNext_GenderV0_sawol import ResNext_GenderV0_SAWOL
from utils.Utility import setSeedEverything, calcF1Score

def train(args):

    # setting
    setSeedEverything(args.seed)
    
    # augmentation
    transform = A.Compose([
        A.RandomResizedCrop(
        args.resize[0], args.resize[1], scale=(0.8, 1.0)),
        A.RandomBrightnessContrast(p=0.3),
        A.RandomGamma(p=0.3),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(
            0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
        ToTensorV2()
    ])

    # dataset & dataloader
    df = pd.read_csv(args.train_csv)
    train_df, val_df, _, _ = train_test_split(df, df, test_size=0.2, random_state=args.seed)
    
    train_dataset = CustomDataset(args.train_path, train_df.path.values, train_df.age_class.values, transforms=transform)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True,
        drop_last=True)

    val_dataset = CustomDataset(args.train_path, val_df.path.values, val_df.age_class.values, transforms=transform)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True,
        drop_last=True)

    # model
    model = ResNext_GenderV0_SAWOL().to(args.device)

    # loss & metric
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    scheduler = None

    # logging
    # logger = SummaryWriter(log_dir=args.log_dir)
    with open(os.path.join(args.save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    # train
    best_val_acc = 0
    best_val_loss = np.inf
    best_model = None
    for epoch in range(1, args.epochs+1):
        model.train()
        loss_value = 0
        matches = 0
        for idx, data in enumerate(tqdm(train_loader)):
            inputs = data['image'].to(args.device)
            labels = data['label'].to(args.device)

            optimizer.zero_grad()

            outs = model(inputs)
            preds = torch.argmax(outs, dim=-1)
            loss = criterion(outs, labels)

            loss.backward()
            optimizer.step()

            loss_value += loss.item()
            matches += (preds == labels).sum().item()
            if (idx + 1) % args.log_interval == 0:
                train_loss = loss_value / args.log_interval
                train_acc = matches / args.batch_size / args.log_interval
                # logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
                # logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)

                loss_value = 0
                matches = 0
        
        # scheduler.step()
        
        with torch.no_grad():
            model.eval()
            val_loss_items = []
            val_acc_items = []
            figure = None
            for val_data in val_loader:
                inputs, labels = val_data['image'], val_data['label']
                inputs = inputs.to(args.device)
                labels = labels.to(args.device)

                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)
                f1_score = calcF1Score(preds.detach().cpu(), labels.detach().cpu())

                loss = criterion(outs, labels).item()
                acc = (labels == preds).sum().item()
                val_loss_items.append(loss)
                val_acc_items.append(acc)

                best_val_loss = min(best_val_loss, loss)
                if acc > best_val_acc:
                    best_val_acc = acc
                    best_model = model
            
            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_acc = np.sum(val_acc_items) / len(val_dataset)
            best_val_loss = min(best_val_loss, val_loss)

            # logger.add_scalar("Val/loss", val_loss, epoch)
            # logger.add_scalar("Val/accuracy", val_acc, epoch)

        print(f'EPOCH: {epoch}/{args.epochs} | train loss: {train_loss:.3f}, train acc: {train_acc:.3f} | val loss: {val_loss:.3f}, val acc: {val_acc:.3f} f1_score: {f1_score:.3f}')

    os.makedirs(f'{args.save_dir}', exist_ok=True)
    torch.save(best_model.state_dict(), f"{args.save_dir}/{args.epochs}_best_model.pth")

    print(f'best_val_acc: {best_val_acc} | best_val_loss: {best_val_loss:.3f}')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=999, help='random seed (default: 999)')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 10)')
    parser.add_argument("--resize", type=list, default=[224, 224], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--log_dir', type=str, default='./log')
    parser.add_argument('--train_path', type=str, default='../data/train/images/')
    parser.add_argument('--train_csv', type=str, default='../train_i.csv')
    parser.add_argument('--save_dir', type=str, default='../models/ckpt')

    parser.add_argument('--device', type=str, default='cuda')


    args = parser.parse_args()
    train(args)