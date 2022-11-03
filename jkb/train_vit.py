import argparse
import glob
import json
import multiprocessing
import os
import random
import re
from importlib import import_module
from pathlib import Path


import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset2 import MaskBaseDataset, TestDataset
from loss import create_criterion
from sklearn.metrics import f1_score


from EarlyStopping import EarlyStopping
from sklearn.model_selection import StratifiedKFold, KFold
import time
from albumentations import *
from albumentations.pytorch import ToTensorV2
import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

import torch.nn as nn


# StratifiedKFold, early stopping, oof, Gradient Accumulation



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


def grid_image(np_images, gts, preds, n=16, shuffle=False):
    batch_size = np_images.shape[0]
    assert n <= batch_size

    choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n))
    figure = plt.figure(figsize=(12, 18 + 2))  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
    plt.subplots_adjust(top=0.8)  # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
    n_grid = int(np.ceil(n ** 0.5))
    tasks = ["mask", "gender", "age"]
    for idx, choice in enumerate(choices):
        gt = gts[choice].item()
        pred = preds[choice].item()
        image = np_images[choice]
        gt_decoded_labels = MaskBaseDataset.decode_multi_class(gt)
        pred_decoded_labels = MaskBaseDataset.decode_multi_class(pred)
        title = "\n".join([
            f"{task} - gt: {gt_label}, pred: {pred_label}"
            for gt_label, pred_label, task
            in zip(gt_decoded_labels, pred_decoded_labels, tasks)
        ])

        plt.subplot(n_grid, n_grid, idx + 1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)

    return figure

def calc_f1score(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')
    

def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"
    
def get_transforms(need=('train', 'val'), img_size=(224, 224), mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)):
    """
    train 혹은 validation의 augmentation 함수를 정의합니다. train은 데이터에 많은 변형을 주어야하지만, validation에는 최소한의 전처리만 주어져야합니다.
    
    Args:
        need: 'train', 혹은 'val' 혹은 둘 다에 대한 augmentation 함수를 얻을 건지에 대한 옵션입니다.
        img_size: Augmentation 이후 얻을 이미지 사이즈입니다.
        mean: 이미지를 Normalize할 때 사용될 RGB 평균값입니다.
        std: 이미지를 Normalize할 때 사용될 RGB 표준편차입니다.

    Returns:
        transformations: Augmentation 함수들이 저장된 dictionary 입니다. transformations['train']은 train 데이터에 대한 augmentation 함수가 있습니다.
    """
    transformations = {}
    if 'train' in need:
        transformations['train'] = Compose([
            Resize(img_size[0], img_size[1], p=1.0),
            HorizontalFlip(p=0.5),
            ShiftScaleRotate(p=0.5),
            HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            GaussNoise(p=0.5),
            Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)
    if 'val' in need:
        transformations['val'] = Compose([
            Resize(img_size[0], img_size[1]),
            Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)
    return transformations


def getDataloader(dataset, train_idx, valid_idx, batch_size, num_workers):
    # 인자로 전달받은 dataset에서 train_idx에 해당하는 Subset 추출
    train_set = torch.utils.data.Subset(dataset,
                                        indices=train_idx)
    # 인자로 전달받은 dataset에서 valid_idx에 해당하는 Subset 추출
    val_set   = torch.utils.data.Subset(dataset,
                                        indices=valid_idx)
    
    # 추출된 Train Subset으로 DataLoader 생성
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        shuffle=True
    )
    # 추출된 Valid Subset으로 DataLoader 생성
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        shuffle=False
    )
    # 생성한 DataLoader 반환
    return train_loader, val_loader



def train(data_dir, model_dir, args):
    start_time = time.time()
    seed_everything(args.seed)                                                                                  # seed 고정
    
    # --test data set
    # /opt/ml/input/data/train
    #test_img_root = './input/data/eval/'
    test_img_root = '../data/eval/'
    #submission = pd.read_csv(os.path.join('./input/data/eval/', 'info.csv'))
    submission = pd.read_csv(os.path.join('../data/eval/', 'info.csv'))
    image_dir = os.path.join(test_img_root, 'images')
    # Test Dataset 클래스 객체를 생성하고 DataLoader를 만듭니다.
    image_paths = [os.path.join(image_dir, img_id) for img_id in submission.ImageID]
    test_dataset = TestDataset(image_paths, resize=(224, 224))                      # vit size

    test_loader = DataLoader(
        test_dataset,
        shuffle=False
    )
    
    patience = 5
    counter = 0
    accumulation_steps = 2
    oof_pred = None
    
    
    # 5-fold Stratified KFold 5개의 fold를 형성하고 5번 Cross Validation을 진행합니다.
    n_splits = 4
    skf = StratifiedKFold(n_splits=n_splits)

    save_dir = increment_path(os.path.join(model_dir, args.name))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    #dataset_module = getattr(import_module("dataset"), args.dataset)  # default: MaskBaseDataset
    dataset_module = getattr(import_module("dataset2"), args.dataset)  # default: MaskBaseDataset
    dataset = dataset_module(
        data_dir=data_dir,
    )
    num_classes = dataset.num_classes  # 18

    transform = get_transforms(mean=dataset.mean, std=dataset.std)
    
    #dataset.set_transform(transform)
    # -- data_loader
    train_set, val_set = dataset.split_dataset()
    
    train_set.dataset.set_transform(transform["train"])
    val_set.dataset.set_transform(transform["val"])
    
    # -- model
    model_module = getattr(import_module("model"), args.model)          # default: BaseModel
    model = model_module(
        num_classes=num_classes
    ).to(device)
    model = torch.nn.DataParallel(model)
    
    

    # -- loss & metric
    criterion = create_criterion(args.criterion)                        # default: cross_entropy
    opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: Adam
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=5e-4
    )
    scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)
    #scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=10, metrics='')
    #scheduler = CosineAnnealingLR(optimizer, T_max=2, eta_min=0.)

    # -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    best_val_acc = 0
    best_val_loss = np.inf
    labels = [dataset.encode_multi_class(mask, gender, age) for mask, gender, age in zip(dataset.mask_labels, dataset.gender_labels, dataset.age_labels)]
    
    for fold, (train_idx, valid_idx) in enumerate(skf.split(dataset.image_paths, labels)):
        
        print(f'Fold {fold + 1}')
        
        train_loader, val_loader = getDataloader(dataset, train_idx, valid_idx, args.batch_size, num_workers=multiprocessing.cpu_count() // 2)
        
    
        for epoch in range(args.epochs):
            # train loop
            model.train()
            loss_value = 0
            matches = 0
            #logger = SummaryWriter(log_dir=f"results/cv{i}_{name}")
            for idx, train_batch in enumerate(train_loader):
                inputs, labels = train_batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                #optimizer.zero_grad()

                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)
                loss = criterion(outs, labels)

                loss.backward()

                # -- Gradient Accumulation
                if (idx+1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                #optimizer.step()

                f1_label = labels.detach().cpu()
                f1_pred = preds.detach().cpu()

                # f1 score 계산
                f1_score = calc_f1score(f1_label, f1_pred)          

                loss_value += loss.item()
                matches += (preds == labels).sum().item()
                if (idx + 1) % args.log_interval == 0:
                    train_loss = loss_value / args.log_interval
                    train_acc = matches / args.batch_size / args.log_interval
                    current_lr = get_lr(optimizer)
                    print(
                        f"Epoch[{epoch+1}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                        f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr} ||"
                        f"f1_score : {f1_score:4.4}"
                    )
                    logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
                    logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)

                    loss_value = 0
                    matches = 0

            scheduler.step()

            # val loop
            with torch.no_grad():
                print("Calculating validation results...")
                model.eval()
                val_loss_items = []                         # val_loss list
                val_acc_items = []                          # val_acc list
                figure = None
                for val_batch in val_loader:
                    inputs, labels = val_batch
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outs = model(inputs)
                    preds = torch.argmax(outs, dim=-1)

                    loss_item = criterion(outs, labels).item()
                    acc_item = (labels == preds).sum().item()
                    val_loss_items.append(loss_item)
                    val_acc_items.append(acc_item)

                    if figure is None:
                        inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                        inputs_np = dataset_module.denormalize_image(inputs_np, dataset.mean, dataset.std)
                        figure = grid_image(
                            inputs_np, labels, preds, n=16, shuffle=args.dataset != "MaskSplitByProfileDataset"
                        )

                val_loss = np.sum(val_loss_items) / len(val_loader)             # val_loss의 평균
                val_acc = np.sum(val_acc_items) / len(valid_idx)                 # val_acc의 평균
                
                best_val_loss = min(best_val_loss, val_loss)
                
                if val_acc > best_val_acc:
                    print(f"New best model for val accuracy : {val_acc:4.2%}! saving the best model..")
                    torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
                    best_val_acc = val_acc
                    counter = 0
                else:
                    counter += 1

                if counter > patience:
                    print("Early Stopping...")
                    break
                
                
                torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
                print(
                    f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                    f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
                )
                logger.add_scalar("Val/loss", val_loss, epoch)
                logger.add_scalar("Val/accuracy", val_acc, epoch)
                logger.add_figure("results", figure, epoch)
                print()

                
                
        all_predictions = []
        with torch.no_grad():
            for images in test_loader:
                images = images.to(device)
                # Test Time Augmentation
                pred = model(images) / 2 # 원본 이미지를 예측하고
                pred += model(torch.flip(images, dims=(-1,))) / 2 # horizontal_flip으로 뒤집어 예측합니다. 
                all_predictions.extend(pred.cpu().numpy())
            fold_pred = np.array(all_predictions)

        if oof_pred is None:
            oof_pred = fold_pred / n_splits
        else:
            oof_pred += fold_pred / n_splits
        
    submission['ans'] = np.argmax(oof_pred, axis=1)
    submission.to_csv(os.path.join(f'./model/{args.name}', 'submission.csv'), index=False)           # 경로 변경

    print('test inference is done!')
            
        
                
    print(f"총 학습 시간 {time.time() - start_time}")
        
            
            
            


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 1)')
    parser.add_argument('--dataset', type=str, default='MaskBaseDataset', help='dataset augmentation type (default: MaskBaseDataset)')
    #parser.add_argument('--dataset', type=str, default='MaskSplitByProfileDataset', help='dataset augmentation type (default: MaskSplitByProfileDataset)')
    #parser.add_argument('--augmentation', type=str, default='MyAugmentation', help='data augmentation type (default: BaseAugmentation)')
    #parser.add_argument('--augmentation', type=str, default='BaseAugmentation', help='data augmentation type (default: CustomAugmentation)')
    #parser.add_argument("--resize", nargs="+", type=list, default=[128, 96], help='resize size for image when training')
    parser.add_argument("--resize", nargs="+", type=list, default=[224, 224], help='resize size for image when training')          # vit
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    #parser.add_argument('--valid_batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    parser.add_argument('--valid_batch_size', type=int, default=64, help='input batch size for validing (default: 64)')
    #parser.add_argument('--model', type=str, default='Custom_EfficientNet', help='model type (default: BaseModel)')
    parser.add_argument('--model', type=str, default='Timm_vit', help='model type (default: Custom_Vit_b_16)')
    parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer type (default: SGD)')
    #parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type (default: SGD)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    #parser.add_argument('--lr', type=float, default=1e-2, help='learning rate (default: 1e-2)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')
    #parser.add_argument('--criterion', type=str, default='f1', help='criterion type (default: f1)')
    parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)
