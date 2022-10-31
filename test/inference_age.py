import argparse
import os
import sys
sys.path.append('../')
from utils.CustomDataset import CustomDataset

from tqdm import tqdm
from utils.Utility import onlyLoadModel
from models.Backbone import EfficientBackBone
import pandas as pd
import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

@torch.no_grad()
def inference(args):
    model = EfficientBackBone(3).to(args.device)
    model_path = os.path.join(args.ckpt_dir, '15_best_model_v1.pth')
    model = onlyLoadModel(model, model_path, args.device)
    model.eval()

    test_df = pd.read_csv(args.test_csv)

    transform = A.Compose([
        A.Normalize((0.560, 0.524, 0.501), (0.098, 0.096, 0.097)),
        A.Resize(args.resize[0], args.resize[1]),
        ToTensorV2(),
    ])
    
    test_dataset = CustomDataset(args.test_path, test_df.images.values, labels=None, transforms=transform)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=False
        )

    preds = []
    with torch.no_grad():
        for idx, data in enumerate(tqdm(test_loader)):
            images = data['image'].to(args.device)
            pred = model(images)
            pred = pred.argmax(dim=-1)
            preds.extend(pred.cpu().numpy())
        
    test_df['ans'] = preds
    save_path = os.path.join(args.save_dir, f'{args.csv_name}.csv')
    test_df.to_csv(save_path, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=500, help='input batch size for validing (default: 1000)')
    parser.add_argument('--resize', type=tuple, default=(224, 224), help='resize size')
    parser.add_argument('--ckpt_dir', type=str, default='../models/ckpt/')
    parser.add_argument('--save_dir', type=str, default='./csv/')
    parser.add_argument('--test_path', type=str, default='../data/eval/images/')
    parser.add_argument('--test_csv', type=str, default='../data/eval_i.csv')
    parser.add_argument('--csv_name', type=str, default='submission_age_v0')


    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    inference(args)