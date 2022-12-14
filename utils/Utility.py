import random
import os
import numpy as np
import pandas as pd
import datetime
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

MASK_CLASS = ['Wear', 'Incorrect', 'NotWear']
GENDER_CLASS = ['Male', 'Female']
AGE_CLASS = ['X<30', '30<=X<60', '60<=X']


def splitDataset(dataset: pd.DataFrame, validation_ratio: float = 0.2, random_state: int = 0, shuffle: bool = True, group_key: str = "id"):
    df_train_data = []
    df_validation_data = []
    ids = dataset[group_key].unique()
    if (shuffle):
        random.Random(random_state).shuffle(ids)
    ratio_validation = int(ids.size * validation_ratio)
    idxs_validation = ids[0:ratio_validation]
    idxjudge = np.zeros_like(ids, np.bool8)
    idxjudge[idxs_validation] = True
    for row in dataset.values:
        if idxjudge[row[0]]:
            df_validation_data.append(row)
        else:
            df_train_data.append(row)
    df_train = pd.DataFrame(df_train_data, columns=dataset.columns)
    df_validation = pd.DataFrame(df_validation_data, columns=dataset.columns)
    return df_train, df_validation


def setSeedEverything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def calcF1Score(true, pred):
    return f1_score(true, pred, average="macro")


def convertImagePathToMaskStatus(path: str):
    file_name = path.split('/')[-1].split('.')[0]
    if file_name == 'normal':
        return MASK_CLASS[2]
    elif file_name == 'incorrect_mask':
        return MASK_CLASS[1]
    elif file_name[0:4] == 'mask':
        return MASK_CLASS[0]
    else:
        return 'ERROR'


def encodeMask(status: str):
    return MASK_CLASS.index(status)


def decodeMask(status: int):
    return MASK_CLASS[status]


def encodeGender(status: str):
    return GENDER_CLASS.index(status)


def decodeGender(status: int):
    return GENDER_CLASS[status]


def encodeAge(status: int or str):
    if type(status) == int:
        if status < 30:
            index = 0
        elif 30 <= status < 60:
            index = 1
        elif status >= 60:
            index = 2
        else:
            raise Exception('age out of range(age is too big or small)')
        return index

    elif type(status) == str:
        return AGE_CLASS.index(status)  # type: ignore

    else:
        raise ValueError


def decodeAge(status: int):
    return AGE_CLASS[status]


def saveModel(model, optimizer, args, datetime, path):
    torch.save({
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'save_time': datetime,
    }, path)


def loadModel(path: str):
    obj = torch.load(path)
    return obj['model'], obj['optim'], obj['args'], obj['save_time']


def onlyLoadModel(model, path: str, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt)
    return model


class Args():
    def __init__(
        self,
        root_path,
        random_seed,
        csv_path,
        lr,
        batch_size,
        epochs,
        device,
        img_size,
        save_dir
    ):
        self.root_path = root_path
        self.random_seed = random_seed
        self.csv_path = csv_path
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        self.img_size = img_size
        self.save_dir = save_dir


def convertAgeGenderMaskToLabel(mask_label: int, gender_label: int, age_label: int) -> int:
    return mask_label*6 + gender_label*3 + age_label

def convertLabelToMaskGenderAge(label: int) -> "tuple[int, int, int]" :
    mask_label = label // 6
    label = label % 6
    gender_label = label // 3
    label = label % 3
    age_label = label
    return mask_label, gender_label, age_label

def combineAgeGenderMaskSubmission(mask_fn: str, gender_fn: str, age_fn: str):
    """
    mask_fn : mask_{version}
    gender_fn : gender_{version}
    age_fn : age_{version}
    """
    csv_path = os.path.join('../test', 'csv')
    csv_file_list = os.listdir(csv_path)

    for file_name in csv_file_list:
        if mask_fn in file_name:
            df_mask = pd.read_csv(os.path.join(csv_path, file_name))
        elif gender_fn in file_name:
            df_gender = pd.read_csv(os.path.join(csv_path, file_name))
        elif age_fn in file_name:
            df_age = pd.read_csv(os.path.join(csv_path, file_name))
    
    sum_of_sub = []
    for m,g,a in zip(df_mask.ans, df_gender.ans, df_age.ans):
        sum_of_sub.append(m*6 + g*3 + a)
    
    submission = pd.read_csv(os.path.join('../data/eval', 'info.csv'))
    submission['ans'] = sum_of_sub
    submission.to_csv(os.path.join('../', 'submission.csv'), index=False)


def randBbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2