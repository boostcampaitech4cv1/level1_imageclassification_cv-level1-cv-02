import random
import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

MASK_CLASS = ['Wear', 'Incorrect', 'NotWear']
GENDER_CLASS = ['Male', 'Female']
AGE_CLASS = ['X<30', '30<=X<60', '60<=X']


def splitDataset(dataset: pd.DataFrame, validation_ratio: float = 0.2, random_state: int = 0, shuffle: bool = True, group_key: str = "id"):
    df_train_data = []
    df_validation_data = []
    ids = dataset['id'].unique()
    if (shuffle):
        random.Random(random_state).shuffle(ids)
    ratio_validation = int(ids.size * 0.2)
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
        return AGE_CLASS.index(status)

    else:
        raise ValueError


def decodeAge(status: int):
    return AGE_CLASS[status]


def mixUp(a, b):
    pass


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
        img_size
    ):
        self.root_path = root_path
        self.random_seed= random_seed
        self.csv_path = csv_path
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        self.img_size = img_size
