import random
import os
import numpy as np
import torch
from sklearn.metrics import f1_score

MASK_CLASS = ['Wear','Incorrect','NotWear']
GENDER_CLASS = ['Male', 'Female']
AGE_CLASS = ['X<30', '30<=X<60', '60<=X']

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def competition_metric(true, pred):
    return f1_score(true, pred, average="macro")

def ConvertImagePathToMaskStatus(path:str):
    file_name = path.split('/')[-1].split('.')[0]
    if(file_name == 'normal'):
        return MASK_CLASS[2]
    elif(file_name == 'incorrect_mask'):
        return MASK_CLASS[1]
    elif(file_name[0:4] == 'mask'):
        return MASK_CLASS[0]
    else:
        return 'ERROR'

def MaskEncoder(status:str):
    return MASK_CLASS.index(status)

def MaskDecoder(status:int):
    return MASK_CLASS[status]


def GenderEncoder(status:str):
    return GENDER_CLASS.index(status)

def GenderDecoder(status:int):
    return GENDER_CLASS[status]


def ageEncoder(status: int or str):
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

def AgeDecoder(status:int):
    return AGE_CLASS[status]


def mixUp(a, b):
    pass


class Args():
    def __init__(
        self,
        root_path,
        random_seed,
        csv_path,
        seed,
        lr,
        batch_size,
        epochs
    ):
        self.root_path = root_path
        self.random_seed= random_seed
        self.csv_path = csv_path
        self.seed = seed
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs