MASK_CLASS = ['Wear','Incorrect','NotWear']
GENDER_CLASS = ['Male', 'Female']
AGE_CLASS = ['X<30', '30<=X<60', '60<=X']

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


def AgeEncoder(status:int or str):
    if(type(status) == int):
        index = -1
        if(status < 30):
            index = 0
        elif(status >= 30 and status < 60):
            index = 1
        elif(status >= 60):
            index = 2
        return index
    elif(type(status) == str):
        return AGE_CLASS.index(status)
    else:
        return "ERROR"

def AgeDecoder(status:int):
    return AGE_CLASS[status]
