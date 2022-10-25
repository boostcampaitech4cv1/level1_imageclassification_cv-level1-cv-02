MASK_CLASS = ['Wear','Incorrect','NotWear']
GENDER_CLASS = ['Male', 'Female']
AGE_CLASS = ['X<30', '30<=X<60', '60<=X']

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
