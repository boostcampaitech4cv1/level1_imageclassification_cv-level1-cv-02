import os

os.system("wget https://aistages-prod-server-public.s3.amazonaws.com/app/Competitions/000206/data/train.tar.gz")
os.system("wget https://aistages-prod-server-public.s3.amazonaws.com/app/Competitions/000206/data/code.tar.gz")
os.system("mkdir ./data")
os.system("tar -zxvf ./train.tar.gz --directory=./data")
os.system("tar -zxvf ./code.tar.gz --directory='./data")
