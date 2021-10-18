import os
import random
import time
import json
import warnings 
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import label_accuracy_score, add_hist
from pathlib import Path

from dataset import CustomDataLoader, category_names

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

import argparse

# print('pytorch version: {}'.format(torch.__version__))
# print('GPU 사용 가능 여부: {}'.format(torch.cuda.is_available()))

# print(torch.cuda.get_device_name(0))
# print(torch.cuda.device_count())


def set_seed(seed) :
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

# collate_fn needs for batch
def collate_fn(batch):
    return tuple(zip(*batch))

def save_model(model, saved_dir, file_name='fcn_resnet50_best_model(pretrained).pt'):

    if not os.path.isdir(saved_dir):                                                           
        os.mkdir(saved_dir)
    check_point = {'net': model.state_dict()}
    output_path = os.path.join(saved_dir, file_name)
    torch.save(model, output_path)

def train(num_epochs, model, train_loader, val_loader, criterion, optimizer, saved_dir, val_every, device):
    print(f'Start training..')
    n_class = 11
    best_loss = 9999999
    # GPU 사용 가능 여부에 따라 device 정보 저장

    for epoch in range(num_epochs):
        model.train()

        hist = np.zeros((n_class, n_class))
        for step, (images, masks, _) in enumerate(train_loader):
            images = torch.stack(images)       
            masks = torch.stack(masks).long() 
            
            # gpu 연산을 위해 device 할당
            images, masks = images.to(device), masks.to(device)
            
            # device 할당
            model = model.to(device)
            
            # inference
            outputs = model(images)['out']
            
            # loss 계산 (cross entropy loss)
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            masks = masks.detach().cpu().numpy()
            
            hist = add_hist(hist, masks, outputs, n_class=n_class)
            acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
            
            # step 주기에 따른 loss 출력
            if (step + 1) % 25 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{len(train_loader)}], \
                        Loss: {round(loss.item(),4)}, mIoU: {round(mIoU,4)}, \
                        Time: {time.strftime("%H:%M:%S")}')

        # validation 주기에 따른 loss 출력 및 best model 저장
        if (epoch + 1) % val_every == 0:
            avrg_loss = validation(epoch + 1, model, val_loader, criterion, device, saved_dir)
            if avrg_loss < best_loss:
                print(f"Best performance at epoch: {epoch + 1}")
                print(f"Save model in {saved_dir}")
                best_loss = avrg_loss
                save_model(model, saved_dir)

def validation(epoch, model, val_loader, criterion, device, saved_dir):
    print(f'Start validation #{epoch}')
    model.eval()

    with torch.no_grad():
        n_class = 11
        total_loss = 0
        cnt = 0
        
        hist = np.zeros((n_class, n_class))
        for step, (images, masks, _) in enumerate(tqdm(val_loader)):
            
            images = torch.stack(images)       
            masks = torch.stack(masks).long()  

            images, masks = images.to(device), masks.to(device)            
            
            # device 할당
            model = model.to(device)
            
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1
            
            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            masks = masks.detach().cpu().numpy()
            
            hist = add_hist(hist, masks, outputs, n_class=n_class)
        
        acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
        IoU_by_class = [{classes : round(IoU,4)} for IoU, classes in zip(IoU , category_names)]
        
        avrg_loss = total_loss / cnt
        print(f'Validation #{epoch}  Average Loss: {round(avrg_loss.item(), 4)}, Accuracy : {round(acc, 4)}, \
                mIoU: {round(mIoU, 4)}')
        print(f'IoU by class : {IoU_by_class}')

        f = open(f'{saved_dir}/valid.txt', 'a')
        f.write(f'Validation #{epoch}  Average Loss: {round(avrg_loss.item(), 4)}, Accuracy : {round(acc, 4)}, \
                mIoU: {round(mIoU, 4)}\n')
        f.write(f'{IoU_by_class}\n')
        f.close()
        
    return avrg_loss

def increment_path(path):   
    n = 0
    while True:
        path_ = Path(f"{path}{n}")
        if not path_.exists():
            break
        elif path_.exists():
            n += 1

    path_ = str(path_)
    path = ''
    for p in path_.split('/'):
        path += f'{p}/'
        if not Path(path).exists():
            os.mkdir(path)

    return path_

def main(args) :

    set_seed(args.seed)

    train_transform = A.Compose([
                            ToTensorV2()
                                ])

    val_transform = A.Compose([
                            ToTensorV2()
                            ])

    train_dataset = CustomDataLoader(data_dir=args.train_path, mode='train', transform=train_transform)
    if not args.noval :
        val_dataset = CustomDataLoader(data_dir=args.valid_path, mode='val', transform=val_transform)

    # DataLoader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            num_workers=4,
                                            collate_fn=collate_fn)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                            num_workers=4,
                                            collate_fn=collate_fn)



    model = models.segmentation.fcn_resnet50(pretrained=True)

    # output class를 data set에 맞도록 수정
    model.classifier[4] = nn.Conv2d(512, 11, kernel_size=1)

    # Loss function 정의
    criterion = nn.CrossEntropyLoss()

    # Optimizer 정의
    optimizer = torch.optim.Adam(params = model.parameters(), lr = args.lr, weight_decay=1e-6)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    saved_dir = os.path.join(args.saved_dir, 'exp')
    saved_dir = increment_path(saved_dir)

    train(args.epochs, model, train_loader, val_loader, criterion, optimizer, saved_dir, args.val_interval, device)

    
if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=16, help='set train batch size')
    parser.add_argument('--epochs', type=int, default=20, help='set train epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='set learing rate')
    parser.add_argument('--seed', type=int, default=2021, help='set random seed')
    parser.add_argument('--noval', action='store_true', help='only train')

    parser.add_argument('--train-path', type=str, default='/opt/ml/segmentation/input/data/train.json', help='train json path')
    parser.add_argument('--valid-path', type=str, default='/opt/ml/segmentation/input/data/val.json', help='valid json path')

    parser.add_argument('--saved_dir', type=str, default='./saved', help='model save path')
    parser.add_argument('--val_interval', type=int, default=1, help='set valid interval')

    args = parser.parse_args()
    print(args)
    main(args)
