import os
import random
import time
import collections
import wandb

import numpy as np
from tqdm import tqdm
from utils import label_accuracy_score, add_hist, label_to_color_image, increment_path, set_seed, save_model, load_weight
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models


import argparse
from parse_config import ConfigParser
from datasets import  category_names
import datasets as module_dataset
import models as module_model
import transforms as module_transform
import losses as module_loss


# collate_fn needs for batch
def collate_fn(batch):
    return tuple(zip(*batch))


def train(num_epochs, model, train_loader, val_loader, criterion, optimizer, lr_scheduler, saved_dir, val_every, device):
    print(f'Start training..')
    wandb.watch(model)

    n_class = 11
    best_loss = 9999999
    best_mIoU = 0

    for epoch in range(num_epochs):
        print()
        model.train()
        wandb.log({"learning_rate": optimizer.param_groups[0]['lr'], "epoch":epoch+1})
        hist = np.zeros((n_class, n_class))
        for step, (images, masks, _) in enumerate(train_loader):
            images = torch.stack(images)       
            masks = torch.stack(masks).long() 
            
            # gpu 연산을 위해 device 할당
            images, masks = images.to(device), masks.to(device)
            
            # inference
            outputs = model(images)
            
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
                wandb.log({"train/loss": loss.item(), "train/mIoU":mIoU, "epoch":epoch+1}, step=epoch*len(train_loader)+step)
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{len(train_loader)}], \
                        Loss: {round(loss.item(),4)}, mIoU: {round(mIoU,4)}, \
                        Time: {time.strftime("%H:%M:%S")}')

        # validation 주기에 따른 loss 출력 및 best model 저장
        if (epoch + 1) % val_every == 0:
            avrg_loss, mIoU = validation(epoch + 1, model, val_loader, criterion, device, saved_dir)
            if avrg_loss < best_loss:
                print(f"Best performance at epoch: {epoch + 1}")
                print(f"Save model in {saved_dir}")
                best_loss = avrg_loss
                save_model(model, optimizer, lr_scheduler, saved_dir, file_name='best_loss.pt')
            if mIoU > best_mIoU:
                print(f"Best mIoU performance at epoch: {epoch + 1}")
                print(f"Save model in {saved_dir}")
                best_mIoU = mIoU
                save_model(model, optimizer, lr_scheduler, saved_dir, file_name='best_mIoU.pt')
        lr_scheduler.step()
def validation(epoch, model, val_loader, criterion, device, saved_dir):
    print(f'Start validation #{epoch}')
    model.eval()
    
    with torch.no_grad():
        n_class = 11
        total_loss = 0
        cnt = 0
        
        hist = np.zeros((n_class, n_class))
        for step, (images, masks, infos) in enumerate(tqdm(val_loader)):
            
            images = torch.stack(images)       
            masks = torch.stack(masks).long()  

            images, masks = images.to(device), masks.to(device)            
            
            # device 할당
            model = model.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            cnt += 1
            
            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            masks = masks.detach().cpu().numpy()
            
            hist = add_hist(hist, masks, outputs, n_class=n_class)
        
        acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
        IoU_by_class = [{classes : round(IoU,4)} for IoU, classes in zip(IoU , category_names)]
        
        avrg_loss = total_loss / cnt
        print(f'Validation #{epoch}  Average Loss: {round(avrg_loss, 4)}, Accuracy : {round(acc, 4)}, \
                mIoU: {round(mIoU, 4)}')
        print(f'IoU by class : {IoU_by_class}')

        f = open(f'{saved_dir}/valid.txt', 'a')
        f.write(f'Validation #{epoch}  Average Loss: {round(avrg_loss, 4)}, Accuracy : {round(acc, 4)}, \
                mIoU: {round(mIoU, 4)}\n')
        f.write(f'{IoU_by_class}\n')
        f.close()
        valid_log = {"val/loss": avrg_loss, "val/mIoU":mIoU, "epoch":epoch+1}
        for iou, classes in zip(IoU , category_names):
            valid_log["val/IoU."+classes] = iou
        wandb.log(valid_log)


    return avrg_loss, mIoU


def main(config) :

    set_seed(config['seed'])

    train_dataset = config.init_obj('train_dataset', module_dataset)
    valid_dataset = config.init_obj('valid_dataset', module_dataset)

    train_transform = config.init_obj('train_transform', module_transform)
    valid_transform = config.init_obj('valid_transform', module_transform)

    train_dataset.set_transform(train_transform)
    valid_dataset.set_transform(valid_transform)

    # DataLoader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                            batch_size=config["batch_size"],
                                            shuffle=True,
                                            num_workers=4,
                                            collate_fn=collate_fn)

    val_loader = torch.utils.data.DataLoader(dataset=valid_dataset, 
                                            batch_size=config["batch_size"],
                                            shuffle=False,
                                            num_workers=4,
                                            collate_fn=collate_fn)

    # Set model
    model = config.init_obj('model', module_model)

    # Set Loss function
    criterion_dict = config["criterion"]
    criterion = getattr(module_loss, criterion_dict["type"])(**criterion_dict["args"])

    # Set Optimizer & Scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    if 'ckpt' in config.config.keys():
        load_weight(config['ckpt'], model, optimizer, lr_scheduler)
        print("success loading ckpt")

    saved_dir = config.save_dir
    NAME = config['name']

    # Set wandb
    wandb.init(project='Trash_Segmentation', entity='friends', config=config.config, name = NAME)
    wandb.define_metric("epoch")
    wandb.define_metric("learning_rate", step_metric="epoch")
    wandb.define_metric("val/*", step_metric="epoch")
    wandb.define_metric("val/mIoU", summary="max")
    
    train(config['epochs'], model, train_loader, val_loader, criterion, optimizer, lr_scheduler, saved_dir, config['val_interval'], device)

    
if __name__ == "__main__" :
    args = argparse.ArgumentParser()
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('--ckpt', default=None, type=str,
                      help='path to latest checkpoint (default: None)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
