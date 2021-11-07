import os
import pandas as pd
import numpy as np
import json

from mmcv import Config
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmcv.runner import load_checkpoint

import torch.nn.functional as F
import torch
from tqdm import tqdm
import albumentations as A

def load_dataset(cfg) :
    # dataset config 수정
    root= '/opt/ml/segmentation/input/data/mmseg/test'
    cfg.data.test.img_dir = root
    cfg.data.test.pipeline[1]['img_scale'] = [(512, 512), (448, 448), (576, 576), 
                                            (640, 640), (704, 704), (768, 768)]
    cfg.data.test.pipeline[1]['flip'] = True
    cfg.data.test.test_mode = True

    # build dataset
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
            dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False)    

    return data_loader , dataset.CLASSES

def load_models(config_names,checkpoint_pathes,classes) :
    # load model 
    models = []
    for config_name, checkpoint_path in zip(config_names, checkpoint_pathes):
        cfg = Config.fromfile(config_name)
        model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
        checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')

        model.CLASSES = classes
        model.eval();
        model = model.to('cuda')
        models.append(model)

    return models

def save(preds_array, name) :
    # sample_submisson.csv 열기
    submission = pd.DataFrame({"image_id":[],"PredictionString":[]})
    json_dir = os.path.join("/opt/ml/segmentation/input/data/test.json")
    with open(json_dir, "r", encoding="utf8") as outfile:
        datas = json.load(outfile)

    # PredictionString 대입
    for image_id, predict in enumerate(preds_array):
        image_id = datas["images"][image_id]
        file_name = image_id["file_name"]
        
        submission = submission.append({"image_id" : file_name, "PredictionString" : ' '.join(str(e) for e in predict.tolist())}, 
                                    ignore_index=True)

    # submission.csv로 저장
    submission.to_csv(os.path.join('/opt/ml/segmentation', f'{name}.csv'), index=False)
    print(f"Save {name}.csv")

def ensemble(config_names, checkpoint_pathes, name='ENSEMBLE') :
    # config file 들고오기
    cfg = Config.fromfile(config_names[0])
    
    data_loader,classes = load_dataset(cfg)
    models = (config_names,checkpoint_pathes,classes)


    size = 256
    transform = A.Compose([A.Resize(size, size)])
    preds_array = np.empty((0, size*size), dtype=np.long)
    
    print('Start prediction.')
    with torch.no_grad():
        for data in tqdm(data_loader) :
            softs = []
            for model_i , model in enumerate(models):
                # beit can't use multi scale
                if model_i == 0 :
                    imgs = [d.to('cuda') for d in data['img'][:2]]
                    img_metas = [d.data[0] for d in data['img_metas'][:2]]
                else :
                    imgs = [d.to('cuda') for d in data['img']]
                    img_metas = [d.data[0] for d in data['img_metas']]
                        
                logit = model(imgs,img_metas,return_loss=False)
                soft = F.softmax(logit, dim=1).detach().cpu().numpy().astype(np.float16)
                softs.append(soft)
            soft_sum = sum(softs)
            oms = np.argmax(soft_sum,1)
            
            # resize (256 x 256)
            temp_mask = []
            for img, mask in zip(np.stack(imgs[0].detach().cpu().numpy()), oms):
                transformed = transform(image=img, mask=mask)
                mask = transformed['mask']
                temp_mask.append(mask)

            oms = np.array(temp_mask)

            oms = oms.reshape([oms.shape[0], size*size]).astype(int)
            preds_array = np.vstack((preds_array, oms))

    print("End prediction.")
    save(preds_array,name)

if __name__ == '__main__' : 
    config_names = [
    '/opt/ml/segmentation/mmsegmentation/work_dirs/05_upernet_final0/upernet_beit_large_all.py',
    '/opt/ml/segmentation/mmsegmentation/work_dirs/11_segformer_swin_mixed0/11_segformer_swin_large_mixed_aug_all.py',
    '/opt/ml/segmentation/mmsegmentation/work_dirs/12_segformer_swin_all0/12_segformer_swin_large.py',

    ]
    checkpoint_pathes = [
        '/opt/ml/segmentation/mmsegmentation//work_dirs/05_upernet_final0/best_mIoU_epoch_14.pth',
        '/opt/ml/segmentation/mmsegmentation//work_dirs/11_segformer_swin_mixed0/best_mIoU_epoch_18.pth',
        '/opt/ml/segmentation/mmsegmentation//work_dirs/12_segformer_swin_all0/best_mIoU_epoch_18.pth'
    ]

    ensemble(config_names,checkpoint_pathes,name='ENSEMBLE')