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

def inference(config,weight,work_dir) :
    # config file 들고오기
    cfg = Config.fromfile(config)

    root= '/opt/ml/segmentation/input/data/mmseg/test'
    # dataset config 수정
    cfg.data.test.img_dir = root
    cfg.data.test.pipeline[1]['img_scale'] = (512,512) # Resize
    cfg.data.test.pipeline[1]['flip'] = True 
    cfg.data.test.test_mode = True

    cfg.work_dir = work_dir

    cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)

    # checkpoint path
    checkpoint_path = os.path.join(cfg.work_dir, f'{weight}.pth')

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
            dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False)

    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')

    model.CLASSES = dataset.CLASSES
    model.eval()
    model = model.to('cuda')



    size = 256
    transform = A.Compose([A.Resize(size, size)])
    preds_array = np.empty((0, size*size), dtype=np.long)
    
    print('Start prediction.')
    with torch.no_grad():
        for step, data in enumerate(tqdm(data_loader)):
            
            imgs = [d.to('cuda') for d in data['img']]
            img_metas = [d.data[0] for d in data['img_metas']]
            logit = model(imgs,img_metas,return_loss=False)

            # inference (512 x 512)
            oms = torch.argmax(logit, dim=1).detach().cpu().numpy()

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

    # sample_submisson.csv 열기
    submission = pd.DataFrame({"image_id":[],"PredictionString":[]})
    json_dir = os.path.join("../input/data/test.json")
    with open(json_dir, "r", encoding="utf8") as outfile:
        datas = json.load(outfile)

    # PredictionString 대입
    for image_id, predict in enumerate(preds_array):
        image_id = datas["images"][image_id]
        file_name = image_id["file_name"]
        
        submission = submission.append({"image_id" : file_name, "PredictionString" : ' '.join(str(e) for e in predict.tolist())}, 
                                    ignore_index=True)

    # submission.csv로 저장
    submission.to_csv(os.path.join(cfg.work_dir, f'submission_{weight}.csv'), index=False)


if __name__ == '__main__' : 
    # config file 들고오기
    config = '/opt/ml/segmentation/mmsegmentation/work_dirs/05_upernet_final0/upernet_beit_large_all.py'
    weight = 'best_mIoU_epoch_14'
    work_dir = 'work_dirs/05_upernet_final0/'

    inference(config,weight,work_dir)