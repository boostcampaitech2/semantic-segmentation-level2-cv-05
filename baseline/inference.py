import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import collections
import albumentations as albu

from parse_config import ConfigParser
import datasets as module_dataset
import models as module_model
import transforms as module_transform

# collate_fn needs for batch
def collate_fn(batch):
    return tuple(zip(*batch))

def test(model, test_loader, device):
    size = 256
    transform = albu.Compose([albu.Resize(size, size)])
    print('Start prediction.')
    
    model.eval()
    
    file_name_list = []
    preds_array = np.empty((0, size*size), dtype=np.int64)
    with torch.no_grad():
        for step, (imgs, image_infos) in enumerate(tqdm(test_loader)):
            
            # inference (512 x 512)
            outs = model(torch.stack(imgs).to(device))
            oms = torch.argmax(outs.squeeze(), dim=1).detach().cpu().numpy()
            
            # resize (256 x 256)
            temp_mask = []
            for img, mask in zip(np.stack(imgs), oms):
                transformed = transform(image=img, mask=mask)
                mask = transformed['mask']
                temp_mask.append(mask)
                
            oms = np.array(temp_mask)
            
            oms = oms.reshape([oms.shape[0], size*size]).astype(int)
            preds_array = np.vstack((preds_array, oms))
            
            file_name_list.append([i['file_name'] for i in image_infos])
    print("End prediction.")
    file_names = [y for x in file_name_list for y in x]
    
    return file_names, preds_array

def main(config):
    # Set model
    model = config.init_obj('model', module_model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Load weight
    ckpt = torch.load(config['ckpt'])
    model.load_state_dict(ckpt['model'])

    test_dataset = config.init_obj('test_dataset', module_dataset)
    test_transform = config.init_obj('test_transform', module_transform)
    test_dataset.set_transform(test_transform)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                            batch_size=4,
                                            shuffle=False,
                                            num_workers=4,
                                            collate_fn=collate_fn)
    print(test_loader)

    # submission format
    submission = pd.DataFrame({"image_id":[],"PredictionString":[]})

    # test set에 대한 prediction
    file_names, preds = test(model, test_loader, device)

    # PredictionString 대입
    for file_name, string in zip(file_names, preds):
        submission = submission.append({"image_id" : file_name, "PredictionString" : ' '.join(str(e) for e in string.tolist())}, 
                                    ignore_index=True)

    # submission.csv로 저장
    submission.to_csv(f"./submission/{config['name']}.csv", index=False)

if __name__ == "__main__":
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