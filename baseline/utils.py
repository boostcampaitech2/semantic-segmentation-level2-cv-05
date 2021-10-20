# # https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

import numpy as np
import torch
import json
from pathlib import Path
from collections import OrderedDict
import os
import random


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(n_class * label_true[mask].astype(int) + label_pred[mask],
                        minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def label_accuracy_score(hist):
    """
    Returns accuracy score evaluation result.
      - [acc]: overall accuracy
      - [acc_cls]: mean accuracy
      - [mean_iu]: mean IU
      - [fwavacc]: fwavacc
    """
    acc = np.diag(hist).sum() / hist.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)

    with np.errstate(divide='ignore', invalid='ignore'):
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)

    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc, iu


def add_hist(hist, label_trues, label_preds, n_class):
    """
        stack hist(confusion matrix)
    """

    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)

    return hist




def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

    
def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def set_seed(seed) :
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)

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

def save_model(model, optimizer, lr_scheduler, saved_dir, file_name='best.pt'):

    if not os.path.isdir(saved_dir):                                                           
        os.mkdir(saved_dir)
    check_point = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
    output_path = os.path.join(saved_dir, file_name)
    torch.save(check_point, output_path)

def load_weight(ckpt_file, model, optimizer, lr_scheduler):
    check_point = torch.load(ckpt_file)
    model.load_state_dict(check_point['model'])
    optimizer.load_state_dict(check_point['optimizer'])
    return model, optimizer

# visualize

CLASS_COLORMAP= [['Backgroud', 0, 0, 0],
                ['General trash', 192, 0, 128],
                ['Paper', 0, 128, 192],
                ['Paper pack', 0, 128, 64],
                ['Metal', 128, 0, 0],
                ['Glass', 64, 0, 128],
                ['Plastic', 64, 0, 192],
                ['Styrofoam', 192, 128, 64],
                ['Plastic bag', 192, 192, 128],
                ['Battery', 64, 64, 128],
                ['Clothing', 128, 0, 192]]

def create_trash_label_colormap():
    """Creates a label colormap used in Trash segmentation.
    Returns:
        A colormap for visualizing segmentation results.
    """
    colormap = np.zeros((11, 3), dtype=np.uint8)
    for inex, (_, r, g, b) in enumerate(CLASS_COLORMAP):
        colormap[inex] = [r, g, b]
    
    return colormap

def label_to_color_image(label):
    """Adds color defined by the dataset colormap to the label.

    Args:
        label: A 2D array with integer type, storing the segmentation label.

    Returns:
        result: A 2D array with floating type. The element of the array
                is the color indexed by the corresponding element in the input label
                to the trash color map.

    Raises:
        ValueError: If label is not of rank 2 or its value is larger than color
              map maximum entry.
    """
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = create_trash_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]
