import os
import glob
import cv2
from PIL import Image
import random
import numpy as np
import albumentations as A

from ..builder import PIPELINES

CATEGORIES = ['Backgroud', 'General trash', 'Paper', 'Paper pack', 'Metal',
                  'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing']
GET_TRANSFORM = {
    3 : A.Rotate(p=0.5),    # Paper pack
    4 : A.Compose([A.Rotate(p=0.5, border_mode=0),  # Metal
                    A.RGBShift(p=0.3)]),
    5 : A.Compose([A.Rotate(limit=10, p=0.2, border_mode=0),    # Glass
                    A.RGBShift(p=0.2)]),
    9 : A.Compose([A.Rotate(p=0.5, border_mode=0),  # Battery
                    A.HueSaturationValue(p=0.2)]),
    10 : A.Compose([A.ElasticTransform(alpha=1, sigma=50, alpha_affine=40, border_mode=0, p=0.7),   # Clothing
                    A.Rotate(p=0.5, border_mode=0),
                    A.RGBShift(10,10,10,p=0.7)]),                        
}


@PIPELINES.register_module()
class CopyPaste(object):
    """Random copy&paste

    Args:
        category (int): The label of category.
        p (float): The probability of augmentation.
        max_loop (int): The number of times to find the optimal location. 
            (It find a location that has less overlap with other instances.)
        max_size ((int, int)): The max size of instance being pasted.
        root (str): The root path of instance images and masks.
                ├── root
                │   ├── (category_name)
                │   │   ├── images
                │   │   │   ├── xxx.png
                │   │   │   ├── yyy.png
                │   │   │   ├── zzz.png
                │   │   ├── masks
                │   │   │   ├── xxx.png
                │   │   │   ├── yyy.png
                │   │   │   ├── zzz.png

    """

    def __init__(self, category=5, p=0.05, max_loop=5, max_size=(128, 256), root="/opt/ml/segmentation/input/data/mask_instance_image/"):
        self.p = p
        self.max_loop= max_loop
        self.max_size = max_size

        files = []
        self.category = category
        _files = glob.glob(root+CATEGORIES[self.category]+"/images/*.png")
        files += [os.path.basename(f) for f in _files]
        
        self.mask_imgs = []
        self.masks = []
        for file in files:
            mask_img = cv2.imread(root+CATEGORIES[self.category]+"/images/"+file)
            mask = np.array(Image.open(root+CATEGORIES[self.category]+"/masks/"+file))
            self.mask_imgs.append(mask_img)
            self.masks.append(mask)
        self.length = len(files)
        self.img_inds = random.sample(range(self.length), self.length)

        self.transform = GET_TRANSFORM[category]

    def _get_mask_imgs(self):
        if self.img_inds:
            idx = self.img_inds.pop()
        else:
            self.img_inds = random.sample(range(self.length), self.length)
            idx = self.img_inds.pop()
        return self.mask_imgs[idx], self.masks[idx]

    def _crop_instance(self, img, mask, pad=False):
        # argwhere will give you the coordinates of every non-zero point
        true_points = np.argwhere(img)
        # take the smallest points and use them as the top left of your crop
        top_left = true_points.min(axis=0)
        # take the largest points and use them as the bottom right of your crop
        bottom_right = true_points.max(axis=0)
        img = img[top_left[0]:bottom_right[0]+1,  # plus 1 because slice isn't
                top_left[1]:bottom_right[1]+1]  # inclusive
        mask = mask[top_left[0]:bottom_right[0]+1,  # plus 1 because slice isn't
                top_left[1]:bottom_right[1]+1]  # inclusive
        if pad:
            h = bottom_right[0]-top_left[0]+1
            w = bottom_right[1]-top_left[1]+1
            h_pad = int((max(h,w)*(2**(1/2)) - h)//2 +1)
            w_pad = int((max(h,w)*(2**(1/2)) - w)//2 +1)
            new_img = np.zeros((h+h_pad*2, w+w_pad*2, 3), dtype=np.uint8)
            new_mask = np.zeros((h+h_pad*2, w+w_pad*2), dtype=np.uint8)
            new_img[h_pad:-h_pad, w_pad:-w_pad] = img
            new_mask[h_pad:-h_pad, w_pad:-w_pad] = mask
            return new_img, new_mask
        return img, mask

    def __call__(self, results):
        """Call function to randomly copy&paste instance.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly augmented results.
        """
        if random.random() > self.p:
            return results
        mask_img, mask = self._get_mask_imgs()
        img_x, img_y = results['img_shape'][:2]


        if self.transform:
            mask_img, mask = self._crop_instance(mask_img, mask, pad=True)
            transformed = self.transform(image=mask_img, mask=mask)
            mask_img = transformed['image']
            mask = transformed['mask']
        

        mask_img, mask = self._crop_instance(mask_img, mask)

        x,y,_ = mask_img.shape
        ratio = x/y

        max_size = random.randint(*self.max_size)
        if ratio > 1.2:
            max_size = min(max_size, int(x*1))
            new_x, new_y = (max_size, max_size/ratio)
        else:
            max_size = min(max_size, int(y*1))
            new_x, new_y = (max_size*ratio, max_size)

        mask_img = cv2.resize(mask_img, dsize=(int(new_y), int(new_x)))
        mask = cv2.resize(mask, dsize=(int(new_y), int(new_x)), interpolation=cv2.INTER_NEAREST)
        x,y,_ = mask_img.shape

        img = results['img']
        gt_semantic_seg = results['gt_semantic_seg']

        loop = 0
        best_overlap = 1
        while(loop<self.max_loop):
            # pos_x = random.randint(64,512-64-x)
            # pos_y = random.randint(64,512-64-y)
            pos_x = random.randint(0,img_x-x)
            pos_y = random.randint(0,img_y-y)
            if (pos_x+(x//2) > 200 and pos_x+(x//2) <312) or (pos_y+(y//2) > 200 and pos_y+(y//2) <312):
                continue

            new_mask = np.zeros((img_x, img_y), dtype= np.uint8)
            new_mask[pos_x:pos_x+x, pos_y:pos_y+y] = mask

            tmp_overlap = gt_semantic_seg[new_mask>0]
            tmp_overlap = sum(tmp_overlap>0) / len(tmp_overlap)
            if tmp_overlap == 0:
                best_pos_x = pos_x
                best_pos_y = pos_y
                break
            if tmp_overlap <= best_overlap:
                best_overlap = tmp_overlap
                best_pos_x = pos_x
                best_pos_y = pos_y


            loop += 1


        new_mask = np.zeros((img_x, img_y), dtype= np.uint8)
        new_mask[best_pos_x:best_pos_x+x, best_pos_y:best_pos_y+y] = mask

        new_mask_img = np.zeros((img_x,img_y,3), dtype=np.uint8)
        new_mask_img[best_pos_x:best_pos_x+x, best_pos_y:best_pos_y+y] = mask_img

        img[new_mask>0] = new_mask_img[new_mask>0]
        results['img'] = img

        gt_semantic_seg[new_mask>0] = self.category
        results['gt_semantic_seg'] = gt_semantic_seg
    

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(copy_paste={CATEGORIES[self.category]})'

@PIPELINES.register_module()
class MultiCopyPaste(object):
    """MultiCopyPaste

    """

    def __init__(self, categories=[5, 9, 10], p=0.05, max_loop=5, max_size=(128, 256), root="/opt/ml/segmentation/input/data/mask_instance_image/"):
        self.categories = categories

        if not isinstance(p, list):
            probs = [p] * len(categories)
        else:
            probs = p
        if not isinstance(max_loop, list):
            max_loops = [max_loop] * len(categories)
        else:
            max_loops= max_loop
        if not isinstance(max_size, list):
            max_sizes = [max_size] * len(categories)
        else:
            max_sizes= max_size

        self.copypastes = []
        for category, prob, max_loop, max_size in zip(categories, probs, max_loops, max_sizes):
            self.copypastes.append(CopyPaste(category, prob, max_loop, max_size, root))

    def __call__(self, results):
        """Call multi copypastes randomly ordered.

        """
        order = random.sample(self.copypastes, len(self.copypastes))
        for _ in range(len(order)):
            copypaste = order.pop()
            results = copypaste(results)

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(copy_paste={CATEGORIES[self.categories]})'