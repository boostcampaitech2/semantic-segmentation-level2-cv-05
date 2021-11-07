# semantic-segmentation-level2-cv-05


## Project 개요
- 목표 : 이미지에서 배경과 10종류 분리수거 쓰레기를 분류하기
  - Background 
  - General trash, Paper, Paper pack, Metal, Glass, Plastic, Styrofoam, Plastic bag, Battery, Clothing 10 종류의 쓰레기
- Data
  - Training Data : masking 되어있는 고유의 좌표와, 객체가 해당하는 카테고리를 포함한 2647장의 데이터
  - Validation Data : masking 되어있는 고유의 좌표와, 객체가 해당하는 카테고리를 포함한 655장의 데이터
  - Test Data : 무작위로 선정된 819장의 데이터
- Data 분석 
  - Mislabeling 문제 : Labeling이 잘못된 데이터 문제가 있었다.
  - Class 불균형 문제 : battery와 clothing 클래스가 다른 class에 비해 현저히 적은 데이터 불균형 문제가 있었다. 
  

## Table of Contents
1. [Train](#Train)
2. [Code Structure](#code-structure)
3. [Detail](#detail)
4. [Contributor](#contributor)


### Result
- Public mIoU score : 0.780 -> Private mIoU score: 0.766


## Getting Started
```bash
cd mmsegmentation
pip install -r requirements.txt
```

### Train
```bash
cd mmsegmentation
python tools/train.py configs/...
```

- 사용한 config 목록

| model                                | augment                              | config file                   |
|--------------------------------------|:------------------------------------:|:-----------------------------:|
| Upernet - BeiT (All data)            | Copy Paste, Grid Mask                |[config](https://github.com/boostcampaitech2/semantic-segmentation-level2-cv-05/blob/main/mmsegmentation/configs/upernet/upernet_beit_large_all.py)      
| segformer - swin large(All data)     | Copy Paste, Grid Mask                |[config](https://github.com/boostcampaitech2/semantic-segmentation-level2-cv-05/blob/main/mmsegmentation/configs/segformer/segformer_swin_large.py)|
| segformer - swin large (All data)    | Copy Paste, Grid Mask, Heavy aug, Mixed precision|[config](https://github.com/boostcampaitech2/semantic-segmentation-level2-cv-05/blob/main/mmsegmentation/configs/segformer/segformer_swin_large_mixed_aug_all.py)|




### Inference
```bash
cd mmsegmentation
python inference.py
python ensemble.py
```



## Code Structure
```
├── mmsegmentation                 # code from mmsegmentation
│   ├── tools/train.py             # to train
│   ├── inference.py               # to inference
│   └── ensemble.py                # to ensemble

```


## Detail
---
### Model
- <a href = 'https://github.com/open-mmlab/mmsegmentation'>mmsegmentation</a>를 기반으로 실험을 진행하였다.
- 효율적으로 ensemble을 하기 위해서 다양한 model을 사용하려고 노력하였다.



### Dataset Augmentation
- Copy Paste : Class 불균형 문제를 해결하기 위해 instance 단위로 잘라 이미지에 붙이는 작업을 했다.
Paper pack, Metal, Glass, Battery, Clothing 총 5개의 category를 적용시켰고, 각 category 별로 augmentation도 다양하게 주어 효과를 극대화 해주었다. 
- Cut out, Grid Mask, Heavy Aug, TTA(Flip, Multi Scale) : 일반화 성능을 올리기 위해 최대한 다양한 augmentation을 주려고 노력했다. Cut out과 Grid Mask는 성능 향상에 도움이 되었고, Heavy Aug의 경우에는 성능이 줄어들지 않는 선에서 최대한 강하게 다양한 transform을 적용시켜 주었다.
- Pseudo Labeling : 앙상블한 test dataset 결과를 이용해 pseudo label을 생성하여 사용했다.





## Contributor
- 강수빈([github](https://github.com/suuuuuuuubin)) : 
- 김인재([github](https://github.com/K-nowing)) : 
- 원상혁([github](https://github.com/wonsgong)) : 
- 이경민([github](https://github.com/lkm2835)) : 
- 최민서([github](https://github.com/minseo0214)) : 

