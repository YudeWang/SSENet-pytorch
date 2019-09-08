# SSENet-pytorch

The pytorch implementation of [Self-supervised Scale Equivariant Network for Weakly Supervised Semantic Segmentaion](). 

## Introduction
As well-known, conventional CAM tends to be incomplete or over-activated due to weak supervision. Fortunately, we ﬁnd that semantic segmentation has a characteristic of spatial transformation equivariance, which can form a few self-supervisions to help weakly supervised learning. This work mainly explores the advantages of scale equivariant constrains for CAM generation, formulated as a self supervised scale equivariant network (SSENet). Extensive experiments on PASCAL VOC 2012 datasets demonstrate that our method achieves outstanding performance comparing with other state-of-the-arts.

Thanks to [jiwoon-ahn](https://github.com/jiwoon-ahn), the code of this repository borrow heavly from his [AffinityNet](https://github.com/jiwoon-ahn/psa) project, and we follw the same pipeline to verify the effectiveness of our SSENet.

## Dependency

- This repo is tested on Ubuntu 16.04, with python 3.6, pytorch 0.4, torchvision 0.2.1, CUDA 9.0, 4xGPUs (NVIDIA TITAN XP 12GB)
- Please install [tensorboardX](https://github.com/lanpa/tensorboardX) for training visualization.
- The dataset we used is PASCAL VOC 2012, please download the VOC [development kit](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/). It is suggested to make a soft link toward downloaded dataset.
```
ln -s $your_dataset_path/VOCdevkit/VOC2012 $your_voc12_root
``` 
- (Optional) The image-level labels have already been given in `voc12/cls_label.npy`. If you want to regenerate it (which is unnecessary), please download the annotation of VOC 2012 SegmentationClassAug training set (containing 10582 images), which can be download [here](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0) and place them all as `$your_voc12_root/SegmentationClassAug/xxxxxx.png`. Then run the code
```
cd voc12
python make_cls_labels.py --voc12_root $your_voc12_root
```
- (Optional) If you want to train the network by yourself, here is ImageNet pretrained model for VGG16 [vgg16_20M.caffemodel](http://liangchiehchen.com/projects/Init%20Models.html) and ResNet38 [ilsvrc-cls_rna-a1_cls1000_ep-0001.params](https://github.com/itijyou/ademxapp). Noting that our SSENet is **only tested on ResNet38**.


## Usage
### CAM generalization step

1. SSENet training 
```
python train_cls_ser.py --voc12_root $your_voc12_root --weights $your_weights_file --session_name $your_session_name
```

2. SSENet inference. Noting that the the crf results will be saved in `$your_crf_dir`+`_4.0` and `$your_crf_dir`+`_24.0`, where the parameters can be modified in `infer_cls_ser.py`. These two folders will be used further used in following AffinityNet training step.

```
python infer_cls_ser.py --weights $your_SSENet_checkpoint --infer_list [voc12/val.txt | voc12/train.txt | voc12/train_aug.txt] --out_cam $your_cam_dir --out_crf $your_crf_dir --out_cam_pred $your_pred_dir
```

3. CAM step evaluation. We provide python mIoU evaluation script `evaluation.py`, or you can use official development kit.
```
python evaluation.py --list $your_voc12_root/ImageSets/Segmentation/[val.txt | train.txt] --predict_dir $your_pred_dir --gt_dir $your_voc12_root/SegmentationClass
```
### Random walk step
The random walk step keep the same with AffinityNet project. 
1. Train AffinityNet.
```
python train_aff.py --weights $your_weights_file --voc12_root $your_voc12_root --la_crf_dir $your_crf_dir_4.0 --ha_crf_dir $your_crf_dir_24.0 --session_name $your_session_name
```
2. Random walk propagation
```
python infer_aff.py --weights $your_weights_file --infer_list [voc12/val.txt | voc12/train.txt] --cam_dir $your_cam_dir --voc12_root $your_voc12_root --out_rw $your_rw_dir
```
3. Random walk step evaluation
```
python evaluation.py --list $your_voc12_root/ImageSets/Segmentation/[val.txt | train.txt] --predict_dir $your_rw_dir --gt_dir $your_voc12_root/SegmentationClass
```

## Results

The generated pseudo labels are evaluated on PASCAL VOC 2012 train set.

Model | CAM step (mIoU) | CAM+rw step (mIoU) |           |
:----:|:---------------:|:------------------:|:----------:|
ResNet38 | 48.0 | 58.1 | AffinityNet cvpr submission|
ResNet38 | 47.3 | 58.8 | reimplemented baseline |
SSENet-ResNet38 | 49.8 | 62.1 | branch downsampling rate = 0.3  ([weights]())




