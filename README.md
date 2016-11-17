liu@0929
This repo is cloned from the following published research work for study and practice, all rights belong to the original authors.


==========================================
# Is Faster R-CNN Doing Well for Pedestrian Detection?

By Liliang Zhang, Liang Lin, Xiaodan Liang, Kaiming He

### Introduction

This code is relative to an [arXiv tech report](https://arxiv.org/abs/1607.07032), which is accepted on ECCV 2016.

The RPN code in this repo is written based on the MATLAB implementation of Faster R-CNN. Details about Faster R-CNN are in: [ShaoqingRen/faster_rcnn](https://github.com/ShaoqingRen/faster_rcnn).

This BF code in this repo is written based on Piotr's Image & Video Matlab Toolbox. Details about Piotr's Toolbox are in: [pdollar/toolbox](https://github.com/pdollar/toolbox).

This code has been tested on Ubuntu 14.04 with MATLAB 2014b and CUDA 7.5.

### Citing RPN+BF

If you find this repo useful in your research, please consider citing:

    @article{zhang2016faster,
      title={Is Faster R-CNN Doing Well for Pedestrian Detection?},
      author={Zhang, Liliang and Lin, Liang and Liang, Xiaodan and He, Kaiming},
      journal={arXiv preprint arXiv:1607.07032},
      year={2016}
    }

### Requirements

0. `Caffe` build for RPN+BF (see [here](https://github.com/zhangliliang/caffe/tree/RPN_BF))
    - If the mex in 'external/caffe/matlab/caffe_faster_rcnn' could not run under your system, please follow the [instructions](https://github.com/zhangliliang/caffe/tree/RPN_BF) on our Caffe branch to compile and replace the mex.

0. MATLAB

0. GPU: Titan X, K40c, etc.


**WARNING**: The `caffe_.mexa64` in `external/caffe/matlab/caffe_faster_rcnn` might be not compatible with your computer. If so, please try to compile [this Caffe version](https://github.com/zhangliliang/caffe/tree/RPN_BF) and replace it. 

### Testing Demo

0. Download `VGG16_caltech_final.zip` from [BaiduYun](https://pan.baidu.com/s/1miNdKZe),or [Onedrive](https://1drv.ms/u/s!AgVYvWT--3HKhBgkVQkeMkLU_A5s) and unzip it in the repo folder.

0. Start MATLAB from the repo folder.

0. Run `faster_rcnn_build`

0. Run `script_rpn_bf_pedestrian_VGG16_caltech_demo` to see the detection results on some images collected in Internet.

### Training on Caltech (RPN)

0. Download "Matlab evaluation/labeling code (3.2.1)" as `external/code3.2.1` by run `fetch_data/fetch_caltech_toolbox.m`

0. Download the annotations and videos in [Caltech Pedestrian Dataset](http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/datasets/USA/) and put them in the proper folder follow the instruction in the [website](http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/).

0. Download the VGG-16 pretrain model and the relative prototxt in `VGG16_caltech_pretrain.zip` from [BaiduYun](http://pan.baidu.com/s/1hrALBus) or [OneDrive](https://1drv.ms/u/s!AAVYvWT--3HKhCY), and unzip it in the repo folder. The md5sum for `vgg16.caffemodel` should be `e54292186923567dc14f21dee292ae36`.

0. Start MATLAB from the repo folder, and run `extract_img_anno` for extracting images in JPEG format and annotations in TEXT format from the Caltech dataset.

0. Run `script_rpn_pedestrian_VGG16_caltech` to train and test the RPN model on Caltech. Wait about half day for training and testing.

0. Hopefully it would give the evaluation results around ~14% MR after running.   

### Training on Caltech (RPN+BF)

0. Follow the instruction in "Training on Caltech (RPN)" for obtaining the RPN model.

0. Run `script_rpn_bf_pedestrian_VGG16_caltech` to train and test the BF model on Caltech. Wait about two or three days for training and testing.

0. Hopefully it would give the evaluation results around ~10% MR after running.  


% liu@0922: should run demo like this
export LD_LIBRARY_PATH=/usr/local/data/yuguang/software/cudnn-6.5-linux-x64-v2:$LD_LIBRARY_PATH
matlab  (start from cmd)
in root_dir,  run "./startup", then "script_rpn_bf_pedestrian_VGG16_caltech_demo"

# liu@0925
only require 1.5G gpu memory

#liu#1019
run(fullfile('experiments','script_rpn_bf_face_VGG16_widerface_conv4_submit'))

#1103 res50 conv4==================
gt recall rate = 0.7381
gt recall rate after nms-3 = 0.5875

#1108
1. conv4_atros_reduce three atros:
1) normal 3x3; 2)3x3 with hole 2; 3) 3x3 with hole 4
and elemltwise SUM them, but the result diverge

2. conv4_atros_reduce2
same three atros as 1., but concatenate them and them use a 1x1 conv to reduce dimension
test res:
gt recall rate = 0.7063
gt recall rate after nms-3 = 0.6029

3. conv4_atros_reduce3 ===> add a max_pooling before the atros conv layer
gt recall rate = 0.6366
gt recall rate after nms-3 = 0.5407
But it seems from the train net plot that the val loss is even smaller than reduce2, why recall rate is so low? check it out.

#1116
%=========show ohem selected labels
aa = labels_weights_ohem;
aa = aa';
aa = reshape(aa, 7, [], 150);
aa(aa == 0) = 255;
aa(aa~=0) == 0;
aa = uint8(aa);
figure, imshow(squeeze(aa(1,:,:)));
