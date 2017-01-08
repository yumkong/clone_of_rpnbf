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

#1119
ohem (see results at VGG16_widerface_conv4_ohem_final)
gt recall rate = 0.5409
gt recall rate after nms-3 = 0.5000

#1120
(1)analyze conv4_multibox result (obtained on 1113)
check the value range of the conv layer right before conv4_clf(/reg) and conv5_clf(/reg)
conv4: caffe_net.blobs('conv_proposal_conv4').get_data();
conv5: caffe_net.blobs('reduce_proposal').get_data();
               conv4                conv5
       median   max     min | median    max    min   
img1   0.2453  7.6251    0  | 0.5161  17.5855   0
img2   0.2612  6.7812    0  | 0.4494  14.5742   0
img3   0.2327  8.2849    0  | 0.3966  21.6427   0
img4   0.2767  5.0860    0  | 0.6816  18.1023   0
img5   0.2658  7.3290    0  | 0.4877  14.6057   0
img6   0.2431  8.0495    0  | 0.4829  15.0042   0

(2)analyze conv4_ohem2 result (obtained on 1119)
check how the training loss changes as iteration goes:
*** img1409
Iter 1,     Image 1409: 2.2 Hz, accuarcy = 0.2865, loss_bbox = 0.0508, loss_cls = 12.0586, accuracy_fg = 0.5217, accuracy_bg = 0.0000, 
Iter 4700,  Image 1409: 4.3 Hz, accuarcy = 0.7572, loss_bbox = 0.0019, loss_cls = 0.1269, accuracy_fg = 0.6522, accuracy_bg = 0.9844, 
Iter 8021,  Image 1409: 4.3 Hz, accuarcy = 0.7600, loss_bbox = 0.0015, loss_cls = 0.1477, accuracy_fg = 0.6957, accuracy_bg = 0.9531, 
Iter 8963,  Image 1409: 4.4 Hz, accuarcy = 0.7434, loss_bbox = 0.0015, loss_cls = 0.1572, accuracy_fg = 0.6522, accuracy_bg = 0.9531, 
Iter 11515, Image 1409: 4.2 Hz, accuarcy = 0.7632, loss_bbox = 0.0014, loss_cls = 0.1435, accuracy_fg = 0.7826, accuracy_bg = 0.9583, 
Iter 16007, Image 1409: 4.5 Hz, accuarcy = 0.7646, loss_bbox = 0.0014, loss_cls = 0.1047, accuracy_fg = 0.7391, accuracy_bg = 0.9740, 
Iter 17782, Image 1409: 4.4 Hz, accuarcy = 0.7573, loss_bbox = 0.0014, loss_cls = 0.1488, accuracy_fg = 0.9130, accuracy_bg = 0.9583,
Iter 20694, Image 1409: 4.2 Hz, accuarcy = 0.7580, loss_bbox = 0.0012, loss_cls = 0.1151, accuracy_fg = 0.8696, accuracy_bg = 0.9583,
Iter 23993, Image 1409: 4.2 Hz, accuarcy = 0.7609, loss_bbox = 0.0011, loss_cls = 0.1106, accuracy_fg = 0.8696, accuracy_bg = 0.9583, 
Iter 27648, Image 1409: 4.4 Hz, accuarcy = 0.7604, loss_bbox = 0.0012, loss_cls = 0.1299, accuracy_fg = 0.8696, accuracy_bg = 0.9427,
Iter 28560, Image 1409: 4.4 Hz, accuarcy = 0.7593, loss_bbox = 0.0011, loss_cls = 0.1097, accuracy_fg = 0.8696, accuracy_bg = 0.9635, 

*** img1695
Iter 100, Image 1695: 4.1 Hz, accuarcy = 0.7400, loss_bbox = 0.0066, loss_cls = 0.5579, accuracy_fg = 0.0000, accuracy_bg = 0.9792,
Iter 4466, Image 1695: 3.7 Hz, accuarcy = 0.7348, loss_bbox = 0.0060, loss_cls = 0.5196, accuracy_fg = 0.2969, accuracy_bg = 0.9792, 
Iter 8311, Image 1695: 4.0 Hz, accuarcy = 0.7328, loss_bbox = 0.0058, loss_cls = 0.6274, accuracy_fg = 0.2969, accuracy_bg = 0.9531, 
Iter 10529, Image 1695: 3.8 Hz, accuarcy = 0.7416, loss_bbox = 0.0061, loss_cls = 0.5737, accuracy_fg = 0.3125, accuracy_bg = 0.9635, 
Iter 12262, Image 1695: 3.9 Hz, accuarcy = 0.7392, loss_bbox = 0.0059, loss_cls = 0.4063, accuracy_fg = 0.3750, accuracy_bg = 0.9844, 
Iter 15879, Image 1695: 4.0 Hz, accuarcy = 0.7563, loss_bbox = 0.0059, loss_cls = 0.5814, accuracy_fg = 0.2188, accuracy_bg = 0.9948,
Iter 18156, Image 1695: 3.7 Hz, accuarcy = 0.7346, loss_bbox = 0.0050, loss_cls = 0.4632, accuracy_fg = 0.3125, accuracy_bg = 0.9896,
Iter 21444, Image 1695: 3.8 Hz, accuarcy = 0.7364, loss_bbox = 0.0051, loss_cls = 0.3209, accuracy_fg = 0.4375, accuracy_bg = 0.9844, 
Iter 22244, Image 1695: 3.9 Hz, accuarcy = 0.7407, loss_bbox = 0.0051, loss_cls = 0.3018, accuracy_fg = 0.4844, accuracy_bg = 0.9896, 
Iter 25377, Image 1695: 4.0 Hz, accuarcy = 0.7329, loss_bbox = 0.0049, loss_cls = 0.3008, accuracy_fg = 0.4531, accuracy_bg = 0.9844, 
Iter 28352, Image 1695: 3.9 Hz, accuarcy = 0.7368, loss_bbox = 0.0050, loss_cls = 0.2924, accuracy_fg = 0.4688, accuracy_bg = 0.9948, 

(3) check the scale param of normalize_layer
caffe_net.params('conv4_3_norm',1).get_data();

#1121
*** these are done by old code ()
(1) when add 32 to conv4 (now it is in charge of [8 16 32]), while conv5 still [32 64 128 256 512 900], normalize_layer scale is 40
gt recall rate = 0.7958
gt recall rate after nms-3 = 0.6050
(2) previously, normalize_layer scale is 20, conv4 [8 16], conv5 [32 64 128 256 512 900]
gt recall rate = 0.7723
gt recall rate after nms-3 = 0.6099
(3) original conv4
gt recall rate = 0.7728
gt recall rate after nms-3 = 0.5450
(4) conv4_ohem
gt recall rate = 0.4997
gt recall rate after nms-3 = 0.4525


#1121 added multibox_final2: add a conv4_atros (or conv3_atros? or both?) in parallel with conv4
==> conv3_atros cannot be used because of different size from conv4
(5)(1)+conv4_atros this is done by new code ()
gt recall rate = 0.6732
gt recall rate after nms-3 = 0.6111
#1123
(6) multibox final3 (conv4 + conv5 + conv6)
gt recall rate = 0.6649
gt recall rate after nms-3 = 0.6106
(7) multibox_ohem (conv4 + conv5 + conv6)
gt recall rate = 0.6604
gt recall rate after nms-3 = 0.6278
#1124
(8) multibox final3_flip
gt recall rate = 0.6309
gt recall rate after nms-3 = 0.5769
(9) since conv4 val err diverges as iteration goes, reduce conv4 clf loss weight from 4 to 2, call it  multibox_ohem_2
gt recall rate = 0.6400
gt recall rate after nms-3 = 0.5982
(10) althought in multibox_ohem_2, the conv4 val err seems to be convergent, there is still space to improve it.
now reduce conv4 clf loss weight from 2 to 1, conv4 bbox loss weight from 20 to 10, also add flipped training data, double training iterations and stepsize ==> name it multibox_ohem_3
gt recall rate = 0.5701
gt recall rate after nms-3 = 0.5194
(score_threshold conv4 = 0.851287, conv5 = 0.803228, conv6 = 0.902687)
#1125
(11)
Since (10) is so bad, recover to multibox ohem's setting:
conv4: cls 4, reg 20; conv5: cls 1-->2, 5--> 10; conv6: cls 1, reg 5.
base_lr also return to previous 0.0005 (in multibox_ohem3 set to 0.001). 
multibox_ohem_4
gt recall rate = 0.6139
gt recall rate after nms-3 = 0.5702
score_threshold conv4 = 0.783169, conv5 = 0.776060, conv6 = 0.770421
(12) 
since (11) is still not so good as (7), is it because of the flipped image? remove them,just use the normal image
gt recall rate = 0.6564
gt recall rate after nms-3 = 0.6251
(13)
train with all train images (with multi-box ohem model)
gt recall rate = 0.6802
gt recall rate after nms-3 = 0.6441

#1201
(14)
multibox ohem with flip (randomly ud, lr, rot90, rot90+lr)
gt recall rate = 0.6438
gt recall rate after nms-3 = 0.6138
(15) [The current best result!] 
remove the conv4 atros from [multibox ohem] to see if it can improve small faces detection rate, see  script_rpn_face_VGG16_widerface_multibox_ohem_singleconv4.m
the training gpu memory usage reduces to [3862M]!
gt recall rate = 0.6850
gt recall rate after nms-3 = 0.6530

(16) Res50 multibox_ohem [gpu memory usage 6689M]
======================
how to set normlize layer scale
conv4: caffe_solver.net.blobs('conv_proposal_conv4').get_data();
conv5: caffe_solver.net.blobs('reduce_proposal').get_data();
conv6: caffe_solver.net.blobs('reduce_proposal_conv6').get_data();

=== code =========
aa = caffe_solver.net.blobs('conv_proposal_conv4').get_data();
bb = caffe_solver.net.blobs('reduce_proposal').get_data();
cc = caffe_solver.net.blobs('reduce_proposal_conv6').get_data();
fprintf('conv4: %.2f, %.2f\n', median(aa(:)), max(aa(:)));
fprintf('conv5: %.2f, %.2f\n', median(bb(:)), max(bb(:)));
fprintf('conv6: %.2f, %.2f\n', median(cc(:)), max(cc(:)));

==> found that they are of the same magnitude level, so temperarily not use normlize layer
=================
result: slightly better than vgg16 multibox_ohem, but worse than 15
gt recall rate = 0.6640
gt recall rate after nms-3 = 0.6299

================== 
1205
on puck, "script_rpn_bf_face_VGG16_widerface_multibox_ohem_submit" is so slow while doing the following:
In caffe.Net/forward (line 142)
      self.forward_prefilled();

In rois_get_features_ratio (line 75)
        output_blobs = caffe_net.forward(net_inputs);

In DeepTrain_otf_trans_ratio>sampleWins (line 486)
           sel_feat = rois_get_features_ratio(opts.conf, opts.caffe_net, im, sel_box, opts.max_rois_num_in_gpu, opts.ratio);

In DeepTrain_otf_trans_ratio (line 165)
    [X0, X0_score, sel_idxes] = sampleWins( detector, stage, 0 );

In script_rpn_bf_face_VGG16_widerface_multibox_ohem_submit (line 249)
detector = DeepTrain_otf_trans_ratio( opts );

In run (line 96)
evalin('caller', [script ';']);
======================


1205
conv3 series:
(1) conv3  afternms-100
gt recall rate = 0.5317
gt recall rate after nms-3 = 0.3967
(2) conv3_4 (aka conv4) afternms-100
gt recall rate = 0.5950
gt recall rate after nms-3 = 0.4695
(3) conv3plus4 afternms-100 (thresh: 0.943959)
gt recall rate = 0.6122
gt recall rate after nms-3 = 0.5055
https://groups.google.com/forum/#!topic/caffe-users/owQEvAhP7gM


------------------------- Iteration 1000 -------------------------
Training : error 0.449, loss (cls 0.698, reg 0.0942)
Testing  : error 0.604, loss (cls 0.698, reg 0.0486)
------------------------- Iteration 2000 -------------------------
Training : error 0.461, loss (cls 0.698, reg 0.0936)
Testing  : error 0.396, loss (cls 0.681, reg 0.049)

1210:
==============================
divergence problem occurs when training ohem happy
Iter 12647, Image 2677: 0.6 Hz, accuarcy_conv34 = 0.9937, loss_bbox_conv34 = 0.0004, loss_cls_conv34 = 0.0311, accuracy_fg_conv4 = 1.0000, accuracy_bg_conv4 = 0.9740, 
			    accuarcy_conv5 = 0.9947, loss_bbox_conv5 = 0.0041, loss_cls_conv5 = 0.1472, accuracy_fg_conv5 = 0.9444, accuracy_bg_conv5 = 0.9479, 
			    accuarcy_conv6 = 1.0000, loss_bbox_conv6 = 0.0000, loss_cls_conv6 = 0.0000, accuracy_fg_conv6 = 0.0000, accuracy_bg_conv6 = 1.0000, 
Iter 12648, Image 1012: 0.9 Hz, accuarcy_conv34 = 0.9976, loss_bbox_conv34 = 0.0000, loss_cls_conv34 = 0.0103, accuracy_fg_conv4 = 0.0000, accuracy_bg_conv4 = 0.9948, 
			    accuarcy_conv5 = 0.9937, loss_bbox_conv5 = 0.0030, loss_cls_conv5 = 0.1339, accuracy_fg_conv5 = 0.5000, accuracy_bg_conv5 = 0.9688, 
			    accuarcy_conv6 = 1.0000, loss_bbox_conv6 = 0.0000, loss_cls_conv6 = 0.0000, accuracy_fg_conv6 = 0.0000, accuracy_bg_conv6 = 1.0000, 
Iter 12649, Image 2163: 0.7 Hz, accuarcy_conv34 = 0.9997, loss_bbox_conv34 = 0.0000, loss_cls_conv34 = 0.0061, accuracy_fg_conv4 = 0.0000, accuracy_bg_conv4 = 0.9948, 
			    accuarcy_conv5 = 0.9888, loss_bbox_conv5 = 7.5825, loss_cls_conv5 = 16.9762, accuracy_fg_conv5 = 0.3125, accuracy_bg_conv5 = 0.9375, 
			    accuarcy_conv6 = 0.9644, loss_bbox_conv6 = 43.7842, loss_cls_conv6 = 17.4673, accuracy_fg_conv6 = 0.0000, accuracy_bg_conv6 = 1.0000, 
Iter 12650, Image 1499: 0.7 Hz, accuarcy_conv34 = 1.0000, loss_bbox_conv34 = 0.0000, loss_cls_conv34 = 0.0009, accuracy_fg_conv4 = 0.0000, accuracy_bg_conv4 = 1.0000, 
			    accuarcy_conv5 = 0.7361, loss_bbox_conv5 = 0.0838, loss_cls_conv5 = 35.9632, accuracy_fg_conv5 = 0.0606, accuracy_bg_conv5 = 0.4010, 
			    accuarcy_conv6 = 0.1197, loss_bbox_conv6 = 0.0000, loss_cls_conv6 = 43.6683, accuracy_fg_conv6 = 0.0000, accuracy_bg_conv6 = 0.5000, 
Iter 12651, Image 2373: 0.6 Hz, accuarcy_conv34 = 1.0000, loss_bbox_conv34 = 0.0000, loss_cls_conv34 = 0.0002, accuracy_fg_conv4 = 0.0000, accuracy_bg_conv4 = 1.0000, 
			    accuarcy_conv5 = 0.7346, loss_bbox_conv5 = 7910.9019, loss_cls_conv5 = 87.3365, accuracy_fg_conv5 = 0.0000, accuracy_bg_conv5 = 0.0000, 
			    accuarcy_conv6 = 0.1236, loss_bbox_conv6 = 0.0000, loss_cls_conv6 = 32.7512, accuracy_fg_conv6 = 0.0000, accuracy_bg_conv6 = 0.6250, 
Iter 12652, Image 373: 0.8 Hz, accuarcy_conv34 = 0.9975, loss_bbox_conv34 = 0.0014, loss_cls_conv34 = 1.1023, accuracy_fg_conv4 = 0.0000, accuracy_bg_conv4 = 1.0000, 
			    accuarcy_conv5 = 0.5796, loss_bbox_conv5 = 906044189293610388276610334720.0000, loss_cls_conv5 = 17.4673, accuracy_fg_conv5 = 0.5938, accuracy_bg_conv5 = 0.9375, 

=== debug code =========
aa = caffe_solver.net.blobs('conv3_3').get_data();
bb = caffe_solver.net.blobs('conv3_3_norm').get_data();
cc = caffe_solver.net.blobs('conv4_3_2x').get_data();
dd = caffe_solver.net.blobs('conv4_3_norm').get_data();
ee = caffe_solver.net.blobs('conv5_3').get_data();
ff = caffe_solver.net.blobs('conv6_2').get_data();
fprintf('conv3_3: %.2f, %.2f\n', median(aa(:)), max(aa(:)));
fprintf('conv3_3_norm: %.2f, %.2f\n', median(bb(:)), max(bb(:)));
fprintf('conv4_3_deconv: %.2f, %.2f\n', median(cc(:)), max(cc(:)));
fprintf('conv4_3_deconv_norm: %.2f, %.2f\n', median(dd(:)), max(dd(:)));
fprintf('conv5_3: %.2f, %.2f\n', median(ee(:)), max(ee(:)));
fprintf('conv6_2: %.2f, %.2f\n', median(ff(:)), max(ff(:)));

img1:
conv3_3: 0.00, 9137.48
conv3_3_norm: 0.00, 32.59
conv4_3_deconv: 0.00, 2093.30
conv4_3_deconv_norm: 0.00, 34.51
conv5_3: 0.00, 176.00
conv6_2: 0.05, 24.14

img2:
conv3_3: 0.00, 10316.69
conv3_3_norm: 0.00, 33.82
conv4_3_deconv: 0.00, 1909.56
conv4_3_deconv_norm: 0.00, 32.58
conv5_3: 0.00, 193.89
conv6_2: 0.08, 34.67

img3:
conv3_3: 0.00, 12206.08
conv3_3_norm: 0.00, 31.41
conv4_3_deconv: 0.00, 2408.85
conv4_3_deconv_norm: 0.00, 34.88
conv5_3: 0.00, 254.27
conv6_2: 0.07, 36.82

img10:
conv3_3: 0.00, 9741.09
conv3_3_norm: 0.00, 35.13
conv4_3_deconv: 0.00, 2719.35
conv4_3_deconv_norm: 0.00, 33.47
conv5_3: 0.00, 179.52
conv6_2: 0.06, 32.35
==> from the above, the problem seems to derive from large scale of conv5_3 (around 200), compared to other conv layers(conv3_3_norm, conv4_3_deconv_norm and conv6_2
all of which are around 35)

1212
ohem_happy (e1_e11) is done!
score_threshold conv4 = 0.862857, conv5 = 0.828735, conv6 = 0.924826
gt recall rate = 0.6291
gt recall rate after nms-3 = 0.6018

1215
ohem_happy_flip (e1_e11) is done!
score_threshold conv4 = 0.891689, conv5 = 0.828444, conv6 = 0.910268
gt recall rate = 0.6289
gt recall rate after nms-3 = 0.6002  (similar to ohem_happy)

1216
CMS (keep average 200 proposal per image)
gt recall rate = 0.5485
gt recall rate after nms-3 = 0.5015


===========
1217
training BF error:
Training AdaBoost: nWeak=1536 nFtrs=38400 pos=159424 neg=200000
Out of memory. Type HELP MEMORY for your options.
==> so set the upper limit to pos=160000 neg=160000

multibox_CRV
gt recall rate = 0.6186
gt recall rate after nms-3 = 0.5820
==> not so good as conv4single, why?

===========
1218
finally choose ohem_happy_flip as CRV submission version

for i = 1:length(image_roidb_train)
  aa = strfind(image_roidb_train(i).image_path, '/');
  image_roidb_train(i).image_path = [image_roidb_train(i).image_path(1:aa(end-1)) image_roidb_train(i).image_id];
end

for i = 1:length(image_roidb_val)
  aa = strfind(image_roidb_val(i).image_path, '/');
  image_roidb_val(i).image_path = [image_roidb_val(i).image_path(1:aa(end-1)) image_roidb_val(i).image_id];
end

=============
1227
(1)
ohem_happy_flip training done, result is very good!
after nms: conv34:50, conv5:100, conv6=10
score_threshold conv4 = 0.936758, conv5 = 0.870751, conv6 = 0.924484
(2)
ohem_happy_flip bf, data preparation has some problem, debug:
===
i = 1;
img = imread(dataset.imdb_train{1}.image_at(i));
im(img);
bbs = [roidb_train_BF.rois(i).boxes roidb_train_BF.rois(i).scores];
bbs(:, 3) = bbs(:, 3) - bbs(:, 1) + 1;
bbs(:, 4) = bbs(:, 4) - bbs(:, 2) + 1;
bbApply('draw',bbs, 'g');
===
==> fix the bug, so 
score_threshold conv4 = 0.630544, conv5 = 0.583918, conv6 = 0.386236
gt recall rate (ol >0.5) = 0.7473
gt recall rate (ol >0.7) = 0.5190
gt recall rate (ol >0.8) = 0.2950
gt recall rate (ol >0.9) = 0.0502
gt_num: 188676

=============
0103
Ablation experiment finally use this
nms = 0.8
model.stage1_rpn.nms.after_nms_topN_conv34      = 150;  %50
model.stage1_rpn.nms.after_nms_topN_conv5      	= 40;  %100
model.stage1_rpn.nms.after_nms_topN_conv6      	= 10;  %1000
32-500
For det-4:
All scales: gt recall rate = 0.6067
8-32: gt recall rate = 0.7471
33-360: gt recall rate = 0.3369
361-900: gt recall rate = 0.0000
For det-16:
All scales: gt recall rate = 0.4747
8-32: gt recall rate = 0.2579
33-360: gt recall rate = 0.8957
361-900: gt recall rate = 0.9839
For det-32:
All scales: gt recall rate = 0.0200
8-32: gt recall rate = 0.0000
33-360: gt recall rate = 0.0495
361-900: gt recall rate = 0.9919
For det-all:
All scales: gt recall rate = 0.8083
8-32: gt recall rate = 0.7524
33-360: gt recall rate = 0.9164
361-900: gt recall rate = 0.9919


==============
0105 happy [noflip] [no atrous] [ohem] train for 20k30k
nms = 0.8
model.stage1_rpn.nms.after_nms_topN_conv34      = 150;  %50
model.stage1_rpn.nms.after_nms_topN_conv5      	= 40;  %100
model.stage1_rpn.nms.after_nms_topN_conv6      	= 10;  %1000
score_threshold conv4 = 0.551139, conv5 = 0.479938, conv6 = 0.000000
For det-4:
All scales: gt recall rate = 0.6285
8-32: gt recall rate = 0.7697
33-360: gt recall rate = 0.3572
361-900: gt recall rate = 0.0000
For det-16:
All scales: gt recall rate = 0.4953
8-32: gt recall rate = 0.2915
33-360: gt recall rate = 0.8911
361-900: gt recall rate = 0.9919
For det-32:
All scales: gt recall rate = 0.0210
8-32: gt recall rate = 0.0000
33-360: gt recall rate = 0.0524
361-900: gt recall rate = 0.9919
For det-all:
All scales: gt recall rate = 0.8239
8-32: gt recall rate = 0.7813
33-360: gt recall rate = 0.9059
361-900: gt recall rate = 0.9919

===========
0106 happy [noflip] [atrous] [no ohem] train for 20k30k
score_threshold conv4 = 0.728043, conv5 = 0.795226, conv6 = 0.268397
For det-4:
All scales: gt recall rate = 0.6686
8-32: gt recall rate = 0.7990
33-360: gt recall rate = 0.4189
361-900: gt recall rate = 0.0000
For det-16:
All scales: gt recall rate = 0.4851
8-32: gt recall rate = 0.2723
33-360: gt recall rate = 0.8987
361-900: gt recall rate = 0.9839
For det-32:
All scales: gt recall rate = 0.0199
8-32: gt recall rate = 0.0000
33-360: gt recall rate = 0.0493
361-900: gt recall rate = 0.9758
For det-all:
All scales: gt recall rate = 0.8435
8-32: gt recall rate = 0.8031
33-360: gt recall rate = 0.9214
361-900: gt recall rate = 0.9919

===================
0106 happy [noflip] [noatrous] [no ohem] train for 20k30k
score_threshold conv4 = 0.708914, conv5 = 0.730325, conv6 = 0.248624
For det-4:
All scales: gt recall rate = 0.6683
8-32: gt recall rate = 0.7994
33-360: gt recall rate = 0.4173
361-900: gt recall rate = 0.0000
For det-16:
All scales: gt recall rate = 0.4973
8-32: gt recall rate = 0.2862
33-360: gt recall rate = 0.9076
361-900: gt recall rate = 0.9919
For det-32:
All scales: gt recall rate = 0.0198
8-32: gt recall rate = 0.0000
33-360: gt recall rate = 0.0488
361-900: gt recall rate = 0.9839
For det-all:
All scales: gt recall rate = 0.8454
8-32: gt recall rate = 0.8035
33-360: gt recall rate = 0.9263
361-900: gt recall rate = 0.9919

