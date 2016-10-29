0925
from 9am -- 8:30 pm, run script_rpn_pedestrian_VGG16_caltech
the results are in "external/code3.2.1/results" and  "external/code3.2.1/data-USA/res"

0929-1001 RPN + BF experiment: all stages are completed, except the last plotting in dbEval_RPNBF
=====log==========
........ 
Test: 4022 / 4024 
miss rate:9.82
Loading detections: /usr/local/data/yuguang/git_all/RPN_BF_pedestrain/RPN_BF-RPN-pedestrian/external/code3.2.1/results/UsaTest
	Algorithm #1: RPN-ped
	Algorithm #2: RPN+BF
Loading ground truth: /usr/local/data/yuguang/git_all/RPN_BF_pedestrain/RPN_BF-RPN-pedestrian/external/code3.2.1/results/UsaTest
	Experiment #1: Reasonable
Evaluating: /usr/local/data/yuguang/git_all/RPN_BF_pedestrain/RPN_BF-RPN-pedestrian/external/code3.2.1/results/UsaTest
	Exp 1/1, Alg 1/2: Reasonable/RPN-ped
	Exp 1/1, Alg 2/2: Reasonable/RPN+BF
Error using figure
This functionality is no longer supported under the -nojvm startup option. For
more information, see "Changes to -nojvm Startup Option" in the MATLAB Release
Notes. To view the release note in your system browser, run
web('http://www.mathworks.com/help/matlab/release-notes.html#btsurqv-6',
'-browser').

Error in dbEval_RPNBF>plotExps (line 209)
  figure(1); clf; grid on; hold on; n=length(xs1); h=zeros(1,n);

Error in dbEval_RPNBF (line 125)
  plotExps( res, plotRoc, plotAlg, plotNum, plotName, ...

Error in script_rpn_bf_pedestrian_VGG16_caltech (line 193)
dbEval_RPNBF;

Error in run (line 96)
evalin('caller', [script ';']);


% ================ 1007: compare recall rate after NMS
gt recall rate = 0.6276
gt recall rate after nms-1 = 0.6168
gt recall rate after nms-2 = 0.5689
gt recall rate after nms-3 = 0.5547

% =================1012: anchor24 + ave-800 before nms + recall after nms
gt recall rate = 0.7613
gt recall rate after nms-1 = 0.7566
gt recall rate after nms-2 = 0.5851

% ================ 1010
training set (wider e1-e3) statistics before BF [before NMS: -1, nms thresh: 0.7, after nms: 1000]
score_threshold:0.006636
gt recall rate (ol >0.5) = 0.5928
gt recall rate (ol >0.7) = 0.2717
gt recall rate (ol >0.8) = 0.1001
gt recall rate (ol >0.9) = 0.0090
gt_num: 23055


% ================== 1015
two method to purify predicted bbox:
(1) select 13 out of 24 bboxes for each anchor position
anchor_num = size(conf.anchors, 1);
tmp = reshape(scores, anchor_num, []);
[~, sel_idx] = sort(tmp,1,'descend');
% select 1 out of 9 for size 10 x 10
[~, size10_sel1_idx] = max(tmp(1:9, :),[], 1);
% select 1 out of 4 for size 16 x 16
[~, size16_sel1_idx] = max(tmp(10:13, :),[], 1);
kept_score_idx = bsxfun(@plus, anchor_num * (0:length(size10_sel1_idx)-1), cat(1, size10_sel1_idx, size16_sel1_idx+9, repmat((14:24)',1, length(size10_sel1_idx))));

(2) select top-5 scoring bboxes for each anchor position
anchor_num = size(conf.anchors, 1);
tmp = reshape(scores, anchor_num, []);
[~, sel_idx] = sort(tmp,1,'descend');
%1013: only keep top-5 score anchors for each position
kept_score_idx = bsxfun(@plus, anchor_num * (0:size(sel_idx,2)-1), sel_idx(1:5,:));
%kept_score_idx = kept_score_idx';
kept_score_idx = kept_score_idx(:);
pred_boxes = pred_boxes(kept_score_idx, :);
scores = scores(kept_score_idx, :);

======1025==================
% plot boxes of each anchor position and find that hwRatio could be decimated to one half
inside proper position of 'im_detect':
figure, imshow(im/256);
bbs = [pred_boxes,scores]; 
bbs(:, 3) = bbs(:, 3) - bbs(:, 1) + 1;
bbs(:, 4) = bbs(:, 4) - bbs(:, 2) + 1;
bbApply('draw',bbs(11775:11781,:), 'g');

%plot position of anchors:
anchor_pos = [0.5*(anchors(:,1)+anchors(:,3)) 0.5*(anchors(:,2)+anchors(:,4))];
anchor_pos = anchor_pos(1:14:end,:);
re_im = imresize(im, im_scales);
figure, imshow(re_im/256);
hold on
plot(anchor_pos(:,1), anchor_pos(:,2),'g.','MarkerSize',5)
hold off

% decimate anchors by one half (only keep one boxes out of each anchor scale position)
anchor_num = size(conf.anchors, 1);  %14
half_anchor_num = size(conf.anchors, 1)/2; %7
tmp_scores = reshape(scores, anchor_num, []); 
hw1_score = tmp_scores(1:half_anchor_num, :);
hw2_score = tmp_scores(1+half_anchor_num:end, :);
hw1_greater_mask = (hw1_score >= hw2_score);
greater_mask = cat(1, hw1_greater_mask, ~hw1_greater_mask);
scores = scores(greater_mask(:),:);
pred_boxes = pred_boxes(greater_mask(:),:);

=====1026/1027===================
when using only halp of all anchors from rpn to do the bf training and set bf param the same as in conv4:
res1: output/VGG16_widerface_conv4_hwRatio_flip/bf_cachedir_1025_conv4_setting: 
      --recover to conv4 setting:
         opts.cascThr = -1;
	 opts.nPerNeg = 15;
 	 opts.max_rois_num_in_gpu = 3000;
	 opts.fg_thres_lo = 0.6;
	 opts.bg_thres_hi = 0.3;
	 atros features use top 0.2 and bottom 0.8
	 roi-pooling size 4x4
      => final 2048 tree training loss: 2.10e-06 (previous 1023 setting: 2.85e-06)
res2: output/VGG16_widerface_conv4_hwRatio_flip/bf_cachedir_fgthresh05
      -- opts.fg_thres_lo = 0.5;
         all other are set the same with res1 
      => final 2048 tree training loss: 2.06e-06
res3: output/VGG16_widerface_conv4_hwRatio_flip/bf_cachedir_casCal001
      -- opts.cascCal = .01; %1026 changed .005-->.01
         all other are set the same with res2 
      => final 2048 tree training loss: 8.31e-07
