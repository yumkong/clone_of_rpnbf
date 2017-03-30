function scores = fast_rcnn_im_detect_widerface_mpfvn_realmp(conf, caffe_net, im, boxes)
% [pred_boxes, scores] = fast_rcnn_im_detect(conf, caffe_net, im, boxes, max_rois_num_in_gpu)
% --------------------------------------------------------
% Fast R-CNN
% Reimplementation based on Python Fast R-CNN (https://github.com/rbgirshick/fast-rcnn)
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------

    [im_blob, rois_s4, rois_cxt_s4, rois_s8, rois_cxt_s8, rois_s16, rois_cxt_s16, ~, score_mask, ind_inv] = get_blobs(conf, im, boxes);
    
    % permute data into caffe c++ memory, thus [num, channels, height, width]
    im_blob = im_blob(:, :, [3, 2, 1], :); % from rgb to brg
    im_blob = permute(im_blob, [2, 1, 3, 4]);
    im_blob = single(im_blob);
    
    rois_blob_s4 = rois_s4 - 1; % to c's index (start from 0)
    rois_blob_s4 = permute(rois_blob_s4, [3, 4, 2, 1]);
    rois_blob_s4 = single(rois_blob_s4);
    %0322 added
    rois_cxt_blob_s4 = rois_cxt_s4 - 1; % to c's index (start from 0)
    rois_cxt_blob_s4 = permute(rois_cxt_blob_s4, [3, 4, 2, 1]);
    rois_cxt_blob_s4 = single(rois_cxt_blob_s4);
    
    rois_blob_s8 = rois_s8 - 1; % to c's index (start from 0)
    rois_blob_s8 = permute(rois_blob_s8, [3, 4, 2, 1]);
    rois_blob_s8 = single(rois_blob_s8);
    %0322 added
    rois_cxt_blob_s8 = rois_cxt_s8 - 1; % to c's index (start from 0)
    rois_cxt_blob_s8 = permute(rois_cxt_blob_s8, [3, 4, 2, 1]);
    rois_cxt_blob_s8 = single(rois_cxt_blob_s8);
    
    rois_blob_s16 = rois_s16 - 1; % to c's index (start from 0)
    rois_blob_s16 = permute(rois_blob_s16, [3, 4, 2, 1]);
    rois_blob_s16 = single(rois_blob_s16);
    %0322 added
    rois_cxt_blob_s16 = rois_cxt_s16 - 1; % to c's index (start from 0)
    rois_cxt_blob_s16 = permute(rois_cxt_blob_s16, [3, 4, 2, 1]);
    rois_cxt_blob_s16 = single(rois_cxt_blob_s16);
 
    net_inputs = {im_blob, rois_blob_s4, rois_cxt_blob_s4, rois_blob_s8, rois_cxt_blob_s8, rois_blob_s16, rois_cxt_blob_s16};

    % Reshape net's input blobs
    caffe_net.reshape_as_input(net_inputs);
    output_blobs = caffe_net.forward(net_inputs);

    scores_s16 = squeeze(output_blobs{1})';
    scores_s4 = squeeze(output_blobs{2})';
    scores_s8 = squeeze(output_blobs{3})';
    scores = cat(1, scores_s4, scores_s8, scores_s16);
    if ~isempty(scores)
        scores = scores(:, 2);
        %0326 added
        scores = scores(score_mask>0, :);
        % reverse the original order
        scores = scores(ind_inv, :);
    else
        scores = []; 
    end
end

function [data_blob, rois_s4, rois_cxt_s4, rois_s8, rois_cxt_s8, rois_s16, rois_cxt_s16, im_scale_factors, score_mask, ind_inv] = get_blobs(conf, im, rois)
    [data_blob, im_scale_factors] = get_image_blob(conf, im);
    %rois_blob = get_rois_blob(conf, rois, im_scale_factors, size(im));
    %0326 added
    %marking newly added 'fake' rois
    score_mask = [];
    % 0326 changed: here it is not training, so should partition with no
    % overlap
    if ~isempty(rois)
        face_hei = rois(:,4) - rois(:,2) + 1;
        ind_s4 = find(face_hei < 12);
        ind_s8 = find(face_hei >= 12 & face_hei < 120);
        ind_s16 = find(face_hei >= 120);
        rois_s4 = rois(ind_s4, :);
        rois_s8 = rois(ind_s8, :);
        rois_s16 = rois(ind_s16, :);
        ind_all = cat(1, ind_s4, ind_s8, ind_s16);
        [~, ind_inv] = sort(ind_all, 'ascend');
    else
        rois_s4 = [];
        rois_s8 = [];
        rois_s16 = [];
        ind_inv = [];
    end
    if ~isempty(rois_s4)
        score_mask = cat(1, score_mask, ones(size(rois_s4, 1), 1));
    else
        score_mask = cat(1, score_mask, 0);
        rois_s4 = single([66 66 76 76]);
    end
    
    if ~isempty(rois_s8)
        score_mask = cat(1, score_mask, ones(size(rois_s8, 1), 1));
    else
        score_mask = cat(1, score_mask, 0);
        rois_s8 = single([66 66 126 126]);
    end
    
    if ~isempty(rois_s16)
        score_mask = cat(1, score_mask, ones(size(rois_s16, 1), 1));
    else
        score_mask = cat(1, score_mask, 0);
        rois_s16 = single([66 66 226 226]);
    end
    [rois_s4, rois_cxt_s4, rois_s8, rois_cxt_s8, rois_s16, rois_cxt_s16] = get_rois_blob(conf, rois_s4, rois_s8, rois_s16, im_scale_factors, size(im));
end

function [blob, im_scales] = get_image_blob(conf, im)
    [ims, im_scales] = arrayfun(@(x) prep_im_for_blob(im, conf.image_means, x, conf.test_max_size), conf.test_scales, 'UniformOutput', false);
    im_scales = cell2mat(im_scales);
    blob = im_list_to_blob(ims);    
end

function [rois_blob_s4, rois_cxt_blob_s4, rois_blob_s8, rois_cxt_blob_s8, rois_blob_s16, rois_cxt_blob_s16] = get_rois_blob(conf, ...
                                            rois_s4, rois_s8, rois_s16, im_scale_factors, im_siz)
    [feat_rois_s4, feat_rois_cxt_s4,feat_rois_s8, feat_rois_cxt_s8,feat_rois_s16, feat_rois_cxt_s16] = map_im_rois_to_feat_rois(conf, rois_s4, rois_s8, rois_s16, im_scale_factors, im_siz);
    % stride 4
    level_s4 = ones(size(feat_rois_s4, 1),1);
    rois_blob_s4 = single([level_s4, feat_rois_s4]);
    rois_cxt_blob_s4 = single([level_s4, feat_rois_cxt_s4]);
    % stride 8
    level_s8 = ones(size(feat_rois_s8, 1),1);
    rois_blob_s8 = single([level_s8, feat_rois_s8]);
    rois_cxt_blob_s8 = single([level_s8, feat_rois_cxt_s8]);
    % stride 16
    level_s16 = ones(size(feat_rois_s16, 1),1);
    rois_blob_s16 = single([level_s16, feat_rois_s16]);
    rois_cxt_blob_s16 = single([level_s16, feat_rois_cxt_s16]);
end

function [feat_rois_s4, feat_rois_cxt_s4,feat_rois_s8, feat_rois_cxt_s8,feat_rois_s16, feat_rois_cxt_s16] = map_im_rois_to_feat_rois(conf, rois_s4, rois_s8, rois_s16, scales, im_siz)
    assert(length(scales) == 1);
    
    new_siz = round(im_siz * scales);
    height = new_siz(1);
    width = new_siz(2);

    feat_rois_s4 = bsxfun(@times, rois_s4 - 1, scales) + 1;
    feat_rois_s8 = bsxfun(@times, rois_s8 - 1, scales) + 1;
    feat_rois_s16 = bsxfun(@times, rois_s16 - 1, scales) + 1;
    
    boxes = feat_rois_s4;
    boxes_cxt = feat_rois_s4;
    [feat_rois_s4, feat_rois_cxt_s4] = get_output_box(boxes, boxes_cxt, height, width);
    
    boxes = feat_rois_s8;
    boxes_cxt = feat_rois_s8;
    [feat_rois_s8, feat_rois_cxt_s8] = get_output_box(boxes, boxes_cxt, height, width);
    
    boxes = feat_rois_s16;
    boxes_cxt = feat_rois_s16;
    [feat_rois_s16, feat_rois_cxt_s16] = get_output_box(boxes, boxes_cxt, height, width);
end

function [outbox, outbox_cxt] = get_output_box(boxes, boxes_cxt, height, width)
% ============================= for original box
    leftright_ratio = 0;  %1207: 1--> 0.5
    boxes_width_change = (boxes(:,3)-boxes(:,1) + 1) * leftright_ratio;
    boxes(:,1) = round(boxes(:,1) - boxes_width_change);
    boxes(:,3) = round(boxes(:,3) + boxes_width_change);
    
    top_ratio = 0;
    boxes_top_change = (boxes(:,4)-boxes(:,2) + 1) * top_ratio;
    boxes(:,2) = round(boxes(:,2) - boxes_top_change);
    
    bottom_ratio = 0; %1207: 1.8--> 0.8
    boxes_bottom_change = (boxes(:,4)-boxes(:,2) + 1) * bottom_ratio;
    boxes(:,4) = round(boxes(:,4) + boxes_bottom_change);
    
    boxes(:,1) = max(1, boxes(:,1)); %left
    boxes(:,2) = max(1, boxes(:,2)); %top
    boxes(:,3) = min(width, boxes(:,3)); %right
    boxes(:,4) = min(height, boxes(:,4)); % bottom
    
    outbox = boxes;
    
    % 0322 added
    % ================== for contextual box
    leftright_ratio = 0.5;  %1
    boxes_width_change = (boxes_cxt(:,3)-boxes_cxt(:,1) + 1) * leftright_ratio;
    boxes_cxt(:,1) = round(boxes_cxt(:,1) - boxes_width_change);
    boxes_cxt(:,3) = round(boxes_cxt(:,3) + boxes_width_change);
    
    top_ratio = 0;%0.2
    boxes_top_change = (boxes_cxt(:,4)-boxes_cxt(:,2) + 1) * top_ratio;
    boxes_cxt(:,2) = round(boxes_cxt(:,2) - boxes_top_change);
    
    bottom_ratio = 1; %2
    boxes_bottom_change = (boxes_cxt(:,4)-boxes_cxt(:,2) + 1) * bottom_ratio;
    boxes_cxt(:,4) = round(boxes_cxt(:,4) + boxes_bottom_change);
    
    %new_siz = round(im_siz * im_scale_factor);
    %height = new_siz(1);
    %width = new_siz(2);
    boxes_cxt(:,1) = max(1, boxes_cxt(:,1)); %left
    boxes_cxt(:,2) = max(1, boxes_cxt(:,2)); %top
    boxes_cxt(:,3) = min(width, boxes_cxt(:,3)); %right
    boxes_cxt(:,4) = min(height, boxes_cxt(:,4)); % bottom
    
    outbox_cxt = boxes_cxt;
end

function boxes = clip_boxes(boxes, im_width, im_height)
    % x1 >= 1 & <= im_width
    boxes(:, 1:4:end) = max(min(boxes(:, 1:4:end), im_width), 1);
    % y1 >= 1 & <= im_height
    boxes(:, 2:4:end) = max(min(boxes(:, 2:4:end), im_height), 1);
    % x2 >= 1 & <= im_width
    boxes(:, 3:4:end) = max(min(boxes(:, 3:4:end), im_width), 1);
    % y2 >= 1 & <= im_height
    boxes(:, 4:4:end) = max(min(boxes(:, 4:4:end), im_height), 1);
end
    
