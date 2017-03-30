function [im_blob, rois_blob_s4, rois_cxt_blob_s4,rois_blob_s8, rois_cxt_blob_s8,rois_blob_s16, rois_cxt_blob_s16, labels_s4, labels_s8, labels_s16] = fast_rcnn_get_minibatch_mpfvn_cxt_realmp(conf, image_roidb)
% [im_blob, rois_blob, labels_blob, bbox_targets_blob, bbox_loss_blob] ...
%    = fast_rcnn_get_minibatch(conf, image_roidb)
% --------------------------------------------------------
% Fast R-CNN
% Reimplementation based on Python Fast R-CNN (https://github.com/rbgirshick/fast-rcnn)
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------

    num_images = length(image_roidb);
    % Infer number of classes from the number of columns in gt_overlaps
    num_classes = size(image_roidb(1).overlap, 2);
    % Sample random scales to use for each image in this batch
    random_scale_inds = randi(length(conf.scales), num_images, 1);
    
    assert(mod(conf.batch_size_s4, num_images) == 0, ...
        sprintf('num_images %d must divide BATCH_SIZE %d', num_images, conf.batch_size_s4));
    
    %s4
    rois_per_image_s4 = conf.batch_size_s4 / num_images;
    fg_rois_per_image_s4 = round(rois_per_image_s4 * conf.fg_fraction);
    %s8
    rois_per_image_s8 = conf.batch_size_s8 / num_images;
    fg_rois_per_image_s8 = round(rois_per_image_s8 * conf.fg_fraction);
    %s16
    rois_per_image_s16 = conf.batch_size_s16 / num_images;
    fg_rois_per_image_s16 = round(rois_per_image_s16 * conf.fg_fraction);
    
    % Get the input image blob
    [im_blob, im_scales] = get_image_blob(conf, image_roidb, random_scale_inds);
    
    % build the region of interest and label blobs
    rois_blob_s4 = zeros(0, 5, 'single');
    %0325 added
    rois_blob_s8 = zeros(0, 5, 'single');
    rois_blob_s16 = zeros(0, 5, 'single');
    %0322 added
    rois_cxt_blob_s4 = zeros(0, 5, 'single');
    rois_cxt_blob_s8 = zeros(0, 5, 'single');
    rois_cxt_blob_s16 = zeros(0, 5, 'single');
    
    labels_s4 = zeros(0, 1, 'single');
    labels_s8 = zeros(0, 1, 'single');
    labels_s16 = zeros(0, 1, 'single');
    
    for i = 1:num_images
        [l_s4,l_s8,l_s16, ~, im_rois_s4, im_rois_s8, im_rois_s16] = sample_rois(conf, image_roidb(i), fg_rois_per_image_s4, rois_per_image_s4, ...
                                             fg_rois_per_image_s8, rois_per_image_s8, fg_rois_per_image_s16, rois_per_image_s16);
        
        % Add to ROIs blob
        %1207: extend to [left, top, right, bottom] = [-0.5 -0.2 0.5 0.8]
        [feat_rois_s4, feat_cxt_rois_s4, feat_rois_s8, feat_cxt_rois_s8, feat_rois_s16, feat_cxt_rois_s16] = fast_rcnn_map_im_rois_to_feat_rois_mpfvn(conf, im_rois_s4, im_rois_s8, im_rois_s16, im_scales(i), image_roidb(i).im_size);
        %feat_rois = fast_rcnn_map_im_rois_to_feat_rois(conf, im_rois, im_scales(i));
        % for stride4
        %  *self
        batch_ind = i * ones(size(feat_rois_s4, 1), 1);
        rois_blob_this_image_s4 = [batch_ind, feat_rois_s4];
        rois_blob_s4 = [rois_blob_s4; rois_blob_this_image_s4];
        % * context
        rois_cxt_blob_this_image_s4 = [batch_ind, feat_cxt_rois_s4];
        rois_cxt_blob_s4 = [rois_cxt_blob_s4; rois_cxt_blob_this_image_s4];
        % for stride8
        %  *self
        batch_ind = i * ones(size(feat_rois_s8, 1), 1);
        rois_blob_this_image_s8 = [batch_ind, feat_rois_s8];
        rois_blob_s8 = [rois_blob_s8; rois_blob_this_image_s8];
        % * context
        rois_cxt_blob_this_image_s8 = [batch_ind, feat_cxt_rois_s8];
        rois_cxt_blob_s8 = [rois_cxt_blob_s8; rois_cxt_blob_this_image_s8];
        % for stride4
        %  *self
        batch_ind = i * ones(size(feat_rois_s16, 1), 1);
        rois_blob_this_image_s16 = [batch_ind, feat_rois_s16];
        rois_blob_s16 = [rois_blob_s16; rois_blob_this_image_s16];
        % * context
        rois_cxt_blob_this_image_s16 = [batch_ind, feat_cxt_rois_s16];
        rois_cxt_blob_s16 = [rois_cxt_blob_s16; rois_cxt_blob_this_image_s16];
        
        % Add to labels, bbox targets, and bbox loss blobs
        labels_s4 = [labels_s4; l_s4];
        labels_s8 = [labels_s8; l_s8];
        labels_s16 = [labels_s16; l_s16];
    end
    
    % permute data into caffe c++ memory, thus [num, channels, height, width]
    im_blob = im_blob(:, :, [3, 2, 1], :); % from rgb to brg
    im_blob = single(permute(im_blob, [2, 1, 3, 4]));
    
    %for stride4
    rois_blob_s4 = rois_blob_s4 - 1; % to c's index (start from 0)
    rois_blob_s4 = single(permute(rois_blob_s4, [3, 4, 2, 1]));
    rois_cxt_blob_s4 = rois_cxt_blob_s4 - 1; % to c's index (start from 0)
    rois_cxt_blob_s4 = single(permute(rois_cxt_blob_s4, [3, 4, 2, 1]));
    
    %for stride8
    rois_blob_s8 = rois_blob_s8 - 1; % to c's index (start from 0)
    rois_blob_s8= single(permute(rois_blob_s8, [3, 4, 2, 1]));
    rois_cxt_blob_s8 = rois_cxt_blob_s8 - 1; % to c's index (start from 0)
    rois_cxt_blob_s8 = single(permute(rois_cxt_blob_s8, [3, 4, 2, 1]));
    
    %for stride16
    rois_blob_s16 = rois_blob_s16 - 1; % to c's index (start from 0)
    rois_blob_s16 = single(permute(rois_blob_s16, [3, 4, 2, 1]));
    rois_cxt_blob_s16 = rois_cxt_blob_s16 - 1; % to c's index (start from 0)
    rois_cxt_blob_s16 = single(permute(rois_cxt_blob_s16, [3, 4, 2, 1]));
    
    %labels_blob = single(permute(labels_blob, [3, 4, 2, 1]));
    labels_s4 = single(permute(labels_s4, [3, 4, 2, 1]));
    labels_s8 = single(permute(labels_s8, [3, 4, 2, 1]));
    labels_s16 = single(permute(labels_s16, [3, 4, 2, 1]));
    
    %0325 can we remove this limitation because it is too strong?
    assert(~isempty(im_blob));
    %assert(~isempty(rois_blob));
    assert(~isempty(rois_blob_s4));
    assert(~isempty(rois_cxt_blob_s4));
    assert(~isempty(rois_blob_s8));
    assert(~isempty(rois_cxt_blob_s8));
    assert(~isempty(rois_blob_s16));
    assert(~isempty(rois_cxt_blob_s16));
    assert(~isempty(labels_s4));
    assert(~isempty(labels_s8));
    assert(~isempty(labels_s16));
end

%% Build an input blob from the images in the roidb at the specified scales.
function [im_blob, im_scales] = get_image_blob(conf, images, random_scale_inds)
    
    num_images = length(images);
    processed_ims = cell(num_images, 1);
    im_scales = nan(num_images, 1);
    for i = 1:num_images
        im = imread(images(i).image_path);
        target_size = conf.scales(random_scale_inds(i));
        
        [im, im_scale] = prep_im_for_blob(im, conf.image_means, target_size, conf.max_size);
        
        im_scales(i) = im_scale;
        processed_ims{i} = im; 
    end
    
    im_blob = im_list_to_blob(processed_ims);
end

%% Generate a random sample of ROIs comprising foreground and background examples.
function [label_s4,label_s8,label_s16, overlaps, rois_s4, rois_s8, rois_s16] = sample_rois(conf, image_roidb, fg_rois_per_image_s4, rois_per_image_s4, ...
                                            fg_rois_per_image_s8, rois_per_image_s8, fg_rois_per_image_s16, rois_per_image_s16)
    % 0325 range: s4: 6-12, s8: 12-160, s16: 90 -
    [overlaps, labels] = max(image_roidb(1).overlap, [], 2);
%     labels = image_roidb(1).max_classes;
%     overlaps = image_roidb(1).max_overlaps;
    rois = image_roidb(1).boxes;
    face_hei = rois(:,4) - rois(:,2) + 1;
    ind_s4 = find(face_hei >= 6 & face_hei <= 12);
    ind_s8 = find(face_hei >= 12 & face_hei <= 160);
    ind_s16 = find(face_hei >= 90);
    
    overlap_s4 = overlaps(ind_s4,:);
    overlap_s8 = overlaps(ind_s8,:);
    overlap_s16 = overlaps(ind_s16,:);
    % ################################################3
    % Select foreground ROIs as those with >= FG_THRESH overlap
    fg_ind_s4 = find(overlap_s4 >= conf.fg_thresh);
    % Guard against the case when an image has fewer than fg_rois_per_image
    % foreground ROIs
    fg_rois_per_this_image_s4 = min(fg_rois_per_image_s4, length(fg_ind_s4));
    % Sample foreground regions without replacement
    if ~isempty(fg_ind_s4)
       fg_ind_s4 = fg_ind_s4(randperm(length(fg_ind_s4), fg_rois_per_this_image_s4));
    end
    
    % Select background ROIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_ind_s4 = find(overlap_s4 < conf.bg_thresh_hi & overlap_s4 >= conf.bg_thresh_lo);
    % Compute number of background ROIs to take from this image (guarding
    % against there being fewer than desired)
    bg_rois_per_this_image_s4 = rois_per_image_s4 - fg_rois_per_this_image_s4;
    bg_rois_per_this_image_s4 = min(bg_rois_per_this_image_s4, length(bg_ind_s4));
    % Sample foreground regions without replacement
    if ~isempty(bg_ind_s4)
       bg_ind_s4 = bg_ind_s4(randperm(length(bg_ind_s4), bg_rois_per_this_image_s4));
    end
    
    if length(fg_ind_s4) + length(bg_ind_s4) > 0
        % The indices that we're selecting (both fg and bg)
        keep_ind_s4 = [ind_s4(fg_ind_s4,:); ind_s4(bg_ind_s4,:)];
        % Select sampled values from various arrays
        label_s4 = labels(keep_ind_s4);
        % Clamp labels for the background ROIs to 0
        label_s4((length(fg_ind_s4)+1):end) = 0;
        overlap_s4 = overlaps(keep_ind_s4);
        rois_s4 = rois(keep_ind_s4, :);
    else
        % if no available roi, just create a 'negative' one
        label_s4 = 0;
        overlap_s4 = 0;
        rois_s4 = single([66 66 76 76]);
    end
    % ################################### stride 8
    fg_ind_s8 = find(overlap_s8 >= conf.fg_thresh);
    % Guard against the case when an image has fewer than fg_rois_per_image
    % foreground ROIs
    fg_rois_per_this_image_s8 = min(fg_rois_per_image_s8, length(fg_ind_s8));
    % Sample foreground regions without replacement
    if ~isempty(fg_ind_s8)
       fg_ind_s8 = fg_ind_s8(randperm(length(fg_ind_s8), fg_rois_per_this_image_s8));
    end
    
    % Select background ROIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_ind_s8 = find(overlap_s8 < conf.bg_thresh_hi & overlap_s8 >= conf.bg_thresh_lo);
    % Compute number of background ROIs to take from this image (guarding
    % against there being fewer than desired)
    bg_rois_per_this_image_s8 = rois_per_image_s8 - fg_rois_per_this_image_s8;
    bg_rois_per_this_image_s8 = min(bg_rois_per_this_image_s8, length(bg_ind_s8));
    % Sample foreground regions without replacement
    if ~isempty(bg_ind_s8)
       bg_ind_s8 = bg_ind_s8(randperm(length(bg_ind_s8), bg_rois_per_this_image_s8));
    end
    
    if length(fg_ind_s8) + length(bg_ind_s8) > 0
        % The indices that we're selecting (both fg and bg)
        keep_ind_s8 = [ind_s8(fg_ind_s8,:); ind_s8(bg_ind_s8,:)];
        % Select sampled values from various arrays
        label_s8 = labels(keep_ind_s8);
        % Clamp labels for the background ROIs to 0
        label_s8((length(fg_ind_s8)+1):end) = 0;
        overlap_s8 = overlaps(keep_ind_s8);
        rois_s8 = rois(keep_ind_s8, :);
    else
        % if no available roi, just create a 'negative' one
        label_s8 = 0;
        overlap_s8 = 0;
        rois_s8 = single([66 66 126 126]);
    end
    % ################################### stride 16
    fg_ind_s16 = find(overlap_s16 >= conf.fg_thresh);
    % Guard against the case when an image has fewer than fg_rois_per_image
    % foreground ROIs
    fg_rois_per_this_image_s16 = min(fg_rois_per_image_s16, length(fg_ind_s16));
    % Sample foreground regions without replacement
    if ~isempty(fg_ind_s16)
       fg_ind_s16 = fg_ind_s16(randperm(length(fg_ind_s16), fg_rois_per_this_image_s16));
    end
    
    % Select background ROIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_ind_s16 = find(overlap_s16 < conf.bg_thresh_hi & overlap_s16 >= conf.bg_thresh_lo);
    % Compute number of background ROIs to take from this image (guarding
    % against there being fewer than desired)
    bg_rois_per_this_image_s16 = rois_per_image_s16 - fg_rois_per_this_image_s16;
    bg_rois_per_this_image_s16 = min(bg_rois_per_this_image_s16, length(bg_ind_s16));
    % Sample foreground regions without replacement
    if ~isempty(bg_ind_s16)
       bg_ind_s16 = bg_ind_s16(randperm(length(bg_ind_s16), bg_rois_per_this_image_s16));
    end
    
    if length(fg_ind_s16) + length(bg_ind_s16) > 0
        % The indices that we're selecting (both fg and bg)
        keep_ind_s16 = [ind_s16(fg_ind_s16,:); ind_s16(bg_ind_s16,:)];
        % Select sampled values from various arrays
        label_s16 = labels(keep_ind_s16);
        % Clamp labels for the background ROIs to 0
        label_s16((length(fg_ind_s16)+1):end) = 0;
        overlap_s16 = overlaps(keep_ind_s16);
        rois_s16 = rois(keep_ind_s16, :);
    else
        % if no available roi, just create a 'negative' one
        label_s16 = 0;
        overlap_s16 = 0;
        rois_s16 = single([66 66 226 226]);
    end
    
    %0325 combine all strides
    %labels = cat(1, label_s4, label_s8, label_s16);
    overlaps = cat(1, overlap_s4, overlap_s8, overlap_s16);
end
