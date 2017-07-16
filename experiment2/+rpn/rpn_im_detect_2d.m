function [feat_list, label_list] = rpn_im_detect_2d(conf, caffe_net, im, im_roidb, feat_per_img)
% [pred_boxes, scores, box_deltas_, anchors_, scores_] = proposal_im_detect(conf, im, net_idx)
% --------------------------------------------------------
% Faster R-CNN
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------    

    im = single(im);
    [im_blob, ~] = get_image_blob(conf, im);
   
    % permute data into caffe c++ memory, thus [num, channels, height, width]
    im_blob = im_blob(:, :, [3, 2, 1], :); % from rgb to brg
    im_blob = permute(im_blob, [2, 1, 3, 4]);
    im_blob = single(im_blob);

    net_inputs = {im_blob};

    % Reshape net's input blobs
    caffe_net.reshape_as_input(net_inputs);
    output_blobs = caffe_net.forward(net_inputs);

    % Apply bounding-box regression deltas
    % use softmax estimated probabilities
    outfeat = output_blobs{1};
    % permute from [width, height, channel] to [channel, height, width], where channel is the
        % fastest dimension
    outfeat = permute(outfeat, [3, 2, 1]);
    % 0715: 2 x hei x wid => 2 x N(=hei x wid)
    outfeat = reshape(outfeat, size(outfeat, 1), []);
    % 0715: N x 2
    outfeat = outfeat';
    [feat_list, label_list] = sample_rois(conf, im_roidb, outfeat, feat_per_img);
end

function [blob, im_scales] = get_image_blob(conf, im)
    if length(conf.test_scales) == 1
        %0123 changed
        %[blob, im_scales] = prep_im_for_blob_keepsize(im, conf.image_means, conf.test_scales, conf.test_max_size);
        %0124 changed, add a argument of 'input_scale' (set as 1) to be consistent with three inputs (x0.5 x1 x2)
        %[blob, im_scales] = prep_im_for_blob_keepsize(im, conf.image_means, conf.min_test_length, conf.max_test_length);
        [blob, im_scales] = helper.rpn_prep_im_keepsize(im, conf.image_means, 1, conf.min_test_length, conf.max_test_length);
    else
        [ims, im_scales] = arrayfun(@(x) helper.rpn_prep_im_keepsize(im, conf.image_means, x, conf.test_max_size), conf.test_scales, 'UniformOutput', false);
        im_scales = cell2mat(im_scales);
        blob = im_list_to_blob(ims);    
    end
end

% Generate a random sample of ROIs comprising foreground and background examples.
function [feat_list, label_list] = sample_rois(conf, im_roidb,outfeat, feat_per_img)

    bbox_targets = im_roidb.bbox_targets{1};
    % 0715: 1st column is overlap rate
    ex_asign_labels = full(bbox_targets(:, 1));
    
    % Select foreground ROIs as those with >= FG_THRESH overlap
    fg_inds = find(bbox_targets(:, 1) > 0);
    
    % Select background ROIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = find(bbox_targets(:, 1) < 0);
    
    % select foreground when no_ohem
    fg_num = min(feat_per_img, length(fg_inds));
    fg_inds = fg_inds(randperm(length(fg_inds), fg_num));
    bg_num = min(feat_per_img - fg_num, length(bg_inds));
    bg_inds = bg_inds(randperm(length(bg_inds), bg_num));

    feat_list = [outfeat(fg_inds, :); outfeat(bg_inds, :)];
    label_list = [ex_asign_labels(fg_inds, :); ex_asign_labels(bg_inds, :)];
end
