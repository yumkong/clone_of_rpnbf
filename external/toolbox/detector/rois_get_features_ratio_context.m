function [feats] = rois_get_features_ratio_context(conf, caffe_net, im, boxes, max_rois_num_in_gpu, ratio)
% [pred_boxes, scores] = fast_rcnn_im_detect(conf, caffe_net, im, boxes, max_rois_num_in_gpu)
% --------------------------------------------------------
% Fast R-CNN
% Reimplementation based on Python Fast R-CNN (https://github.com/rbgirshick/fast-rcnn)
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------
% 1027 added for double rois feature (small context: [l r t b] = [0.5w 0.5w 0.2h 0.8h]
%                                    + big  context: [l r t b] = [1w   1w   0.2h  3h ])

    boxes_cxt = boxes; %1027 added full body context
    % *** 1215 commented: Part I: generate features from pure face regions
%     % liu@1001: ratio = 1, so no bbox width and height changes
%     outer_box_ratio = ratio;
%     %1004 changed
%     boxes_width_change = (boxes(:,3)-boxes(:,1) + 1)*(outer_box_ratio-1)/2;
%     boxes(:,1) = boxes(:,1) - boxes_width_change;
%     boxes(:,3) = boxes(:,3) + boxes_width_change;
%     top_ratio = 0.2; %0.25
%     bottom_ratio = 0.8; %1.75
%     boxes_top_change = (boxes(:,4)-boxes(:,2) + 1) * top_ratio;
%     boxes_bottom_change = (boxes(:,4)-boxes(:,2) + 1) * bottom_ratio;
%     boxes(:,2) = boxes(:,2) - boxes_top_change;
%     boxes(:,4) = boxes(:,4) + boxes_bottom_change;
%     [height, width, ~] = size(im);
%     boxes(:,1) = max(1, boxes(:,1)); %left
%     boxes(:,2) = max(1, boxes(:,2)); %top
%     boxes(:,3) = min(width, boxes(:,3)); %right
%     boxes(:,4) = min(height, boxes(:,4)); % bottom
    
    %========1027 add begin========================
    % *** 1215: generate features from context regions [-w -0.2h w 2h]
    boxes_width_change = (boxes_cxt(:,3)-boxes_cxt(:,1) + 1);
    boxes_cxt(:,1) = boxes_cxt(:,1) - boxes_width_change;
    boxes_cxt(:,3) = boxes_cxt(:,3) + boxes_width_change;

    top_ratio = 0.2; %1229: 0.2 --> 1
    bottom_ratio = 2; % 1229: 2 --> 1
    boxes_top_change = (boxes_cxt(:,4)-boxes_cxt(:,2) + 1) * top_ratio;
    boxes_bottom_change = (boxes_cxt(:,4)-boxes_cxt(:,2) + 1) * bottom_ratio;
    boxes_cxt(:,2) = boxes_cxt(:,2) - boxes_top_change;
    boxes_cxt(:,4) = boxes_cxt(:,4) + boxes_bottom_change;
    [height, width, ~] = size(im);

    boxes_cxt(:,1) = max(1, boxes_cxt(:,1)); %left
    boxes_cxt(:,2) = max(1, boxes_cxt(:,2)); %top
    boxes_cxt(:,3) = min(width, boxes_cxt(:,3)); %right
    boxes_cxt(:,4) = min(height, boxes_cxt(:,4)); % bottom
    boxes_full = cat(1, boxes, boxes_cxt);
    
    [im_blob, rois_blob, ~] = get_blobs(conf, im, boxes_full);
    
    % When mapping from image ROIs to feature map ROIs, there's some aliasing
    % (some distinct image ROIs get mapped to the same feature ROI).
    % Here, we identify duplicate feature ROIs, so we only compute features
    % on the unique subset.
    %[~, index, inv_index] = unique(rois_blob, 'rows');
    %rois_blob = rois_blob(index, :);
    %boxes_full = boxes_full(index, :);
    
    % permute data into caffe c++ memory, thus [num, channels, height, width]
    im_blob = im_blob(:, :, [3, 2, 1], :); % from rgb to brg
    im_blob = permute(im_blob, [2, 1, 3, 4]);
    im_blob = single(im_blob);
    rois_blob = rois_blob - 1; % to c's index (start from 0)
    rois_blob = permute(rois_blob, [3, 4, 2, 1]);
    rois_blob = single(rois_blob);
    
    total_rois = size(rois_blob, 4);
    %========1027 add end=========================
%     total_scores = cell(ceil(total_rois / max_rois_num_in_gpu), 1);
%     total_box_deltas = cell(ceil(total_rois / max_rois_num_in_gpu), 1);
%     
    
    total_feats =  cell(ceil(total_rois / max_rois_num_in_gpu), 1);
    for i = 1:ceil(total_rois / max_rois_num_in_gpu)
        
        sub_ind_start = 1 + (i-1) * max_rois_num_in_gpu;
        sub_ind_end = min(total_rois, i * max_rois_num_in_gpu);
        sub_rois_blob = rois_blob(:, :, :, sub_ind_start:sub_ind_end);
        
        
        net_inputs = {im_blob, sub_rois_blob};
        
        
        % Reshape net's input blobs
        caffe_net.reshape_as_input(net_inputs);
        
        
        output_blobs = caffe_net.forward(net_inputs);
        
% 
%         feats = output_blobs{1};
%         feats = permute(feats, [4 2 1 3]);
%         total_feats{i} = feats;
        
        % support networks with multiple outputs
%         tic;
       
        if length(output_blobs) == 1 && length(size(output_blobs{1})) == 2
            total_feats{i} = output_blobs{1}';   %'
        else
            for j = 1:length(output_blobs)
                feats = output_blobs{j};
                feats = permute(feats, [4 2 1 3]);
                feats = reshape(feats, size(feats, 1), size(feats, 2)*size(feats, 3)*size(feats, 4));
                total_feats{i} = [total_feats{i} feats];
            end
        end
%         toc;

%         tic;
%         feat_len = sum(cellfun(@(x) numel(x) / size(x, 4), output_blobs));
%         total_feats{i} = zeros(size(rois_blob, 4), feat_len);
%         idx = 1;
%         for j = 1:length(output_blobs)
%             feats = output_blobs{j};
% %             feats = permute(feats, [4 2 1 3]);
% %             feats = reshape(feats, size(feats, 1), size(feats, 2)*size(feats, 3)*size(feats, 4));
%             
%             feats = reshape(feats, numel(feats) / size(feats, 4), size(feats, 4));
%             feats = feats';
%             total_feats{i}(:, idx:idx+size(feats,2)-1) = feats;
%             idx = idx + size(feats,2);
%         end
%         toc;

    end 
    
    feats = cell2mat(total_feats);
    %feats = feats(inv_index, :);
    
    %1027 added
    half_feat_num = total_rois / 2;
    feats = cat(2, feats(1:half_feat_num,:), feats(1+half_feat_num:end,:));
end

function [data_blob, rois_blob, im_scale_factors] = get_blobs(conf, im, rois)
    [data_blob, im_scale_factors] = get_image_blob(conf, im);
    rois_blob = get_rois_blob(conf, rois, im_scale_factors);
end

function [blob, im_scales] = get_image_blob(conf, im)
    [ims, im_scales] = arrayfun(@(x) prep_im_for_blob(im, conf.image_means, x, conf.test_max_size), conf.test_scales, 'UniformOutput', false);
    im_scales = cell2mat(im_scales);
    blob = im_list_to_blob(ims);    
end

function [rois_blob] = get_rois_blob(conf, im_rois, im_scale_factors)
    [feat_rois, levels] = map_im_rois_to_feat_rois(conf, im_rois, im_scale_factors);
    rois_blob = single([levels, feat_rois]);
end

function [feat_rois, levels] = map_im_rois_to_feat_rois(conf, im_rois, scales)
    im_rois = single(im_rois);
    
    if length(scales) > 1
        widths = im_rois(:, 3) - im_rois(:, 1) + 1;
        heights = im_rois(:, 4) - im_rois(:, 2) + 1;
        
        areas = widths .* heights;
        scaled_areas = bsxfun(@times, areas(:), scales(:)'.^2);  %'
        levels = max(abs(scaled_areas - 224.^2), 2); 
    else
        levels = ones(size(im_rois, 1), 1);
    end
    
    feat_rois = round(bsxfun(@times, im_rois-1, scales(levels))) + 1;
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
    