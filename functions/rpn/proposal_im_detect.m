function [pred_boxes, scores, box_deltas_, anchors_, scores_] = proposal_im_detect(conf, caffe_net, im, im_idx)
% [pred_boxes, scores, box_deltas_, anchors_, scores_] = proposal_im_detect(conf, im, net_idx)
% --------------------------------------------------------
% Faster R-CNN
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------    

    im = single(im);
    [im_blob, im_scales] = get_image_blob(conf, im);
    im_size = size(im);
    scaled_im_size = round(im_size * im_scales);
    
    % permute data into caffe c++ memory, thus [num, channels, height, width]
    im_blob = im_blob(:, :, [3, 2, 1], :); % from rgb to brg
    im_blob = permute(im_blob, [2, 1, 3, 4]);
    im_blob = single(im_blob);

    net_inputs = {im_blob};

    % Reshape net's input blobs
    caffe_net.reshape_as_input(net_inputs);
    output_blobs = caffe_net.forward(net_inputs);

    % Apply bounding-box regression deltas
    box_deltas = output_blobs{1};
    featuremap_size = [size(box_deltas, 2), size(box_deltas, 1)];
    % permute from [width, height, channel] to [channel, height, width], where channel is the
        % fastest dimension
    box_deltas = permute(box_deltas, [3, 2, 1]);
    box_deltas = reshape(box_deltas, 4, [])';
    
    anchors = proposal_locate_anchors(conf, size(im), conf.test_scales, featuremap_size);
    pred_boxes = fast_rcnn_bbox_transform_inv(anchors, box_deltas);
      % scale back
    pred_boxes = bsxfun(@times, pred_boxes - 1, ...
        ([im_size(2), im_size(1), im_size(2), im_size(1)] - 1) ./ ([scaled_im_size(2), scaled_im_size(1), scaled_im_size(2), scaled_im_size(1)] - 1)) + 1;
    pred_boxes = clip_boxes(pred_boxes, size(im, 2), size(im, 1));
    
    assert(conf.test_binary == false);
    % use softmax estimated probabilities
    scores = output_blobs{2}(:, :, end);
    scores = reshape(scores, size(output_blobs{1}, 1), size(output_blobs{1}, 2), []);
    % permute from [width, height, channel] to [channel, height, width], where channel is the
        % fastest dimension
    scores = permute(scores, [3, 2, 1]);
    
    % 1029 added
    % also should change the name of 
    %D:\RPN_BF_master\output\VGG16_widerface_conv4\rpn_cachedir\yolo_widerface_VGG16_stage1_rpn\WIDERFACE_test\proposal_boxes_WIDERFACE_test.mat
    % so that algo will re-process each test image one by one
    show_mask = false;
    %1110 added to display the anchor center position as well as plot the
    %512-d vector of each anchor position
    show_mask2 = false;
    debug_position_sensitive = false;
    if show_mask
        scores_max = max(scores, [], 1);
        score_plot = squeeze(scores_max);
        score_plot_resize = imresize(score_plot, [size(im,1) size(im,2)]);
    end
    
    scores = scores(:);

    % 1025: decimate anchors by one half (only keep one boxes out of each anchor scale position)
%     anchor_num = size(conf.anchors, 1);  %14
%     half_anchor_num = size(conf.anchors, 1)/2; %7
%     tmp_scores = reshape(scores, anchor_num, []); 
%     hw1_score = tmp_scores(1:half_anchor_num, :);
%     hw2_score = tmp_scores(1+half_anchor_num:end, :);
%     hw1_greater_mask = (hw1_score >= hw2_score);
%     greater_mask = cat(1, hw1_greater_mask, ~hw1_greater_mask);
%     scores = scores(greater_mask(:),:);  %new scores
%     pred_boxes = pred_boxes(greater_mask(:),:);  % new pred_boxes
    %====== end of 1025
    
    box_deltas_ = box_deltas;
    anchors_ = anchors;
    scores_ = scores;
    
    if conf.test_drop_boxes_runoff_image
        contained_in_image = is_contain_in_image(anchors, round(size(im) * im_scales));
        pred_boxes = pred_boxes(contained_in_image, :);
        scores = scores(contained_in_image, :);
    end
    
    % drop too small boxes
    %1006 changed to get rid of too small boxes
    %[pred_boxes, scores] = filter_boxes(conf.test_min_box_size-3, pred_boxes, scores);
    [pred_boxes, scores] = filter_boxes(conf.test_min_box_size-3, pred_boxes, scores);
    
    % sort
    [scores, scores_ind] = sort(scores, 'descend');
    pred_boxes = pred_boxes(scores_ind, :);
    
    if show_mask
        figure(1), imshow(im/255);
        %figure(2), imshow(score_plot_resize);
        figure(2), h = imshow(im/255);
        set(h,'AlphaData',score_plot_resize);
    end
    
    if show_mask2
        %plot position of anchors:
        anchor_pos = [0.5*(anchors(:,1)+anchors(:,3)) 0.5*(anchors(:,2)+anchors(:,4))];
        anchor_pos = anchor_pos(1:7:end,:);
        re_im = imresize(im, im_scales);
        figure(1), imshow(re_im/255);
        
        [x_pos, y_pos] = meshgrid(1:featuremap_size(2),1:featuremap_size(1));
        rectangle_anchor_pos = reshape(anchor_pos, featuremap_size(1), featuremap_size(2), 2);
        rectangle_anchor_pos = cat(3, rectangle_anchor_pos, y_pos, x_pos);
        sample_rectangle_anchor_pos = rectangle_anchor_pos(1:10:end, 1:10:end, :);
        sample_rectangle_anchor_pos = reshape(sample_rectangle_anchor_pos, [], 4);
        pos_info_cell = num2cell(sample_rectangle_anchor_pos(:,3:4), 2);
        pos_info_txt = cellfun(@(x) sprintf('(%d,%d)',x(1), x(2)), pos_info_cell, 'UniformOutput',false);
        
        hold on
        text(sample_rectangle_anchor_pos(:,1),sample_rectangle_anchor_pos(:,2),pos_info_txt, 'Color', 'c','HorizontalAlignment','center');
        plot(anchor_pos(:,1), anchor_pos(:,2),'g.','MarkerSize',5);
        hold off
    end
    if debug_position_sensitive
        layer_name = 'conv4_3';%conv_proposal1
        layer_mat = caffe_net.blobs(layer_name).get_data();
        % from [w x h x feat_dim] to [h x w x feat_dim]
        layer_mat = permute(layer_mat, [2, 1, 3]);
        posY1 = 36; posX1 = 76; %77
        posY2 = 56; posX2 = 18; 
        figure(2);
        subplot(2,1,1),plot(squeeze(layer_mat(posY1, posX1, :)));
        subplot(2,1,2),plot(squeeze(layer_mat(posY2, posX2, :)));
    end
%     image(im/255); 
%     axis image;
%     axis off;
%     set(gcf, 'Color', 'white');
%     endNum = sum(scores >= 0.9);
%     for i = 1:endNum  % can be changed to any positive number to show different #proposals
%         bbox = pred_boxes(i,:);
%         rect = [bbox(:, 1), bbox(:, 2), bbox(:, 3)-bbox(:,1)+1, bbox(:,4)-bbox(2)+1];
%         rectangle('Position', rect, 'LineWidth', 2, 'EdgeColor', [0 1 0]);
%     end
%     saveName = sprintf('val_res\\img_%d_pro200',im_idx);
%     export_fig(saveName, '-png', '-a1', '-native');
%     fprintf('image %d saved.\n', im_idx);
end

function [data_blob, rois_blob, im_scale_factors] = get_blobs(conf, im, rois)
    [data_blob, im_scale_factors] = get_image_blob(conf, im);
    rois_blob = get_rois_blob(conf, rois, im_scale_factors);
end

function [blob, im_scales] = get_image_blob(conf, im)
    if length(conf.test_scales) == 1
        [blob, im_scales] = prep_im_for_blob(im, conf.image_means, conf.test_scales, conf.test_max_size);
    else
        [ims, im_scales] = arrayfun(@(x) prep_im_for_blob(im, conf.image_means, x, conf.test_max_size), conf.test_scales, 'UniformOutput', false);
        im_scales = cell2mat(im_scales);
        blob = im_list_to_blob(ims);    
    end
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
        scaled_areas = bsxfun(@times, areas(:), scales(:)'.^2);
        levels = max(abs(scaled_areas - 224.^2), 2); 
    else
        levels = ones(size(im_rois, 1), 1);
    end
    
    feat_rois = round(bsxfun(@times, im_rois-1, scales(levels)) / conf.feat_stride) + 1;
end

function [boxes, scores] = filter_boxes(min_box_size, boxes, scores)
    widths = boxes(:, 3) - boxes(:, 1) + 1;
    heights = boxes(:, 4) - boxes(:, 2) + 1;
    
    valid_ind = widths >= min_box_size & heights >= min_box_size;
    boxes = boxes(valid_ind, :);
    scores = scores(valid_ind, :);
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

function contained = is_contain_in_image(boxes, im_size)
    contained = boxes >= 1 & bsxfun(@le, boxes, [im_size(2), im_size(1), im_size(2), im_size(1)]);
    
    contained = all(contained, 2);
end
    
