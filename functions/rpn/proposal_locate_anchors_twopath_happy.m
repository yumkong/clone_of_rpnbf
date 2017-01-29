function [anchors_res23,anchors_res45, im_scales] = proposal_locate_anchors_twopath_happy(conf, im_size, target_scale, feat23_size, feat45_size)
% [anchors, im_scales] = proposal_locate_anchors(conf, im_size, target_scale, feature_map_size)
% --------------------------------------------------------
% Faster R-CNN
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------   
% generate anchors for each scale

    % only for fcn
    if ~exist('feat23_size', 'var')
        feat23_size = [];
    end
    if ~exist('feat45_size', 'var')
        feat45_size = [];
    end

    func = @proposal_locate_anchors_single_scale;

    if exist('target_scale', 'var')
        [anchors_res23,anchors_res45, im_scales] = func(im_size, conf, target_scale, feat23_size, feat45_size);
    else
        [anchors_res23,anchors_res45, im_scales] = arrayfun(@(x) func(im_size, conf, x, feat23_size, feat45_size), ...
            conf.scales, 'UniformOutput', false);
    end

end

function [anchors_res23,anchors_res45, im_scale] = proposal_locate_anchors_single_scale(im_size, conf, target_scale, feat23_size, feat45_size)
    if isempty(feat23_size)
        im_scale = prep_im_for_blob_size(im_size, target_scale, conf.max_size);
        img_size = round(im_size * im_scale);
		% 1206 added: enlarge the height and width to 8N
        %img_size = ceil(img_size/8)*8;
        %output_size = cell2mat([conf.output_height_map.values({img_size(1)}), conf.output_width_map.values({img_size(2)})]);
        output_size_res23 = cell2mat([conf.output_height_res23.values({img_size(1)}), conf.output_width_res23.values({img_size(2)})]);
        output_size_res45 = cell2mat([conf.output_height_res45.values({img_size(1)}), conf.output_width_res45.values({img_size(2)})]);
    else
        im_scale = prep_im_for_blob_size(im_size, target_scale, conf.max_size);
        output_size_res23 = feat23_size;
        output_size_res45 = feat45_size;
    end
    
    shift_x_res23 = [0:(output_size_res23(2)-1)] * conf.feat_stride_res23;
    shift_y_res23 = [0:(output_size_res23(1)-1)] * conf.feat_stride_res23;
    [shift_x_res23, shift_y_res23] = meshgrid(shift_x_res23, shift_y_res23);
    
    % concat anchors as [channel, height, width], where channel is the fastest dimension.
    anchors_res23 = reshape(bsxfun(@plus, permute(conf.anchors_res23, [1, 3, 2]), ...
        permute([shift_x_res23(:), shift_y_res23(:), shift_x_res23(:), shift_y_res23(:)], [3, 1, 2])), [], 4);
    
    % 1112: add conv5 similarly
    shift_x_res45 = [0:(output_size_res45(2)-1)] * conf.feat_stride_res45;
    shift_y_res45 = [0:(output_size_res45(1)-1)] * conf.feat_stride_res45;
    [shift_x_res45, shift_y_res45] = meshgrid(shift_x_res45, shift_y_res45);
    
    % concat anchors as [channel, height, width], where channel is the fastest dimension.
    anchors_res45 = reshape(bsxfun(@plus, permute(conf.anchors_res45, [1, 3, 2]), ...
        permute([shift_x_res45(:), shift_y_res45(:), shift_x_res45(:), shift_y_res45(:)], [3, 1, 2])), [], 4);

end