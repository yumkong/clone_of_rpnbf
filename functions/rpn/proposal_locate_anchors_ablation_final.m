function [anchors_s4,anchors_s8,anchors_s16, im_scales] = proposal_locate_anchors_ablation_final(conf, im_size, target_scale, feature_map_size)
% [anchors, im_scales] = proposal_locate_anchors(conf, im_size, target_scale, feature_map_size)
% --------------------------------------------------------
% Faster R-CNN
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------   
% generate anchors for each scale

    % only for fcn
    if ~exist('feature_map_size', 'var')
        feature_map_size = [];
    end

    func = @proposal_locate_anchors_single_scale;

    if exist('target_scale', 'var')
        [anchors_s4,anchors_s8,anchors_s16, im_scales] = func(im_size, conf, target_scale, feature_map_size);
    else
        [anchors_s4,anchors_s8,anchors_s16, im_scales] = arrayfun(@(x) func(im_size, conf, x, feature_map_size), ...
            conf.scales, 'UniformOutput', false);
    end

end

function [anchors_s4,anchors_s8,anchors_s16, im_scale] = proposal_locate_anchors_single_scale(im_size, conf, target_scale, feature_map_size)
    if isempty(feature_map_size)
        im_scale = prep_im_for_blob_size(im_size, target_scale, conf.max_size);
        img_size = round(im_size * im_scale);
		% 1206 added: enlarge the height and width to 8N
        img_size = ceil(img_size/8)*8;
        %output_size = cell2mat([conf.output_height_map.values({img_size(1)}), conf.output_width_map.values({img_size(2)})]);
        output_size_s4 = cell2mat([conf.output_height_s4.values({img_size(1)}), conf.output_width_s4.values({img_size(2)})]);
        output_size_s8 = cell2mat([conf.output_height_s8.values({img_size(1)}), conf.output_width_s8.values({img_size(2)})]);
        output_size_s16 = cell2mat([conf.output_height_s16.values({img_size(1)}), conf.output_width_s16.values({img_size(2)})]);
    else
        im_scale = prep_im_for_blob_size(im_size, target_scale, conf.max_size);
        output_size_s4 = feature_map_size;
    end
    
    shift_x_s4 = [0:(output_size_s4(2)-1)] * conf.feat_stride_s4;
    shift_y_s4 = [0:(output_size_s4(1)-1)] * conf.feat_stride_s4;
    [shift_x_s4, shift_y_s4] = meshgrid(shift_x_s4, shift_y_s4);
    
    % concat anchors as [channel, height, width], where channel is the fastest dimension.
    anchors_s4 = reshape(bsxfun(@plus, permute(conf.anchors_s4, [1, 3, 2]), ...
        permute([shift_x_s4(:), shift_y_s4(:), shift_x_s4(:), shift_y_s4(:)], [3, 1, 2])), [], 4);
    
    % 1112: add conv5 similarly
    shift_x_s8 = [0:(output_size_s8(2)-1)] * conf.feat_stride_s8;
    shift_y_s8 = [0:(output_size_s8(1)-1)] * conf.feat_stride_s8;
    [shift_x_s8, shift_y_s8] = meshgrid(shift_x_s8, shift_y_s8);
    
    % concat anchors as [channel, height, width], where channel is the fastest dimension.
    anchors_s8 = reshape(bsxfun(@plus, permute(conf.anchors_s8, [1, 3, 2]), ...
        permute([shift_x_s8(:), shift_y_s8(:), shift_x_s8(:), shift_y_s8(:)], [3, 1, 2])), [], 4);
    
    % 1122: add conv6 similarly
    shift_x_s16 = [0:(output_size_s16(2)-1)] * conf.feat_stride_s16;
    shift_y_s16 = [0:(output_size_s16(1)-1)] * conf.feat_stride_s16;
    [shift_x_s16, shift_y_s16] = meshgrid(shift_x_s16, shift_y_s16);
    
    % concat anchors as [channel, height, width], where channel is the fastest dimension.
    anchors_s16 = reshape(bsxfun(@plus, permute(conf.anchors_s16, [1, 3, 2]), ...
        permute([shift_x_s16(:), shift_y_s16(:), shift_x_s16(:), shift_y_s16(:)], [3, 1, 2])), [], 4);
%   equals to  
%     anchors = arrayfun(@(x, y) single(bsxfun(@plus, conf.anchors, [x, y, x, y])), shift_x, shift_y, 'UniformOutput', false);
%     anchors = reshape(anchors, [], 1);
%     anchors = cat(1, anchors{:});

end