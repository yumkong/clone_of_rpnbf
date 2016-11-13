function [regression_label] = fast_rcnn_bbox_transform_iou(ex_boxes, gt_boxes)
% [regression_label] = fast_rcnn_bbox_transform_iou(ex_boxes, gt_boxes)
% --------------------------------------------------------
% Fast R-CNN
% Reimplementation based on Python Fast R-CNN (https://github.com/rbgirshick/fast-rcnn)
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------

    ex_widths = ex_boxes(:, 3) - ex_boxes(:, 1) + 1;
    ex_heights = ex_boxes(:, 4) - ex_boxes(:, 2) + 1;
    ex_ctr_x = ex_boxes(:, 1) + 0.5 * (ex_widths - 1);
    ex_ctr_y = ex_boxes(:, 2) + 0.5 * (ex_heights - 1);
    
    targets_dt = max(ex_ctr_y - gt_boxes(:, 2), 0);  % mid-top distance
    targets_db = max(gt_boxes(:, 4) - ex_ctr_y, 0);  % bottom-mid distance
    targets_dl = max(ex_ctr_x - gt_boxes(:, 1), 0);  % mid-left distance
    targets_dr = max(gt_boxes(:, 3) - ex_ctr_x, 0);  % right-mid distance
    
    % 1101 normalize them tp [0 1]
    targets_dt_norm = targets_dt; % ./ (targets_dt + targets_db);
    targets_db_norm = targets_db; % ./ (targets_dt + targets_db);
    targets_dl_norm = targets_dl; % ./ (targets_dl + targets_dr);
    targets_dr_norm = targets_dr; % ./ (targets_dl + targets_dr);
    
    regression_label = [targets_dt_norm, targets_db_norm, targets_dl_norm, targets_dr_norm];
end