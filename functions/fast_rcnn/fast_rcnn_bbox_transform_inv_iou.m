function [pred_boxes] = fast_rcnn_bbox_transform_inv_iou(boxes, box_deltas)
% [pred_boxes] = fast_rcnn_bbox_transform_inv_iou(boxes, box_deltas)
% --------------------------------------------------------
% Fast R-CNN
% Reimplementation based on Python Fast R-CNN (https://github.com/rbgirshick/fast-rcnn)
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------
    
    src_w = double(boxes(:, 3) - boxes(:, 1) + 1);
    src_h = double(boxes(:, 4) - boxes(:, 2) + 1);
    src_ctr_x = double(boxes(:, 1) + 0.5*(src_w-1));
    src_ctr_y = double(boxes(:, 2) + 0.5*(src_h-1));
    
    pred_boxes = zeros(size(box_deltas), 'single');
    pred_boxes(:, 1) = src_ctr_x - box_deltas(:,3);
    pred_boxes(:, 2) = src_ctr_y - box_deltas(:,1);
    pred_boxes(:, 3) = src_ctr_x + box_deltas(:,4);
    pred_boxes(:, 4) = src_ctr_y + box_deltas(:,2);
    
end