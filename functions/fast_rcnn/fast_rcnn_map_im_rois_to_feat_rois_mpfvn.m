function [feat_rois_s4, feat_cxt_rois_s4, feat_rois_s8, feat_cxt_rois_s8, feat_rois_s16, feat_cxt_rois_s16] = fast_rcnn_map_im_rois_to_feat_rois_mpfvn(conf,...
                                                                                    im_rois_s4, im_rois_s8, im_rois_s16, im_scale_factor, im_siz)
% [feat_rois] = fast_rcnn_map_im_rois_to_feat_rois(conf, im_rois, im_scale_factor)
% --------------------------------------------------------
% Fast R-CNN
% Reimplementation based on Python Fast R-CNN (https://github.com/rbgirshick/fast-rcnn)
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------

%% Map a ROI in image-pixel coordinates to a ROI in feature coordinates.
% in matlab's index (start from 1)
    new_siz = round(im_siz * im_scale_factor);
    height = new_siz(1);
    width = new_siz(2);

    % =======================================================for s4
    feat_rois_s4 = (im_rois_s4 - 1) * im_scale_factor + 1;
    boxes = feat_rois_s4;
    %0322 added
    boxes_cxt = feat_rois_s4;
    [feat_rois_s4, feat_cxt_rois_s4] = get_output_box(boxes, boxes_cxt, height, width);
    
    % =======================================================for s8
    feat_rois_s8 = (im_rois_s8 - 1) * im_scale_factor + 1;
    boxes = feat_rois_s8;
    %0322 added
    boxes_cxt = feat_rois_s8;
    [feat_rois_s8, feat_cxt_rois_s8] = get_output_box(boxes, boxes_cxt, height, width);
    
    % =======================================================for s16
    feat_rois_s16 = (im_rois_s16 - 1) * im_scale_factor + 1;
    boxes = feat_rois_s16;
    %0322 added
    boxes_cxt = feat_rois_s16;
    [feat_rois_s16, feat_cxt_rois_s16] = get_output_box(boxes, boxes_cxt, height, width);

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