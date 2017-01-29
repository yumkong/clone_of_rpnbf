function [pred_boxes_all, scores_all] = proposal_im_detect_conv3_4_scale3(conf, caffe_net, im, im_idx)
% [pred_boxes, scores, box_deltas_, anchors_, scores_] = proposal_im_detect(conf, im, net_idx)
% --------------------------------------------------------
% Faster R-CNN
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------    

    im = single(im);
    [im_blob, im_scales] = get_image_blob(conf, im);
    im_size = size(im);
    scaled_im_size = round(im_scales' * im_size);
    im_blob_all = im_blob;
    scaled_im_size_all = scaled_im_size;
    %0124 added
    pred_boxes_all = cell(1, numel(im_blob_all));
    scores_all = cell(1, numel(im_blob_all));
    
    for i = 1:numel(im_blob_all)
        %0124 send an individual image and scaled size here
        im_blob = im_blob_all{i};
        scaled_im_size = scaled_im_size_all(i,:);
        % 0124 added. treat small and medium size images as a whole
        if (size(im_blob, 1) <= 1500) && (size(im_blob, 2) <= 1500)
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
            % constrain bboxes within im_wid and im_hei
            pred_boxes = clip_boxes(pred_boxes, size(im, 2), size(im, 1));

            assert(conf.test_binary == false);
            % use softmax estimated probabilities
            scores = output_blobs{2}(:, :, end);
            scores = reshape(scores, size(output_blobs{1}, 1), size(output_blobs{1}, 2), []);
            % permute from [width, height, channel] to [channel, height, width], where channel is the
                % fastest dimension
            scores = permute(scores, [3, 2, 1]);

            scores = scores(:);

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
            % 0124: assign them to output
            pred_boxes_all{i} = pred_boxes;
            scores_all{i} = scores;
        else % 0124 added. treat large size (2x) images as four parts
            hei_im = size(im_blob, 1);
            wid_im = size(im_blob, 2);
            hei_middle = round(hei_im/2);
            wid_middle = round(wid_im/2);
            % [top-left bottom-left top-right bottom-right]
            %part_h = hei_middle + 16;
            %part_w = wid_middle + 16;
            % start position, also the offset position of bboxes
            y_start = [1 hei_middle-16 1 hei_middle-16];
            x_start = [1 1 wid_middle-16 wid_middle-16];
            % end position
            y_end = [hei_middle+16 hei_im hei_middle+16 hei_im];
            x_end = [wid_middle+16 wid_middle+16 wid_im wid_im];
            
            im_blob_complete = im_blob;
            scores_tmp = [];
            pred_boxes_tmp = [];
            for kk = 1:4
                im_blob = im_blob_complete(y_start(kk):y_end(kk), x_start(kk):x_end(kk), :);
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

                %anchors = proposal_locate_anchors(conf, size(im), conf.test_scales, featuremap_size);
                anchors = proposal_locate_anchors(conf, size(im_blob), conf.test_scales, featuremap_size);
                pred_boxes = fast_rcnn_bbox_transform_inv(anchors, box_deltas);
                % scale back
                pred_boxes = bsxfun(@times, pred_boxes - 1, ...
                    ([im_size(2), im_size(1), im_size(2), im_size(1)] - 1) ./ ([scaled_im_size(2), scaled_im_size(1), scaled_im_size(2), scaled_im_size(1)] - 1)) + 1;
                % constrain bboxes within im_wid and im_hei
                %0124 changed from original im size to cropped size
                %pred_boxes = clip_boxes(pred_boxes, size(im, 2), size(im, 1));
                pred_boxes = clip_boxes(pred_boxes, size(im_blob, 2), size(im_blob, 1));

                assert(conf.test_binary == false);
                % use softmax estimated probabilities
                scores = output_blobs{2}(:, :, end);
                scores = reshape(scores, size(output_blobs{1}, 1), size(output_blobs{1}, 2), []);
                % permute from [width, height, channel] to [channel, height, width], where channel is the
                    % fastest dimension
                scores = permute(scores, [3, 2, 1]);

                scores = scores(:);
                % false, not do this
                if conf.test_drop_boxes_runoff_image
                    contained_in_image = is_contain_in_image(anchors, round(size(im) * im_scales));
                    pred_boxes = pred_boxes(contained_in_image, :);
                    scores = scores(contained_in_image, :);
                end

                % drop too small boxes
                %1006 changed to get rid of too small boxes
                %[pred_boxes, scores] = filter_boxes(conf.test_min_box_size-3, pred_boxes, scores);
                [pred_boxes, scores] = filter_boxes(conf.test_min_box_size-3, pred_boxes, scores);
                
                % 0124: specially added for croppped image parts
                % filter out the (potentiall incomplete boxes on board)
                [pred_boxes, scores] = filter_border_boxes(y_start(kk),y_end(kk), x_start(kk),x_end(kk), pred_boxes, scores);

                % plus offset
                scores_tmp = cat(1, scores_tmp, scores);
                %0129: specially for 2x images
                pred_boxes_tmp = cat(1, pred_boxes_tmp, bsxfun(@plus, pred_boxes, [x_start(kk) y_start(kk) x_start(kk) y_start(kk)]*0.5-0.5));
            end
            % sort
            [scores_tmp, scores_ind] = sort(scores_tmp, 'descend');
            pred_boxes_tmp = pred_boxes_tmp(scores_ind, :);
            
            % 0124: assign them to output
            pred_boxes_all{i} = pred_boxes_tmp;
            scores_all{i} = scores_tmp;
        end
    end
end

function [data_blob, rois_blob, im_scale_factors] = get_blobs(conf, im, rois)
    [data_blob, im_scale_factors] = get_image_blob(conf, im);
    rois_blob = get_rois_blob(conf, rois, im_scale_factors);
end

function [blob, im_scales] = get_image_blob(conf, im)
    if length(conf.test_scale_range) == 1
        %0123 changed
        %[blob, im_scales] = prep_im_for_blob_keepsize(im, conf.image_means, conf.test_scales, conf.test_max_size);
        [blob, im_scales] = prep_im_for_blob_keepsize(im, conf.image_means, 1, conf.min_test_length, conf.max_test_length);
    else
        % arrayfun: function work on a vector
        [ims, im_scales] = arrayfun(@(x) prep_im_for_blob_keepsize(im, conf.image_means, x, conf.min_test_length, conf.max_test_length), conf.test_scale_range, 'UniformOutput', false);
        im_scales = cell2mat(im_scales);
        %0124: instead of a 4-d array, use a 1x3 cell to hold diff sizes im
        %blob = im_list_to_blob(ims); 
        blob = ims;
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

function [boxes, scores] = filter_border_boxes(y1,y2, x1,x2, boxes, scores)
    part_hei = y2 - y1 + 1;
    part_wid = x2 - x1 + 1;
    if y1 > 1 %y starts from middle line 
        valid_ind = boxes(:, 2) > 1; % not touch the top border
    else %y starts from begining 
        valid_ind = boxes(:, 4) < part_hei; % not touch the bottom border
    end
    if x1 > 1 %y starts from middle line 
        valid_ind = valid_ind & (boxes(:, 1) > 1); % not touch the left border
    else %x starts from begining 
        valid_ind = valid_ind & (boxes(:, 3) < part_wid); % not touch the right border
    end

    boxes = boxes(valid_ind, :);
    scores = scores(valid_ind, :);
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
    
