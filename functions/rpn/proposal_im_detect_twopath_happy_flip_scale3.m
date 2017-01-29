function [pred_boxes_res23_all, scores_res23_all, pred_boxes_res45_all, scores_res45_all] = proposal_im_detect_twopath_happy_flip_scale3(conf, caffe_net, im, im_idx)
%function [pred_boxes, scores] = proposal_im_detect_multibox_happy(conf, caffe_net, im, im_idx)
% [pred_boxes, scores, box_deltas_, anchors_, scores_] = proposal_im_detect(conf, im, net_idx)
% --------------------------------------------------------
% Faster R-CNN
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------    

    im = single(im);
	%1209: get_image_blob has been changed to have im_blob size of 8N
    [im_blob, im_scales] = get_image_blob(conf, im);
    im_size = size(im);
    %scaled_im_size = round(im_size * im_scales);
    scaled_im_size = round(im_scales' * im_size);
    im_blob_all = im_blob;
    scaled_im_size_all = scaled_im_size;
    %0124 added
    pred_boxes_res23_all = cell(1, numel(im_blob_all));
    scores_res23_all = cell(1, numel(im_blob_all));
    pred_boxes_res45_all = cell(1, numel(im_blob_all));
    scores_res45_all = cell(1, numel(im_blob_all));
    
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
            box_deltas_res23 = output_blobs{1};
            box_deltas_res45 = output_blobs{2};  %put it here
            %0129 added
            feat23_size = [size(box_deltas_res23, 2), size(box_deltas_res23, 1)];
            feat45_size = [size(box_deltas_res45, 2), size(box_deltas_res45, 1)];
            %featuremap_size = [size(box_deltas_conv4, 2), size(box_deltas_conv4, 1)];
            % permute from [width, height, channel] to [channel, height, width], where channel is the
                % fastest dimension
            box_deltas_res23 = permute(box_deltas_res23, [3, 2, 1]);
            box_deltas_res23 = reshape(box_deltas_res23, 4, [])';

            %1209 changed
            %[anchors_conv4, anchors_conv5, anchors_conv6] = proposal_locate_anchors_multibox_final3(conf, size(im), conf.test_scales);
            [anchors_res23, anchors_res45] = proposal_locate_anchors_twopath_happy(conf, size(im_blob), conf.test_scales, feat23_size, feat45_size);
            pred_boxes_res23 = fast_rcnn_bbox_transform_inv(anchors_res23, box_deltas_res23);
              % scale back
            pred_boxes_res23 = bsxfun(@times, pred_boxes_res23 - 1, ...
                ([im_size(2), im_size(1), im_size(2), im_size(1)] - 1) ./ ([scaled_im_size(2), scaled_im_size(1), scaled_im_size(2), scaled_im_size(1)] - 1)) + 1;
            pred_boxes_res23 = clip_boxes(pred_boxes_res23, size(im, 2), size(im, 1));

            assert(conf.test_binary == false);
            % use softmax estimated probabilities
            scores_res23 = output_blobs{3}(:, :, end);
            scores_res23 = reshape(scores_res23, size(output_blobs{1}, 1), size(output_blobs{1}, 2), []);
            % permute from [width, height, channel] to [channel, height, width], where channel is the
                % fastest dimension
            scores_res23 = permute(scores_res23, [3, 2, 1]);

            % 1204: spatially decimate anchors by one half (only keep highest scoring boxes out of eight spatial neighbors)
            % in this way the output boxes should be similar with that of conv4_3
            % in vertical direction
            score_mask = false(size(scores_res23));
            score_mask(:,1,:) = scores_res23(:,1,:) >= scores_res23(:,2,:);
            for idx = 2:size(scores_res23, 2)-1
                score_mask(:,idx,:) = (scores_res23(:,idx,:) >= scores_res23(:,idx-1,:)) & (scores_res23(:,idx,:) >= scores_res23(:,idx+1,:));
            end
            score_mask(:,end,:) = scores_res23(:,end,:) >= scores_res23(:,end-1,:);
            % in horizontal direction
            score_mask(:,:,1) = scores_res23(:,:,1) >= scores_res23(:,:,2);
            for idx = 2:size(scores_res23, 3)-1
                score_mask(:,:,idx) = (scores_res23(:,:,idx) >= scores_res23(:,:,idx-1)) & (scores_res23(:,:,idx) >= scores_res23(:,:, idx+1));
            end
            score_mask(:,:, end) = scores_res23(:,:, end) >= scores_res23(:,:, end-1);
            score_mask = score_mask(:);
            % end of 1204

            scores_res23 = scores_res23(:);

             % 1025: decimate anchors by one half (only keep one boxes out of each anchor scale position)
            anchor_num = size(conf.anchors_res23, 1);  %14
            half_anchor_num = anchor_num/2; %7
            tmp_scores = reshape(scores_res23, anchor_num, []); 
            hw1_score = tmp_scores(1:half_anchor_num, :);
            hw2_score = tmp_scores(1+half_anchor_num:end, :);
            hw1_greater_mask = (hw1_score >= hw2_score);
            greater_mask = cat(1, hw1_greater_mask, ~hw1_greater_mask);
            %1212 added: combine two masks
            final_mask = greater_mask(:) & score_mask;
            %1205 added: keep only those local maximum scores
            scores_res23 = scores_res23(final_mask,:);
            pred_boxes_res23 = pred_boxes_res23(final_mask,:);  % new pred_boxes

            if conf.test_drop_boxes_runoff_image
                contained_in_image = is_contain_in_image(anchors_res23, round(size(im) * im_scales));
                pred_boxes_res23 = pred_boxes_res23(contained_in_image, :);
                scores_res23 = scores_res23(contained_in_image, :);
            end

            % drop too small boxes
            %0101: conf.test_min_box_size-2 ==> conf.test_min_box_size-3
            [pred_boxes_res23, scores_res23] = filter_boxes(conf.test_min_box_size-2, pred_boxes_res23, scores_res23);

            % sort
            [scores_res23, scores_ind] = sort(scores_res23, 'descend');
            pred_boxes_res23 = pred_boxes_res23(scores_ind, :);
            % 0124: assign them to output
            pred_boxes_res23_all{i} = pred_boxes_res23;
            scores_res23_all{i} = scores_res23;
            % ===================================================== for conv5
            % Apply bounding-box regression deltas
            %box_deltas_res45 = output_blobs{2};  %put it above
            %featuremap_size = [size(box_deltas_conv5, 2), size(box_deltas_conv5, 1)];
            % permute from [width, height, channel] to [channel, height, width], where channel is the
                % fastest dimension
            box_deltas_res45 = permute(box_deltas_res45, [3, 2, 1]);
            box_deltas_res45 = reshape(box_deltas_res45, 4, [])';

            pred_boxes_res45 = fast_rcnn_bbox_transform_inv(anchors_res45, box_deltas_res45);
              % scale back
            pred_boxes_res45 = bsxfun(@times, pred_boxes_res45 - 1, ...
                ([im_size(2), im_size(1), im_size(2), im_size(1)] - 1) ./ ([scaled_im_size(2), scaled_im_size(1), scaled_im_size(2), scaled_im_size(1)] - 1)) + 1;
            pred_boxes_res45 = clip_boxes(pred_boxes_res45, size(im, 2), size(im, 1));

            assert(conf.test_binary == false);
            % use softmax estimated probabilities
            scores_res45 = output_blobs{4}(:, :, end);
            scores_res45 = reshape(scores_res45, size(output_blobs{2}, 1), size(output_blobs{2}, 2), []);
            % permute from [width, height, channel] to [channel, height, width], where channel is the
                % fastest dimension
            scores_res45 = permute(scores_res45, [3, 2, 1]);

            scores_res45 = scores_res45(:);

            % 1025: decimate anchors by one half (only keep one boxes out of each anchor scale position)
            anchor_num = size(conf.anchors_res45, 1);  %14
            half_anchor_num = anchor_num/2; %7
            tmp_scores = reshape(scores_res45, anchor_num, []); 
            hw1_score = tmp_scores(1:half_anchor_num, :);
            hw2_score = tmp_scores(1+half_anchor_num:end, :);
            hw1_greater_mask = (hw1_score >= hw2_score);
            greater_mask = cat(1, hw1_greater_mask, ~hw1_greater_mask);
            scores_res45 = scores_res45(greater_mask(:),:);  %new scores
            pred_boxes_res45 = pred_boxes_res45(greater_mask(:),:);  % new pred_boxes
            %====== end of 1025

            if conf.test_drop_boxes_runoff_image
                contained_in_image = is_contain_in_image(anchors_res45, round(size(im) * im_scales));
                pred_boxes_res45 = pred_boxes_res45(contained_in_image, :);
                scores_res45 = scores_res45(contained_in_image, :);
            end

            % drop too small boxes *** liu@1114: here tempararily set conv5's
            % thresh as 20, may change later (conv5 minimal anchor size is 32)
            [pred_boxes_res45, scores_res45] = filter_boxes(32, pred_boxes_res45, scores_res45);

            % sort
            [scores_res45, scores_ind] = sort(scores_res45, 'descend');
            pred_boxes_res45 = pred_boxes_res45(scores_ind, :);
            % 0124: assign them to output
            pred_boxes_res45_all{i} = pred_boxes_res45;
            scores_res45_all{i} = scores_res45;
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
            scores_tmp23 = [];
            pred_boxes_tmp23 = [];
            scores_tmp45 = [];
            pred_boxes_tmp45 = [];
            for kk = 1:4
                im_blob = im_blob_complete(y_start(kk):y_end(kk), x_start(kk):x_end(kk), :);
                im_blob = im_blob(:, :, [3, 2, 1], :); % from rgb to brg
                im_blob = permute(im_blob, [2, 1, 3, 4]);
                im_blob = single(im_blob);

                net_inputs = {im_blob};

                % Reshape net's input blobs
                caffe_net.reshape_as_input(net_inputs);
                output_blobs = caffe_net.forward(net_inputs);

                % Apply bounding-box regression deltas
                box_deltas_res23 = output_blobs{1};
                box_deltas_res45 = output_blobs{2}; %0129: put it here
                %0129 added
                feat23_size = [size(box_deltas_res23, 2), size(box_deltas_res23, 1)];
                feat45_size = [size(box_deltas_res45, 2), size(box_deltas_res45, 1)];
                %featuremap_size = [size(box_deltas_conv4, 2), size(box_deltas_conv4, 1)];
                % permute from [width, height, channel] to [channel, height, width], where channel is the
                    % fastest dimension
                box_deltas_res23 = permute(box_deltas_res23, [3, 2, 1]);
                box_deltas_res23 = reshape(box_deltas_res23, 4, [])';

                %1209 changed
                %[anchors_conv4, anchors_conv5, anchors_conv6] = proposal_locate_anchors_multibox_final3(conf, size(im), conf.test_scales);
                %[anchors_res23, anchors_res45] = proposal_locate_anchors_twopath_happy(conf, size(im), conf.test_scales);
                [anchors_res23, anchors_res45] = proposal_locate_anchors_twopath_happy(conf, size(im_blob), conf.test_scales, feat23_size, feat45_size);
                pred_boxes_res23 = fast_rcnn_bbox_transform_inv(anchors_res23, box_deltas_res23);
                  % scale back
                pred_boxes_res23 = bsxfun(@times, pred_boxes_res23 - 1, ...
                    ([im_size(2), im_size(1), im_size(2), im_size(1)] - 1) ./ ([scaled_im_size(2), scaled_im_size(1), scaled_im_size(2), scaled_im_size(1)] - 1)) + 1;
                pred_boxes_res23 = clip_boxes(pred_boxes_res23, size(im, 2), size(im, 1));

                assert(conf.test_binary == false);
                % use softmax estimated probabilities
                scores_res23 = output_blobs{3}(:, :, end);
                scores_res23 = reshape(scores_res23, size(output_blobs{1}, 1), size(output_blobs{1}, 2), []);
                % permute from [width, height, channel] to [channel, height, width], where channel is the
                    % fastest dimension
                scores_res23 = permute(scores_res23, [3, 2, 1]);

                % 1204: spatially decimate anchors by one half (only keep highest scoring boxes out of eight spatial neighbors)
                % in this way the output boxes should be similar with that of conv4_3
                % in vertical direction
                score_mask = false(size(scores_res23));
                score_mask(:,1,:) = scores_res23(:,1,:) >= scores_res23(:,2,:);
                for idx = 2:size(scores_res23, 2)-1
                    score_mask(:,idx,:) = (scores_res23(:,idx,:) >= scores_res23(:,idx-1,:)) & (scores_res23(:,idx,:) >= scores_res23(:,idx+1,:));
                end
                score_mask(:,end,:) = scores_res23(:,end,:) >= scores_res23(:,end-1,:);
                % in horizontal direction
                score_mask(:,:,1) = scores_res23(:,:,1) >= scores_res23(:,:,2);
                for idx = 2:size(scores_res23, 3)-1
                    score_mask(:,:,idx) = (scores_res23(:,:,idx) >= scores_res23(:,:,idx-1)) & (scores_res23(:,:,idx) >= scores_res23(:,:, idx+1));
                end
                score_mask(:,:, end) = scores_res23(:,:, end) >= scores_res23(:,:, end-1);
                score_mask = score_mask(:);
                % end of 1204

                scores_res23 = scores_res23(:);

                 % 1025: decimate anchors by one half (only keep one boxes out of each anchor scale position)
                anchor_num = size(conf.anchors_res23, 1);  %14
                half_anchor_num = anchor_num/2; %7
                tmp_scores = reshape(scores_res23, anchor_num, []); 
                hw1_score = tmp_scores(1:half_anchor_num, :);
                hw2_score = tmp_scores(1+half_anchor_num:end, :);
                hw1_greater_mask = (hw1_score >= hw2_score);
                greater_mask = cat(1, hw1_greater_mask, ~hw1_greater_mask);
                %1212 added: combine two masks
                final_mask = greater_mask(:) & score_mask;
                %1205 added: keep only those local maximum scores
                scores_res23 = scores_res23(final_mask,:);
                pred_boxes_res23 = pred_boxes_res23(final_mask,:);  % new pred_boxes

                if conf.test_drop_boxes_runoff_image
                    contained_in_image = is_contain_in_image(anchors_res23, round(size(im) * im_scales));
                    pred_boxes_res23 = pred_boxes_res23(contained_in_image, :);
                    scores_res23 = scores_res23(contained_in_image, :);
                end

                % drop too small boxes
                %0101: conf.test_min_box_size-2 ==> conf.test_min_box_size-3
                [pred_boxes_res23, scores_res23] = filter_boxes(conf.test_min_box_size-2, pred_boxes_res23, scores_res23);
                
                % 0124: specially added for croppped image parts
                % filter out the (potentiall incomplete boxes on board)
                [pred_boxes_res23, scores_res23] = filter_border_boxes(y_start(kk),y_end(kk), x_start(kk),x_end(kk), pred_boxes_res23, scores_res23);

                
                assert(i==2 || i == 3); %assume only 1x or 2x image can be larger than 1500 x 1500
                % plus offset
                scores_tmp23 = cat(1, scores_tmp23, scores_res23);
                %0129: specially for 2x images
                if i == 3  %2x
                    pred_boxes_tmp23 = cat(1, pred_boxes_tmp23, bsxfun(@plus, pred_boxes_res23, [x_start(kk) y_start(kk) x_start(kk) y_start(kk)]*0.5-0.5));
                else %1x
                    pred_boxes_tmp23 = cat(1, pred_boxes_tmp23, bsxfun(@plus, pred_boxes_res23, [x_start(kk) y_start(kk) x_start(kk) y_start(kk)]-1)); 
                end
                % ===================================================== for conv5
                % Apply bounding-box regression deltas
                %box_deltas_res45 = output_blobs{2};
                %featuremap_size = [size(box_deltas_conv5, 2), size(box_deltas_conv5, 1)];
                % permute from [width, height, channel] to [channel, height, width], where channel is the
                    % fastest dimension
                box_deltas_res45 = permute(box_deltas_res45, [3, 2, 1]);
                box_deltas_res45 = reshape(box_deltas_res45, 4, [])';

                pred_boxes_res45 = fast_rcnn_bbox_transform_inv(anchors_res45, box_deltas_res45);
                  % scale back
                pred_boxes_res45 = bsxfun(@times, pred_boxes_res45 - 1, ...
                    ([im_size(2), im_size(1), im_size(2), im_size(1)] - 1) ./ ([scaled_im_size(2), scaled_im_size(1), scaled_im_size(2), scaled_im_size(1)] - 1)) + 1;
                pred_boxes_res45 = clip_boxes(pred_boxes_res45, size(im, 2), size(im, 1));

                assert(conf.test_binary == false);
                % use softmax estimated probabilities
                scores_res45 = output_blobs{4}(:, :, end);
                scores_res45 = reshape(scores_res45, size(output_blobs{2}, 1), size(output_blobs{2}, 2), []);
                % permute from [width, height, channel] to [channel, height, width], where channel is the
                    % fastest dimension
                scores_res45 = permute(scores_res45, [3, 2, 1]);

                scores_res45 = scores_res45(:);

                % 1025: decimate anchors by one half (only keep one boxes out of each anchor scale position)
                anchor_num = size(conf.anchors_res45, 1);  %14
                half_anchor_num = anchor_num/2; %7
                tmp_scores = reshape(scores_res45, anchor_num, []); 
                hw1_score = tmp_scores(1:half_anchor_num, :);
                hw2_score = tmp_scores(1+half_anchor_num:end, :);
                hw1_greater_mask = (hw1_score >= hw2_score);
                greater_mask = cat(1, hw1_greater_mask, ~hw1_greater_mask);
                scores_res45 = scores_res45(greater_mask(:),:);  %new scores
                pred_boxes_res45 = pred_boxes_res45(greater_mask(:),:);  % new pred_boxes
                %====== end of 1025

                if conf.test_drop_boxes_runoff_image
                    contained_in_image = is_contain_in_image(anchors_res45, round(size(im) * im_scales));
                    pred_boxes_res45 = pred_boxes_res45(contained_in_image, :);
                    scores_res45 = scores_res45(contained_in_image, :);
                end

                % drop too small boxes *** liu@1114: here tempararily set conv5's
                % thresh as 20, may change later (conv5 minimal anchor size is 32)
                [pred_boxes_res45, scores_res45] = filter_boxes(32, pred_boxes_res45, scores_res45);
                
                % 0124: specially added for croppped image parts
                % filter out the (potentiall incomplete boxes on board)
                [pred_boxes_res45, scores_res45] = filter_border_boxes(y_start(kk),y_end(kk), x_start(kk),x_end(kk), pred_boxes_res45, scores_res45);

                
                assert(i==2 || i == 3); %assume only 1x or 2x image can be larger than 1500 x 1500
                % plus offset
                scores_tmp45 = cat(1, scores_tmp45, scores_res45);
                %0129: specially for 2x images
                if i == 3  %2x
                    pred_boxes_tmp45 = cat(1, pred_boxes_tmp45, bsxfun(@plus, pred_boxes_res45, [x_start(kk) y_start(kk) x_start(kk) y_start(kk)]*0.5-0.5));
                else %1x
                    pred_boxes_tmp45 = cat(1, pred_boxes_tmp45, bsxfun(@plus, pred_boxes_res45, [x_start(kk) y_start(kk) x_start(kk) y_start(kk)]-1)); 
                end
            end
            % sort
            [scores_tmp23, scores_ind] = sort(scores_tmp23, 'descend');
            pred_boxes_tmp23 = pred_boxes_tmp23(scores_ind, :);
            
            % 0124: assign them to output
            pred_boxes_res23_all{i} = pred_boxes_tmp23;
            scores_res23_all{i} = scores_tmp23;
            
            % sort
            [scores_tmp45, scores_ind] = sort(scores_tmp45, 'descend');
            pred_boxes_tmp45 = pred_boxes_tmp45(scores_ind, :);
            
            % 0124: assign them to output
            pred_boxes_res45_all{i} = pred_boxes_tmp45;
            scores_res45_all{i} = scores_tmp45;
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
    
