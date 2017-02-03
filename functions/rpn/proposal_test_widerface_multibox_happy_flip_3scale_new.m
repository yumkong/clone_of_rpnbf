function [aboxes_conv4, aboxes_conv5, aboxes_conv6] = proposal_test_widerface_multibox_happy_flip_3scale_new(conf, imdb, varargin)
% aboxes = proposal_test_caltech(conf, imdb, varargin)
% --------------------------------------------------------
% RPN_BF
% Copyright (c) 2016, Liliang Zhang
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------   

%% inputs
    ip = inputParser;
    ip.addRequired('conf',                              @isstruct);
    ip.addRequired('imdb',                              @isstruct);
    ip.addParamValue('net_def_file',    fullfile(pwd, 'proposal_models', 'Zeiler_conv5', 'test.prototxt'), ...
                                                        @isstr);
    ip.addParamValue('net_file',        fullfile(pwd, 'proposal_models', 'Zeiler_conv5', 'Zeiler_conv5.caffemodel'), ...
                                                        @isstr);
    ip.addParamValue('cache_name',      'Zeiler_conv5', ...
                                                        @isstr);
                                                    
    ip.addParamValue('suffix',          '',             @isstr);
    ip.addParamValue('start_num',        1,             @isnumeric);
    
    ip.parse(conf, imdb, varargin{:});
    opts = ip.Results;
    
    %##### note@1013:every time you want to regenerate bboxes, need to
    %delete the proposal_boxes_*.mat file in
    %output/VGG16_wider*/rpn_cachedir/rpn_widerface_VGG16_stage1_rpn
    cache_dir = fullfile(pwd, 'output', conf.exp_name, 'rpn_cachedir', opts.cache_name, imdb.name);
	%cache_dir = fullfile(pwd, 'output', opts.cache_name, imdb.name);
    try
        % try to load cache
        ld = load(fullfile(cache_dir, ['proposal_boxes_' imdb.name opts.suffix]));
        aboxes_conv4 = ld.aboxes_conv4;
        aboxes_conv5 = ld.aboxes_conv5;
        aboxes_conv6 = ld.aboxes_conv6;
        clear ld;
    catch    
        %% init net
        % init caffe net
        mkdir_if_missing(cache_dir);
        caffe_log_file_base = fullfile(cache_dir, 'caffe_log');
        caffe.init_log(caffe_log_file_base);
        caffe_net = caffe.Net(opts.net_def_file, 'test');
        caffe_net.copy_from(opts.net_file);

        % init log
        timestamp = datestr(datevec(now()), 'yyyymmdd_HHMMSS');
        mkdir_if_missing(fullfile(cache_dir, 'log'));
        log_file = fullfile(cache_dir, 'log', ['test_', timestamp, '.txt']);
        diary(log_file);

        % set random seed
        prev_rng = seed_rand(conf.rng_seed);
        caffe.set_random_seed(conf.rng_seed);

        % set gpu/cpu
        if conf.use_gpu
            caffe.set_mode_gpu();
        else
            caffe.set_mode_cpu();
        end             

        disp('opts:');
        disp(opts);
        disp('conf:');
        disp(conf);
    
%% testing
        num_images = length(imdb.image_ids);
        % all detections are collected into:
        %    all_boxes[image] = N x 5 array of detections in
        %    (x1, y1, x2, y2, score)
        aboxes_conv4 = cell(num_images, 1);
        aboxes_conv5 = cell(num_images, 1);
        aboxes_conv6 = cell(num_images, 1);
        
        %count = 0;
        for ii = opts.start_num:num_images %15
            %count = count + 1;
            %fprintf('%s: test (%s) %d/%d ', procid(), imdb.name, count, num_images);
            fprintf('%s: test (%s) %d/%d ', procid(), imdb.name, ii, num_images);
            th = tic;
            
            im = imread(imdb.image_at(ii));
            
            [im_blob, im_scales] = get_image_blob(conf, im);
            %im_size = size(im);
            %scaled_im_size = round(im_size * im_scales);
            %scaled_im_size = round(im_scales' * im_size);
            im_blob_all = im_blob;
            %scaled_im_size_all = scaled_im_size;

            %0131 added
            boxes_conv4 = cell(1, numel(im_blob_all));
            scores_conv4 = cell(1, numel(im_blob_all));
            boxes_conv5 = cell(1, numel(im_blob_all));
            scores_conv5 = cell(1, numel(im_blob_all));
            boxes_conv6 = cell(1, numel(im_blob_all));
            scores_conv6 = cell(1, numel(im_blob_all));

            for i = 1:numel(im_blob_all)
                %0124 send an individual image and scaled size here
                im_blob = im_blob_all{i};
                %scaled_im_size = scaled_im_size_all(i,:);

                % no matter 2x, 1x or even 0.5x, always put them to partions:
                % if <= 1504 x 1504, 1 partition, else multiple partitions
                hei_im = size(im_blob, 1);
                wid_im = size(im_blob, 2);

                h_part_num = ceil(hei_im / 1504);%1504 is out memory, so 2048
                w_part_num = ceil(wid_im / 1504);

                hei_middle = ceil(hei_im/h_part_num/8)*8;
                wid_middle = ceil(wid_im/w_part_num/8)*8;
                % [top-left bottom-left top-right bottom-right]
                % start position, also the offset position of bboxes
                y_start = repmat([1 hei_middle*(1:h_part_num-1)-7]',1,w_part_num);%-8
                y_start = y_start(:);
                x_start = repmat([1 wid_middle*(1:w_part_num-1)-7],h_part_num, 1);%-8
                x_start = x_start(:);
                % end position
                y_end = repmat([hei_middle*(1:h_part_num-1)+8 hei_im]',1,w_part_num);
                y_end = y_end(:);
                x_end = repmat([wid_middle*(1:w_part_num-1)+8 wid_im],h_part_num, 1);
                x_end = x_end(:);

                im_blob_complete = im_blob;
                scores_tmp4 = [];
                pred_boxes_tmp4 = [];
                scores_tmp5 = [];
                pred_boxes_tmp5 = [];
                scores_tmp6 = [];
                pred_boxes_tmp6 = [];

                for kk = 1:numel(y_start)
                    im_blob = im_blob_complete(y_start(kk):y_end(kk), x_start(kk):x_end(kk), :);
                    %[boxes, scores, abox_deltas{i}, aanchors{i}, ascores{i}] = proposal_im_detect_multibox(conf, caffe_net, im);
                    [box4, score4, box5, score5, box6, score6] = proposal_im_detect_multibox_happy_flip_3scale_new(conf, caffe_net, im_blob, im_scales(i), x_start(kk), y_start(kk));
                    %0201 added
                    %caffe.reset_all(); 
                    %caffe_net = caffe.Net(opts.net_def_file, 'test');
                    %caffe_net.copy_from(opts.net_file);
                    pred_boxes_tmp4 = cat(1, pred_boxes_tmp4, box4);
                    scores_tmp4 = cat(1, scores_tmp4, score4);
                    pred_boxes_tmp5 = cat(1, pred_boxes_tmp5, box5);
                    scores_tmp5 = cat(1, scores_tmp5, score5);
                    pred_boxes_tmp6 = cat(1, pred_boxes_tmp6, box6);
                    scores_tmp6 = cat(1, scores_tmp6, score6);
                end
                
                % sort
                [scores_tmp4, scores_ind] = sort(scores_tmp4, 'descend');
                pred_boxes_tmp4 = pred_boxes_tmp4(scores_ind, :);
                % 0124: assign them to output
                boxes_conv4{i} = pred_boxes_tmp4;
                scores_conv4{i} = scores_tmp4;

                % sort
                [scores_tmp5, scores_ind] = sort(scores_tmp5, 'descend');
                pred_boxes_tmp5 = pred_boxes_tmp5(scores_ind, :);
                % 0124: assign them to output
                boxes_conv5{i} = pred_boxes_tmp5;
                scores_conv5{i} = scores_tmp5;

                % sort
                [scores_tmp6, scores_ind] = sort(scores_tmp6, 'descend');
                pred_boxes_tmp6 = pred_boxes_tmp6(scores_ind, :);
                % 0124: assign them to output
                boxes_conv6{i} = pred_boxes_tmp6;
                scores_conv6{i} = scores_tmp6;
            end
            %[boxes, scores] = proposal_im_detect_multibox(conf, caffe_net, im);
            
            fprintf(' time: %.3fs\n', toc(th));  
            aboxes4_tmp = cell(1, numel(scores_conv4));
            aboxes5_tmp = cell(1, numel(scores_conv5));
            aboxes6_tmp = cell(1, numel(scores_conv6));
            for j = 1:numel(scores_conv4)
                %1230 added
                boxes_conv4{j} = boxes_conv4{j}(scores_conv4{j} >= 0.6,:);
                scores_conv4{j} = scores_conv4{j}(scores_conv4{j} >= 0.6,:);  %0131:0.55-->0.6

                boxes_conv5{j} = boxes_conv5{j}(scores_conv5{j} >= 0.6,:);
                scores_conv5{j} = scores_conv5{j}(scores_conv5{j} >= 0.6,:);  %0131:0.55-->0.6
                
                boxes_conv6{j} = boxes_conv6{j}(scores_conv6{j} >= 0.6,:);
                scores_conv6{j} = scores_conv6{j}(scores_conv6{j} >= 0.6,:);  %0131:0.55-->0.6

                aboxes4_tmp{j} = [boxes_conv4{j}, scores_conv4{j}];
                aboxes_conv4{ii} = cat(1, aboxes_conv4{ii}, aboxes4_tmp{j});
                aboxes5_tmp{j} = [boxes_conv5{j}, scores_conv5{j}];
                aboxes_conv5{ii} = cat(1, aboxes_conv5{ii}, aboxes5_tmp{j});
                aboxes6_tmp{j} = [boxes_conv6{j}, scores_conv6{j}];
                aboxes_conv6{ii} = cat(1, aboxes_conv6{ii}, aboxes6_tmp{j});
            end
            %save every results individually
            %aboxes4_save = aboxes_conv4{ii};
            %aboxes5_save = aboxes_conv5{ii};
            %aboxes6_save = aboxes_conv6{ii};
            %save(fullfile(cache_dir, sprintf('proposal_boxes_%04d.mat',ii)), 'aboxes4_save', 'aboxes5_save','aboxes6_save');
            if 0
                % debugging visualizations
                im = imread(imdb.image_at(ii));
                figure(1),clf;
                imshow(im);
                color_cell4 = {'g', 'c', 'm'};
                color_cell5 = {'r', 'b', 'y'};
                color_cell6 = {'k', 'w', 'm'};
                hold on
                for j = 1:numel(scores_conv4)
                    if ~isempty(aboxes4_tmp{j})
                        keep = nms(aboxes4_tmp{j}, 0.3);
                        bbs_show = aboxes4_tmp{j}(keep, :);
                        bbs_show = bbs_show(bbs_show(:,5)>=0.9, :);
                        bbs_show(:, 3) = bbs_show(:, 3) - bbs_show(:, 1) + 1;
                        bbs_show(:, 4) = bbs_show(:, 4) - bbs_show(:, 2) + 1;
                        bbApply('draw',bbs_show,color_cell4{j});
                    end
                end
                for j = 1:numel(scores_conv5)
                    if ~isempty(aboxes5_tmp{j})
                        keep = nms(aboxes5_tmp{j}, 0.3);
                        bbs_show = aboxes5_tmp{j}(keep, :);
                        bbs_show = bbs_show(bbs_show(:,5)>=0.9, :);
                        bbs_show(:, 3) = bbs_show(:, 3) - bbs_show(:, 1) + 1;
                        bbs_show(:, 4) = bbs_show(:, 4) - bbs_show(:, 2) + 1;
                        bbApply('draw',bbs_show,color_cell5{j});
                    end
                end
                for j = 1:numel(scores_conv6)
                    if ~isempty(aboxes6_tmp{j})
                        keep = nms(aboxes6_tmp{j}, 0.3);
                        bbs_show = aboxes6_tmp{j}(keep, :);
                        bbs_show = bbs_show(bbs_show(:,5)>=0.9, :);
                        bbs_show(:, 3) = bbs_show(:, 3) - bbs_show(:, 1) + 1;
                        bbs_show(:, 4) = bbs_show(:, 4) - bbs_show(:, 2) + 1;
                        bbApply('draw',bbs_show,color_cell6{j});
                    end
                end
                hold off
            end
        end    
        save(fullfile(cache_dir, ['proposal_boxes_' imdb.name opts.suffix]), 'aboxes_conv4', 'aboxes_conv5','aboxes_conv6', '-v7.3');
        
        diary off;
        caffe.reset_all(); 
        rng(prev_rng);
    end
end

function [blob, im_scales] = get_image_blob(conf, im)
    if length(conf.test_scale_range) == 1
	    %1209 changed
        [blob, im_scales] = prep_im_for_blob_keepsize_conv34(im, conf.image_means,1, conf.min_test_length, conf.max_test_length);
        %[blob, im_scales] = prep_im_for_blob(im, conf.image_means, conf.test_scales, conf.test_max_size);
    else
        [ims, im_scales] = arrayfun(@(x) prep_im_for_blob_keepsize_conv34(im, conf.image_means, x,conf.min_test_length, conf.max_test_length), conf.test_scale_range, 'UniformOutput', false);
        im_scales = cell2mat(im_scales);
        %0124: instead of a 4-d array, use a 1x3 cell to hold diff sizes im
        %blob = im_list_to_blob(ims); 
        blob = ims;
    end
end
