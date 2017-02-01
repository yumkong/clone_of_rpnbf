function [aboxes_conv4, aboxes_conv5, aboxes_conv6] = proposal_test_widerface_multibox_happy_flip_3scale(conf, imdb, varargin)
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
        
        count = 0;
        for i = 1:num_images
            count = count + 1;
            fprintf('%s: test (%s) %d/%d ', procid(), imdb.name, count, num_images);
            th = tic;
            im = imread(imdb.image_at(i));

            %[boxes, scores, abox_deltas{i}, aanchors{i}, ascores{i}] = proposal_im_detect_multibox(conf, caffe_net, im);
            [boxes_conv4, scores_conv4, boxes_conv5, scores_conv5, boxes_conv6, scores_conv6] = proposal_im_detect_multibox_happy_flip_3scale(conf, caffe_net, im);
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
                aboxes_conv4{i} = cat(1, aboxes_conv4{i}, aboxes4_tmp{j});
                aboxes5_tmp{j} = [boxes_conv5{j}, scores_conv5{j}];
                aboxes_conv5{i} = cat(1, aboxes_conv5{i}, aboxes5_tmp{j});
                aboxes6_tmp{j} = [boxes_conv6{j}, scores_conv6{j}];
                aboxes_conv6{i} = cat(1, aboxes_conv6{i}, aboxes6_tmp{j});
            end
            if 0
                % debugging visualizations
                im = imread(imdb.image_at(i));
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
