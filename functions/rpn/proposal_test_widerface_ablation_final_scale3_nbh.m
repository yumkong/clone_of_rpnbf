function [aboxes_s4, aboxes_s8, aboxes_s16] = proposal_test_widerface_ablation_final_scale3_nbh(conf, imdb, varargin)
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
        aboxes_s4 = ld.aboxes_s4;
        aboxes_s8 = ld.aboxes_s8;
        aboxes_s16 = ld.aboxes_s16;
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
        aboxes_s4 = cell(num_images, 1);
        aboxes_s8 = cell(num_images, 1);
        aboxes_s16 = cell(num_images, 1);
        
        count = 0;
        for i = 1:num_images
            count = count + 1;
            fprintf('%s: test (%s) %d/%d ', procid(), imdb.name, count, num_images);
            th = tic;
            im = imread(imdb.image_at(i));

            %[boxes, scores, abox_deltas{i}, aanchors{i}, ascores{i}] = proposal_im_detect_multibox(conf, caffe_net, im);
            [boxes_s4, scores_s4, boxes_s8, scores_s8, boxes_s16, scores_s16] = proposal_im_detect_ablation_final_scale3_nbh(conf, caffe_net, im);
            %[boxes, scores] = proposal_im_detect_multibox(conf, caffe_net, im);
            
            fprintf(' time: %.3fs\n', toc(th)); 
            
            aboxes4_tmp = cell(1, numel(scores_s4));
            aboxes8_tmp = cell(1, numel(scores_s8));
            aboxes16_tmp = cell(1, numel(scores_s16));
            for j = 1:numel(scores_s4)
                %1230 added
                boxes_s4{j} = boxes_s4{j}(scores_s4{j} >= 0.6,:);
                scores_s4{j} = scores_s4{j}(scores_s4{j} >= 0.6,:);  %0101:0.1-->0.55

                boxes_s8{j} = boxes_s8{j}(scores_s8{j} >= 0.6,:);
                scores_s8{j} = scores_s8{j}(scores_s8{j} >= 0.6,:);
                
                boxes_s16{j} = boxes_s16{j}(scores_s16{j} >= 0.6,:);
                scores_s16{j} = scores_s16{j}(scores_s16{j} >= 0.6,:);

                aboxes4_tmp{j} = [boxes_s4{j}, scores_s4{j}];
                aboxes_s4{i} = cat(1, aboxes_s4{i}, aboxes4_tmp{j});
                aboxes8_tmp{j} = [boxes_s8{j}, scores_s8{j}];
                aboxes_s8{i} = cat(1, aboxes_s8{i}, aboxes8_tmp{j});
                aboxes16_tmp{j} = [boxes_s16{j}, scores_s16{j}];
                aboxes_s16{i} = cat(1, aboxes_s16{i}, aboxes16_tmp{j});
            end
            if 0
                % debugging visualizations
                im = imread(imdb.image_at(i));
                figure(1),clf;
                imshow(im);
                %color_cell23 = {'g', 'c', 'm'};
                %color_cell45 = {'r', 'b', 'y'};
                hold on
                for j = 1:numel(scores_s4)
                    keep = nms(aboxes4_tmp{j}, 0.3);
                    bbs_show = aboxes4_tmp{j}(keep, :);
                    bbs_show = bbs_show(bbs_show(:,5)>=0.99, :);
                    bbs_show(:, 3) = bbs_show(:, 3) - bbs_show(:, 1) + 1;
                    bbs_show(:, 4) = bbs_show(:, 4) - bbs_show(:, 2) + 1;
                    %bbApply('draw',bbs_show,color_cell23{j});
                    bbApply('draw',bbs_show,'g');
                end
                for j = 1:numel(scores_s8)
                    keep = nms(aboxes8_tmp{j}, 0.3);
                    bbs_show = aboxes8_tmp{j}(keep, :);
                    bbs_show = bbs_show(bbs_show(:,5)>=0.99, :);
                    bbs_show(:, 3) = bbs_show(:, 3) - bbs_show(:, 1) + 1;
                    bbs_show(:, 4) = bbs_show(:, 4) - bbs_show(:, 2) + 1;
                    %bbApply('draw',bbs_show,color_cell45{j});
                    bbApply('draw',bbs_show,'c');
                end
                for j = 1:numel(scores_s16)
                    keep = nms(aboxes16_tmp{j}, 0.3);
                    bbs_show = aboxes16_tmp{j}(keep, :);
                    bbs_show = bbs_show(bbs_show(:,5)>=0.99, :);
                    bbs_show(:, 3) = bbs_show(:, 3) - bbs_show(:, 1) + 1;
                    bbs_show(:, 4) = bbs_show(:, 4) - bbs_show(:, 2) + 1;
                    %bbApply('draw',bbs_show,color_cell45{j});
                    bbApply('draw',bbs_show,'m');
                end
                hold off
            end
        end    
        save(fullfile(cache_dir, ['proposal_boxes_' imdb.name opts.suffix]), 'aboxes_s4', 'aboxes_s8', 'aboxes_s16', '-v7.3');
        
        diary off;
        caffe.reset_all(); 
        rng(prev_rng);
    end
end
