function [aboxes_conv4, aboxes_conv5, aboxes_conv6] = proposal_test_widerface_multibox_happy_flip(conf, imdb, varargin)
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
        abox_deltas = cell(num_images, 1);
        aanchors = cell(num_images, 1);
        ascores = cell(num_images, 1);
        
        count = 0;
        for i = 1:num_images
            count = count + 1;
            fprintf('%s: test (%s) %d/%d ', procid(), imdb.name, count, num_images);
            th = tic;
            im = imread(imdb.image_at(i));

            %[boxes, scores, abox_deltas{i}, aanchors{i}, ascores{i}] = proposal_im_detect_multibox(conf, caffe_net, im);
            [boxes_conv4, scores_conv4, boxes_conv5, scores_conv5, boxes_conv6, scores_conv6] = proposal_im_detect_multibox_happy_flip(conf, caffe_net, im);
            %[boxes, scores] = proposal_im_detect_multibox(conf, caffe_net, im);
            
            fprintf(' time: %.3fs\n', toc(th)); 
            %1230 added
            scores_conv4 = scores_conv4(scores_conv4 >= 0.55,:);  %0101:0.1-->0.55
            boxes_conv4 = boxes_conv4(scores_conv4 >= 0.55,:);
            scores_conv5 = scores_conv5(scores_conv5 >= 0.55,:);
            boxes_conv5 = boxes_conv5(scores_conv5 >= 0.55,:);
            scores_conv6 = scores_conv6(scores_conv6 >= 0.55,:);
            boxes_conv6 = boxes_conv6(scores_conv6 >= 0.55,:);

            aboxes_conv4{i} = [boxes_conv4, scores_conv4];
            aboxes_conv5{i} = [boxes_conv5, scores_conv5];
            aboxes_conv6{i} = [boxes_conv6, scores_conv6];
            
            if 0
                % debugging visualizations
                im = imread(imdb.image_at(i));
                keep = nms(aboxes{i}, 0.3);
                for k = 1:min(10, length(keep))
                    if aboxes{i}(keep(k),end) > 0.95
                        showboxes2(im, aboxes{i}(keep(k),1:4));
                        title(sprintf('%s %d score: %.3f\n', imdb.classes{1}, ...
                            k, aboxes{i}(keep(k),end)));
                        pause;
                    end
                end
            end
        end    
        save(fullfile(cache_dir, ['proposal_boxes_' imdb.name opts.suffix]), 'aboxes_conv4', 'aboxes_conv5','aboxes_conv6', '-v7.3');
        
        diary off;
        caffe.reset_all(); 
        rng(prev_rng);
    end
end
