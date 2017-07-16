function aboxes = rpn_test(conf, imdb, varargin)
% aboxes = proposal_test_caltech(conf, imdb, varargin)
% --------------------------------------------------------
% RPN_BF
% Copyright (c) 2016, Liliang Zhang
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------   

%% inputs: (liu) must in the order of the input arguments
    ip = inputParser;
    ip.addRequired('conf',                              @isstruct);
    ip.addRequired('imdb',                              @isstruct);
    % 0713 added keep this dir the same with the father function
    ip.addRequired('cache_dir',                       @isstr);
    ip.addParamValue('net_def_file',    fullfile(pwd, 'proposal_models', 'Zeiler_conv5', 'test.prototxt'), ...
                                                        @isstr);
    ip.addParamValue('net_file',        fullfile(pwd, 'proposal_models', 'Zeiler_conv5', 'Zeiler_conv5.caffemodel'), ...
                                                        @isstr);
    ip.addParamValue('cache_name',      'Zeiler_conv5', ...
                                                        @isstr);
                                                    
    ip.addParamValue('suffix',          '',             @isstr);
    %0124 added for three scale input (x0.5 x1 x2)
    ip.addParamValue('three_scales',    false,          @islogical);
    
    ip.parse(conf, imdb, varargin{:});
    opts = ip.Results;
    
    %##### note@1013:every time you want to regenerate bboxes, need to
    %delete the proposal_boxes_*.mat file in
    %output/VGG16_wider*/rpn_cachedir/rpn_widerface_VGG16_stage1_rpn
    cache_dir = opts.cache_dir; %fullfile(pwd, 'output', conf.exp_name, 'rpn_cachedir', opts.cache_name, imdb.name);
	%cache_dir = fullfile(pwd, 'output', opts.cache_name, imdb.name);
    try
        % try to load cache
        ld = load(fullfile(cache_dir, ['proposal_boxes_' imdb.name opts.suffix]));
        aboxes = ld.aboxes;
        clear ld;
    catch    
        %% init net
        % init caffe net
        helper.mkdir_if_missing(cache_dir);
        caffe_log_file_base = fullfile(cache_dir, 'caffe_log');
        caffe.init_log(caffe_log_file_base);
        caffe_net = caffe.Net(opts.net_def_file, 'test');
        caffe_net.copy_from(opts.net_file);

        % init log
        timestamp = datestr(datevec(now()), 'yyyymmdd_HHMMSS');
        helper.mkdir_if_missing(fullfile(cache_dir, 'log'));
        log_file = fullfile(cache_dir, 'log', ['test_', timestamp, '.txt']);
        diary(log_file);

        % set random seed
        prev_rng = helper.seed_rand(conf.rng_seed);
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
        % control number of images to be tested
        num_images = length(imdb.image_ids);
        % all detections are collected into:
        %    all_boxes[image] = N x 5 array of detections in
        %    (x1, y1, x2, y2, score)
        aboxes = cell(num_images, 1);
        %abox_deltas = cell(num_images, 1);
        %aanchors = cell(num_images, 1);
        %ascores = cell(num_images, 1);
        
        count = 0;
        for i = 1:num_images
            count = count + 1;
            fprintf('test (%s) %d/%d ', imdb.name, count, num_images);
            th = tic;
            im = imread(imdb.image_at(i));

            %[boxes, scores, abox_deltas{i}, aanchors{i}, ascores{i}] = proposal_im_detect_conv3_4(conf, caffe_net, im);
            % 0114 get rid of abox_deltas{i}, aanchors{i}, ascores{i}, they
            % are of no use, but takes up too much cpu memory
            if opts.three_scales
                [boxes, scores] = proposal_im_detect_conv3_4_scale3(conf, caffe_net, im);
                fprintf(' time: %.3fs\n', toc(th));  
                %0112 added to save space 0124 changed for scale3
                %scores = scores(scores >= 0.3, :);
                %boxes = boxes(scores >= 0.3, :);
                %aboxes{i} = [boxes, scores];
                aboxes_tmp = cell(1, numel(scores));
                for j = 1:numel(scores)
                    % first prune box then score
                    boxes{j} = boxes{j}(scores{j} >= 0.3, :);
                    scores{j} = scores{j}(scores{j} >= 0.3, :);
                    aboxes_tmp{j} = [boxes{j}, scores{j}];
                    aboxes{i} = cat(1, aboxes{i}, aboxes_tmp{j});
                end
                
                if 0
                    % debugging visualizations
                    im = imread(imdb.image_at(i));
                    figure(1),clf;
                    imshow(im);
                    color_cell = {'g', 'c', 'm'};
                    for j = 1:numel(scores)
                        keep = nms(aboxes_tmp{j}, 0.3);
                        bbs_show = aboxes_tmp{j}(keep, :);
                        bbs_show = bbs_show(bbs_show(:,5)>=0.9, :);
                        bbs_show(:, 3) = bbs_show(:, 3) - bbs_show(:, 1) + 1;
                        bbs_show(:, 4) = bbs_show(:, 4) - bbs_show(:, 2) + 1;
                        bbApply('draw',bbs_show,color_cell{j});
                    end
                end
            else
                [boxes, scores] = helper.rpn_im_detect(conf, caffe_net, im);
                fprintf(' time: %.3fs\n', toc(th));  
                %0112 added to save space
                scores = scores(scores >= 0.5, :);
                boxes = boxes(scores >= 0.5, :);
                aboxes{i} = [boxes, scores];
                if 0
                    % debugging visualizations
                    im = imread(imdb.image_at(i));
                    keep = nms(aboxes{i}, 0.3);
                    figure(1),clf;
                    imshow(im);
                    bbs_show = aboxes{i}(keep, :);
                    bbs_show = bbs_show(bbs_show(:,5)>=0.8, :);
                    bbs_show(:, 3) = bbs_show(:, 3) - bbs_show(:, 1) + 1;
                    bbs_show(:, 4) = bbs_show(:, 4) - bbs_show(:, 2) + 1;
                    bbApply('draw',bbs_show,'m');
                end
            end
        end    
        save(fullfile(cache_dir, ['proposal_boxes_' imdb.name opts.suffix]), 'aboxes', '-v7.3');
        
        diary off;
        caffe.reset_all(); 
        rng(prev_rng);
    end
end
