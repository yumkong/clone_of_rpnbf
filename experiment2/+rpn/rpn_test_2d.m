function [feat_list, label_list] = rpn_test_2d(conf, imdb, roidb, cache_dir, varargin)
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
    ip.addRequired('roidb',                              @isstruct);
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
    %0715 randomly select feat_per_img features from each image for visual
    ip.addParamValue('feat_per_img',    6,          @isnumeric);
    
    ip.parse(conf, imdb, roidb, cache_dir, varargin{:});
    opts = ip.Results;
    
    %##### note@1013:every time you want to regenerate bboxes, need to
    %delete the proposal_boxes_*.mat file in
    %output/VGG16_wider*/rpn_cachedir/rpn_widerface_VGG16_stage1_rpn
    %cache_dir = opts.cache_dir; %fullfile(pwd, 'output', conf.exp_name, 'rpn_cachedir', opts.cache_name, imdb.name);
	%cache_dir = fullfile(pwd, 'output', opts.cache_name, imdb.name);
    try
        % try to load cache
        ld = load(fullfile(cache_dir, ['2Dfeat' imdb.name opts.suffix]));
        %aboxes = ld.aboxes;
        feat_list = ld.feat_list;
        label_list = ld.label_list;
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
        feat_pool = cell(num_images, 1);
        label_pool = cell(num_images, 1);
        
        count = 0;
        for i = 1:num_images
            count = count + 1;
            fprintf('test (%s) %d/%d ', imdb.name, count, num_images);
            th = tic;
            im = imread(imdb.image_at(i));

            %[boxes, scores, abox_deltas{i}, aanchors{i}, ascores{i}] = proposal_im_detect_conv3_4(conf, caffe_net, im);
            % 0114 get rid of abox_deltas{i}, aanchors{i}, ascores{i}, they
            % are of no use, but takes up too much cpu memory
            if ~opts.three_scales
                % feat_list: N x 2, label_list: N x 1 (current N = 50)
                [feat_pool{i}, label_pool{i}] = rpn.rpn_im_detect_2d(conf, caffe_net, im, roidb(i), opts.feat_per_img);
                fprintf(' time: %.3fs\n', toc(th));  
            end
        end    
        % total list of all test images
        feat_list = cell2mat(feat_pool);
        label_list = cell2mat(label_pool);
        save(fullfile(cache_dir, ['2Dfeat' imdb.name opts.suffix]), 'feat_list', 'label_list', '-v7.3');
        
        diary off;
        caffe.reset_all(); 
        rng(prev_rng);
    end
end
