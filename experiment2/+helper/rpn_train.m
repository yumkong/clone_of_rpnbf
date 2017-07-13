function save_model_path = rpn_train(conf, imdb_train, roidb_train, varargin)
% save_model_path = proposal_train_caltech(conf, imdb_train, roidb_train, varargin)
% --------------------------------------------------------
% RPN_BF
% Copyright (c) 2016, Liliang Zhang
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------   
%% -------------------- CONFIG --------------------
% inputs
    ip = inputParser;
    ip.addRequired('conf',                                      @isstruct);
    ip.addRequired('imdb_train',                                @iscell);
    ip.addRequired('roidb_train',                               @iscell);
    ip.addParamValue('do_val',              true,              @isscalar);
    ip.addParamValue('imdb_val',            struct(),           @isstruct);
    ip.addParamValue('roidb_val',           struct(),           @isstruct);
    
    ip.addParamValue('val_iters',           1000,                 @isscalar);%1000
    ip.addParamValue('val_interval',        2000,               @isscalar);%2000
    ip.addParamValue('snapshot_interval',...
                                            2000,              @isscalar); %2000
                                                                       
    % Max pixel size of a scaled input image
    ip.addParamValue('solver_def_file',     fullfile(pwd, 'proposal_models', 'Zeiler_conv5', 'solver.prototxt'), ...
                                                        @isstr);
    ip.addParamValue('net_file',            fullfile(pwd, 'proposal_models', 'Zeiler_conv5', 'Zeiler_conv5.caffemodel'), ...
                                                        @isstr);
    ip.addParamValue('cache_name',          'Zeiler_conv5', ...
                                                        @isstr);
                                                    
    ip.addParamValue('exp_name',          'tmp', ...
                                                        @isstr);
                                                    
    ip.addParamValue('empty_image_sample_step',    1,          @isscalar);     
    
    
    
    ip.parse(conf, imdb_train, roidb_train, varargin{:});
    opts = ip.Results;
    
%% try to find trained model
    imdbs_name = cell2mat(cellfun(@(x) x.name, imdb_train, 'UniformOutput', false));
    cache_dir = fullfile(pwd, 'output2', opts.exp_name, 'rpn_cache', opts.cache_name, imdbs_name);
    %cache_dir = fullfile(pwd, 'output', opts.cache_name, imdbs_name);
    helper.mkdir_if_missing(cache_dir);
	save_model_path = fullfile(cache_dir, 'final');
    if exist(save_model_path, 'file')
        return;
    end

   % in current implentation, we set the ratio of foreground images
   % (which contain at least one pedestrian)
   % as 50% for fully using background images (which contain no pedestrian
   % at all) while balancing their propotion).
   %opts.fg_image_ratio = 0.5;  % no use in widerface
    
%% init  
    % init caffe solver
    %imdbs_name = cell2mat(cellfun(@(x) x.name, imdb_train, 'UniformOutput', false));
    %cache_dir = fullfile(pwd, 'output', opts.exp_name, 'rpn_cachedir', opts.cache_name, imdbs_name);
    %mkdir_if_missing(cache_dir);
    caffe_log_file_base = fullfile(cache_dir, 'caffe_log');
    caffe.init_log(caffe_log_file_base);
    caffe_solver = caffe.Solver(opts.solver_def_file);
    caffe_solver.net.copy_from(opts.net_file);
    
    % init log
    timestamp = datestr(datevec(now()), 'yyyymmdd_HHMMSS');
    helper.mkdir_if_missing(fullfile(cache_dir, 'log'));
    log_file = fullfile(cache_dir, 'log', ['train_', timestamp, '.txt']);
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
    
    disp('conf:');
    disp(conf);
    disp('opts:');
    disp(opts);
    
%% making tran/val data
    fprintf('Preparing training data...');
    train_roi_name = fullfile(cache_dir, 'train_input_roidb_all.mat');
    test_roi_name = fullfile(cache_dir, 'test_input_roidb_all.mat');
    try
        load(train_roi_name);
    catch
        [image_roidb_train, bbox_means, bbox_stds]...
                            = helper.rpn_prepare_roidb(conf, opts.imdb_train, opts.roidb_train);
        save(train_roi_name, 'image_roidb_train','bbox_means', 'bbox_stds','-v7.3');
    end
    fprintf('Done.\n');
    
    if opts.do_val
        fprintf('Preparing validation data...');
        try
            load(test_roi_name);
        catch
            [image_roidb_val]...
                                = helper.rpn_prepare_roidb(conf, opts.imdb_val, opts.roidb_val, bbox_means, bbox_stds);
            save(test_roi_name, 'image_roidb_val','-v7.3');
        end
        fprintf('Done.\n');

        % fix validation data
        % 1012 changed to use all validation set
        shuffled_inds_val   = generate_random_minibatch([], image_roidb_val, conf.ims_per_batch);
        shuffled_inds_val   = shuffled_inds_val(randperm(length(shuffled_inds_val), opts.val_iters));
        %shuffled_inds_val = num2cell(1:size(image_roidb_val,1));
    end
    
    conf.classes        = opts.imdb_train{1}.classes;
    
%%  try to train/val with images which have maximum size potentially, to validate whether the gpu memory is enough  
    check_gpu_memory(conf, caffe_solver, opts.do_val);
     
%% -------------------- Training -------------------- 

    proposal_generate_minibatch_fun = @helper.rpn_generate_minibatch;

    % training
    shuffled_inds = [];
    train_results = [];  
    val_results = [];
    iter_ = caffe_solver.iter();
    max_iter = caffe_solver.max_iter();

    % 0927 added to record plot info
    modelFigPath = fullfile(cache_dir, 'net-train.pdf');  % plot save path
    tmp_struct = struct('err_fg', [], 'err_bg', [], 'loss_cls', [], 'loss_bbox', []);
    history_rec = struct('train',tmp_struct,'val',tmp_struct, 'num', 0);
    %1009 changed so that validation can be done within while loop
    while (iter_ <= max_iter)
        % begin time counting
        start_time = tic;
        
        caffe_solver.net.set_phase('train');

        % generate minibatch training data
        [shuffled_inds, sub_db_inds] = generate_random_minibatch(shuffled_inds, image_roidb_train, conf.ims_per_batch);        
        [net_inputs, scale_inds] = proposal_generate_minibatch_fun(conf, image_roidb_train(sub_db_inds));
        
        caffe_solver.net.reshape_as_input(net_inputs);

        % one iter SGD update
        caffe_solver.net.set_input_data(net_inputs);
        caffe_solver.step(1);
        %end time counting
        cost_time = toc(start_time);
%         fprintf('\n');
%         for kk = 1:length(caffe_blob_names)
%            blob_der_compare(caffe_solver.net, caffe_blob_names{kk}, der_cell{mat_blob_indices(2*kk-1)}, der_cell{mat_blob_indices(2*kk)}); 
%         end
%         fprintf('\n');
        
        rst = caffe_solver.net.get_output();
        rst = check_error(rst, caffe_solver);
        
        %format long
        fprintf('Iter %d, Image %d, %d: %.1f Hz, ', iter_, sub_db_inds(1), sub_db_inds(2), 1/cost_time);
        for kkk = 1:length(rst)
            fprintf('%s = %.4f, ',rst(kkk).blob_name, rst(kkk).data); 
        end
        fprintf('\n');
        
        train_results = helper.parse_rst(train_results, rst);
        % check_loss(rst, caffe_solver, net_inputs);

        % do valdiation per val_interval iterations
        if ~mod(iter_, opts.val_interval) 
            if opts.do_val
                val_results = do_validation(conf, caffe_solver, proposal_generate_minibatch_fun, image_roidb_val, shuffled_inds_val);
            end
            % 0927 changed: showstate + plot
            %show_state(iter_, train_results, val_results);
            history_rec = show_state_and_plot(iter_, train_results, val_results, history_rec, modelFigPath);
            
            %show_state(iter_, train_results, val_results);
            train_results = [];
            diary; diary; % flush diary
        end
        
        % snapshot
        if ~mod(iter_, opts.snapshot_interval)
            snapshot(conf, caffe_solver, bbox_means, bbox_stds, cache_dir, sprintf('iter_%d', iter_));
        end
        
        iter_ = caffe_solver.iter();
    end
    
    % final validation
    % liu@0927 commented, because now all validation can be done during while loop
    %if opts.do_val
    %    do_validation(conf, caffe_solver, proposal_generate_minibatch_fun, image_roidb_val, shuffled_inds_val);
    %end
    % final snapshot
    % liu@0927 commented, because now all snapshot can be done during while loop
    %snapshot(conf, caffe_solver, bbox_means, bbox_stds, cache_dir, sprintf('iter_%d', iter_));
    save_model_path = snapshot(conf, caffe_solver, bbox_means, bbox_stds, cache_dir, 'final');

    diary off;
    caffe.reset_all(); 
    rng(prev_rng);
 
end

function val_results = do_validation(conf, caffe_solver, proposal_generate_minibatch_fun, image_roidb_val, shuffled_inds_val)
    % before each validation clear the history results
    val_results = [];

    caffe_solver.net.set_phase('test');
    for i = 1:length(shuffled_inds_val)
        sub_db_inds = shuffled_inds_val{i};
        [net_inputs, ~] = proposal_generate_minibatch_fun(conf, image_roidb_val(sub_db_inds));

        % Reshape net's input blobs
        caffe_solver.net.reshape_as_input(net_inputs);

        caffe_solver.net.forward(net_inputs);
        rst = caffe_solver.net.get_output();
        rst = check_error(rst, caffe_solver);  
        val_results = helper.parse_rst(val_results, rst);
    end
end

function [shuffled_inds, sub_inds] = generate_random_minibatch(shuffled_inds, image_roidb_train, ims_per_batch)

    % shuffle training data per batch
    if isempty(shuffled_inds)
        % make sure each minibatch, only has horizontal images or vertical
        % images, to save gpu memory
        
        hori_image_inds = arrayfun(@(x) x.im_size(2) >= x.im_size(1), image_roidb_train, 'UniformOutput', true);
        vert_image_inds = ~hori_image_inds;
        hori_image_inds = find(hori_image_inds);
        vert_image_inds = find(vert_image_inds);
        
        % random perm
        lim = floor(length(hori_image_inds) / ims_per_batch) * ims_per_batch;
        hori_image_inds = hori_image_inds(randperm(length(hori_image_inds), lim));
        lim = floor(length(vert_image_inds) / ims_per_batch) * ims_per_batch;
        vert_image_inds = vert_image_inds(randperm(length(vert_image_inds), lim));
        
        % combine sample for each ims_per_batch 
        hori_image_inds = reshape(hori_image_inds, ims_per_batch, []);
        vert_image_inds = reshape(vert_image_inds, ims_per_batch, []);
        
        shuffled_inds = [hori_image_inds, vert_image_inds];
        shuffled_inds = shuffled_inds(:, randperm(size(shuffled_inds, 2)));
        
        shuffled_inds = num2cell(shuffled_inds, 1);
    end
    
    if nargout > 1
        % generate minibatch training data
        sub_inds = shuffled_inds{1};
        assert(length(sub_inds) == ims_per_batch);
        shuffled_inds(1) = [];
    end
end

function rst = check_error(rst, caffe_solver)

    cls_score = caffe_solver.net.blobs('proposal_cls_score_reshape').get_data();
    labels = caffe_solver.net.blobs('labels_reshape').get_data();
    labels_weights = caffe_solver.net.blobs('labels_weights_reshape').get_data();
    
    accurate_fg = (cls_score(:, :, 2, :) > cls_score(:, :, 1, :)) & (labels == 1);
    accurate_bg = (cls_score(:, :, 2, :) <= cls_score(:, :, 1, :)) & (labels == 0);
    accurate = accurate_fg | accurate_bg;
    accuracy_fg = sum(accurate_fg(:) .* labels_weights(:)) / (sum(labels_weights(labels == 1)) + eps);
    accuracy_bg = sum(accurate_bg(:) .* labels_weights(:)) / (sum(labels_weights(labels == 0)) + eps);
    
    rst(end+1) = struct('blob_name', 'accuracy_fg', 'data', accuracy_fg);
    rst(end+1) = struct('blob_name', 'accuracy_bg', 'data', accuracy_bg);
end

function check_gpu_memory(conf, caffe_solver, do_val)
%%  try to train/val with images which have maximum size potentially, to validate whether the gpu memory is enough  

    % generate pseudo training data with max size
    im_blob = single(zeros(max(conf.scales), conf.max_size, 3, conf.ims_per_batch));
    
    anchor_num = size(conf.anchors, 1);
    output_width = conf.output_width_map.values({size(im_blob, 1)});
    output_width = output_width{1};
    output_height = conf.output_width_map.values({size(im_blob, 2)});
    output_height = output_height{1};
    labels_blob = single(zeros(output_width, output_height, anchor_num, conf.ims_per_batch));
    labels_weights = labels_blob;
    bbox_targets_blob = single(zeros(output_width, output_height, anchor_num*4, conf.ims_per_batch));
    bbox_loss_weights_blob = bbox_targets_blob;

    net_inputs = {im_blob, labels_blob, labels_weights, bbox_targets_blob, bbox_loss_weights_blob};
    
     % Reshape net's input blobs
    caffe_solver.net.reshape_as_input(net_inputs);

    % one iter SGD update
    caffe_solver.net.set_input_data(net_inputs);
    caffe_solver.step(1);

    if do_val
        % use the same net with train to save memory
        caffe_solver.net.set_phase('test');
        caffe_solver.net.forward(net_inputs);
        caffe_solver.net.set_phase('train');
    end
end

function model_path = snapshot(conf, caffe_solver, bbox_means, bbox_stds, cache_dir, file_name)
    anchor_size = size(conf.anchors, 1);
    bbox_stds_flatten = repmat(reshape(bbox_stds', [], 1), anchor_size, 1);
    bbox_means_flatten = repmat(reshape(bbox_means', [], 1), anchor_size, 1);
    
    % merge bbox_means, bbox_stds into the model
    bbox_pred_layer_name = 'proposal_bbox_pred';
    weights = caffe_solver.net.params(bbox_pred_layer_name, 1).get_data();
    biase = caffe_solver.net.params(bbox_pred_layer_name, 2).get_data();
    weights_back = weights;
    biase_back = biase;
    
    weights = ...
        bsxfun(@times, weights, permute(bbox_stds_flatten, [2, 3, 4, 1])); % weights = weights * stds; 
    biase = ...
        biase .* bbox_stds_flatten + bbox_means_flatten; % bias = bias * stds + means;
    
    caffe_solver.net.set_params_data(bbox_pred_layer_name, 1, weights);
    caffe_solver.net.set_params_data(bbox_pred_layer_name, 2, biase);
    
    model_path = fullfile(cache_dir, file_name);
    caffe_solver.net.save(model_path);
    fprintf('Saved as %s\n', model_path);
    
    % restore net to original state
    caffe_solver.net.set_params_data(bbox_pred_layer_name, 1, weights_back);
    caffe_solver.net.set_params_data(bbox_pred_layer_name, 2, biase_back);
end

%function show_state(iter, train_results, val_results)
function history_rec = show_state_and_plot(iter, train_results, val_results, history_rec, modelFigPath)
    % --------- begin previously show_state part ------------
    fprintf('\n------------------------- Iteration %d -------------------------\n', iter);
    fprintf('Training : err_fg %.3g, err_bg %.3g, loss (cls %.3g + reg %.3g)\n', ...
        1 - mean(train_results.accuracy_fg.data), 1 - mean(train_results.accuracy_bg.data), ...
        mean(train_results.loss_cls.data), ...
        mean(train_results.loss_bbox.data));
    if exist('val_results', 'var') && ~isempty(val_results)
        fprintf('Testing  : err_fg %.3g, err_bg %.3g, loss (cls %.3g + reg %.3g)\n', ...
            1 - mean(val_results.accuracy_fg.data), 1 - mean(val_results.accuracy_bg.data), ...
            mean(val_results.loss_cls.data), ...
            mean(val_results.loss_bbox.data));
    end
    % --------- end previously show_state part ------------
    % ========= newly added plot part =====================
    history_rec.train.err_fg = [history_rec.train.err_fg; 1 - mean(train_results.accuracy_fg.data)];
    history_rec.train.err_bg = [history_rec.train.err_bg; 1 - mean(train_results.accuracy_bg.data)];
    history_rec.train.loss_cls = [history_rec.train.loss_cls; mean(train_results.loss_cls.data)];
    history_rec.train.loss_bbox = [history_rec.train.loss_bbox; mean(train_results.loss_bbox.data)];
    history_rec.val.err_fg = [history_rec.val.err_fg; 1 - mean(val_results.accuracy_fg.data)];
    history_rec.val.err_bg = [history_rec.val.err_bg; 1 - mean(val_results.accuracy_bg.data)];
    history_rec.val.loss_cls = [history_rec.val.loss_cls; mean(val_results.loss_cls.data)];
    history_rec.val.loss_bbox = [history_rec.val.loss_bbox; mean(val_results.loss_bbox.data)];
    history_rec.num = history_rec.num + 1;
    % draw it
    figure(1) ; clf ;
    plots = {'err_fg', 'err_bg', 'loss_cls', 'loss_bbox'};
    for p = plots
      c_p = char(p) ;
      values = zeros(0, history_rec.num) ;
      leg = {} ;
      for f = {'train', 'val'}
        c_f = char(f) ;
        if isfield(history_rec.(c_f), c_p)
          %tmp = [history_rec.(c_f).(c_p)] ;
          %values(end+1,:) = tmp(1,:)' ;
          values(end+1,:) = [history_rec.(c_f).(c_p)]';
          leg{end+1} = c_f;
        end
      end
      subplot(1,numel(plots),find(strcmp(c_p, plots))) ;
      plot(1:history_rec.num, values','o-') ;
      xlabel('epoch') ;
      title(c_p) ;
      legend(leg{:}) ;
      grid on ;
    end
    drawnow ;
    print(1, modelFigPath, '-dpdf') ;
end