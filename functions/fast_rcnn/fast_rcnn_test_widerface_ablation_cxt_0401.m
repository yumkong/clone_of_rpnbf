function mAP = fast_rcnn_test_widerface_ablation_cxt_0401(conf, imdb, roidb, varargin)
% mAP = fast_rcnn_test(conf, imdb, roidb, varargin)
% --------------------------------------------------------
% Fast R-CNN
% Reimplementation based on Python Fast R-CNN (https://github.com/rbgirshick/fast-rcnn)
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------

%% inputs
    ip = inputParser;
    ip.addRequired('conf',                              @isstruct);
    ip.addRequired('imdb',                              @isstruct);
    ip.addRequired('roidb',                             @isstruct);
    ip.addParamValue('net_def_file',    '', 			@isstr);
    ip.addParamValue('net_file',        '', 			@isstr);
    ip.addParamValue('cache_name',      '', 			@isstr);                                         
    ip.addParamValue('suffix',          '',             @isstr);
    ip.addParamValue('ignore_cache',    false,          @islogical);
    ip.addParamValue('exp_name',          'tmp', ...
                                                        @isstr);
    ip.parse(conf, imdb, roidb, varargin{:});
    opts = ip.Results;
    

%%  set cache dir
    %cache_dir = fullfile(pwd, 'output', 'fast_rcnn_cachedir', opts.cache_name, imdb.name);
    cache_dir = fullfile(pwd, 'output', opts.exp_name, 'fast_rcnn_cachedir', opts.cache_name, imdb.name); 
    mkdir_if_missing(cache_dir);

%%  init log
    timestamp = datestr(datevec(now()), 'yyyymmdd_HHMMSS');
    mkdir_if_missing(fullfile(cache_dir, 'log'));
    log_file = fullfile(cache_dir, 'log', ['test_', timestamp, '.txt']);
    diary(log_file);
    
    num_images = length(imdb.image_ids);
    save_file = fullfile(cache_dir, ['aboxes_' imdb.name opts.suffix]);
    try
        ld = load(save_file);%'aboxes_old', 'aboxes_new','score_ind_old', 'score_ind_new'
        aboxes_old = ld.aboxes_old;
        aboxes_new = ld.aboxes_new;
    catch    
%%      testing 
        % init caffe net
        caffe_log_file_base = fullfile(cache_dir, 'caffe_log');
        caffe.init_log(caffe_log_file_base);
        caffe_net = caffe.Net(opts.net_def_file, 'test');
        caffe_net.copy_from(opts.net_file);

        % set random seed
        prev_rng = seed_rand(conf.rng_seed);
        caffe.set_random_seed(conf.rng_seed);

        % set gpu/cpu
        if conf.use_gpu
            caffe.set_mode_gpu();
        else
            caffe.set_mode_cpu();
        end             

        % determine the maximum number of rois in testing 
        %max_rois_num_in_gpu = check_gpu_memory(conf, caffe_net);
        max_rois_num_in_gpu = 1000;

        disp('opts:');
        disp(opts);
        disp('conf:');
        disp(conf);
        
        aboxes_old = cell(length(imdb.image_ids), 1);
        aboxes_new = cell(length(imdb.image_ids), 1);
        aboxes_v1 = cell(length(imdb.image_ids), 1);
        aboxes_v2 = cell(length(imdb.image_ids), 1);
        aboxes_v3 = cell(length(imdb.image_ids), 1);
        aboxes_v4 = cell(length(imdb.image_ids), 1);
        aboxes_v5 = cell(length(imdb.image_ids), 1);
        aboxes_v6 = cell(length(imdb.image_ids), 1);
        aboxes_v7 = cell(length(imdb.image_ids), 1);
        aboxes_v8 = cell(length(imdb.image_ids), 1);
        aboxes_v9 = cell(length(imdb.image_ids), 1);

        count = 0;
        t_start = tic;
        for i = 1:num_images
            count = count + 1;
            fprintf('%s: test (%s) %d/%d ', procid(), imdb.name, count, num_images);
            th = tic;
            d = roidb.rois(i);
            im = imread(imdb.image_at(i));

            rpn_boxes = d.boxes(~d.gt, :);
            rpn_score = d.scores(~d.gt, :);
            if ~isempty(rpn_score)
                [~, scores_ind] = sort(rpn_score, 'descend');
                rpn_boxes = rpn_boxes(scores_ind, :);
                rpn_score = rpn_score(scores_ind, :);
            end
            % 0326 added: avoid gpu out of memory 
            % max_rois_num_in_gpu = 1000
            if size(rpn_boxes, 1) > max_rois_num_in_gpu
                rpn_boxes = rpn_boxes(1:max_rois_num_in_gpu, :);
                rpn_score = rpn_score(1:max_rois_num_in_gpu, :);
            end
            %[boxes, scores] = fast_rcnn_im_detect_widerface_ablation(conf, caffe_net, im, d.boxes, max_rois_num_in_gpu);
            fastrcnn_score = fast_rcnn_im_detect_widerface_ablation_cxt(conf, caffe_net, im, rpn_boxes, max_rois_num_in_gpu);
			fastrcnn_score_pno = fastrcnn_score;
            
            if ~isempty(rpn_boxes)
                aboxes_old{i} = [rpn_boxes rpn_score];
                aboxes_new{i} = [rpn_boxes fastrcnn_score_pno];
            else
                aboxes_old{i} = [];
                aboxes_new{i} = [];
            end
            fprintf(' time: %.3fs\n', toc(th));     
        end
        %0331: only save these two file can reproduce all the results
        save(save_file, 'aboxes_old', 'aboxes_new');
        fprintf('test all images in %f seconds.\n', toc(t_start));
        
        caffe.reset_all(); 
        rng(prev_rng);
    end
    
    count = 0;
    for i = 1:num_images
        count = count + 1;
        fprintf('%s: test (%s) %d/%d \n', procid(), imdb.name, count, num_images);
        
        if ~isempty(aboxes_old{i})
            rpn_boxes = aboxes_old{i}(:, 1:4);
            rpn_score = aboxes_old{i}(:, 5);
            fastrcnn_score_raw = aboxes_new{i}(:, 5);
            %0328: cubic root - optimal by round 2
            fastrcnn_score = nthroot(fastrcnn_score_raw, 3);
            %0328 shrink to [0.8 1] - optimal by round 1
            fastrcnn_score = (fastrcnn_score - min(fastrcnn_score))/(max(fastrcnn_score) - min(fastrcnn_score))*0.2 + 0.8;
        end
        
        if ~isempty(rpn_boxes)
            aboxes_old{i} = [rpn_boxes rpn_score];
            aboxes_new{i} = [rpn_boxes fastrcnn_score];
            % 1 * rpn + 0.5 * fastrcnn_score is optimal by round 3&4
            aboxes_v1{i} = [rpn_boxes (rpn_score + 0.1 * fastrcnn_score)];
            aboxes_v2{i} = [rpn_boxes (rpn_score + 0.2 * fastrcnn_score)];
            aboxes_v3{i} = [rpn_boxes (rpn_score + 0.3 * fastrcnn_score)];
            aboxes_v4{i} = [rpn_boxes (rpn_score + 0.4 * fastrcnn_score)];
            aboxes_v5{i} = [rpn_boxes (rpn_score + 0.5 * fastrcnn_score)];
            aboxes_v6{i} = [rpn_boxes (rpn_score + 0.6 * fastrcnn_score)];
            aboxes_v7{i} = [rpn_boxes (rpn_score + 0.7 * fastrcnn_score)];
            aboxes_v8{i} = [rpn_boxes (rpn_score + 0.8 * fastrcnn_score)];
            aboxes_v9{i} = [rpn_boxes (rpn_score + 0.9 * fastrcnn_score)];
        else
            aboxes_old{i} = [];
            aboxes_new{i} = [];
            aboxes_v1{i} = [];
            aboxes_v2{i} = [];
            aboxes_v3{i} = [];
            aboxes_v4{i} = [];
            aboxes_v5{i} = [];
            aboxes_v6{i} = [];
            aboxes_v7{i} = [];
            aboxes_v8{i} = [];
            aboxes_v9{i} = [];
        end
        
        % 0310: for rpn score
        aboxes_old{i} = pseudoNMS_v8_twopath(aboxes_old{i}, 3);%nms_option=3
        if ~isempty(aboxes_old{i})
            [~, scores_ind] = sort(aboxes_old{i}(:,5), 'descend');
            aboxes_old{i} = aboxes_old{i}(scores_ind, :);
        end

        % 0310: for fastrcnn score
        aboxes_new{i} = pseudoNMS_v8_twopath(aboxes_new{i}, 3);%nms_option=3
        if ~isempty(aboxes_new{i})
            [~, scores_ind] = sort(aboxes_new{i}(:,5), 'descend');
            aboxes_new{i} = aboxes_new{i}(scores_ind, :);
        end

        %0331 added for v1-v9
        aboxes_v1{i} = pseudoNMS_v8_twopath(aboxes_v1{i}, 3);%nms_option=3
        if ~isempty(aboxes_v1{i})
            [~, scores_ind] = sort(aboxes_v1{i}(:,5), 'descend');
            aboxes_v1{i} = aboxes_v1{i}(scores_ind, :);
        end
        aboxes_v2{i} = pseudoNMS_v8_twopath(aboxes_v2{i}, 3);%nms_option=3
        if ~isempty(aboxes_v2{i})
            [~, scores_ind] = sort(aboxes_v2{i}(:,5), 'descend');
            aboxes_v2{i} = aboxes_v2{i}(scores_ind, :);
        end
        aboxes_v3{i} = pseudoNMS_v8_twopath(aboxes_v3{i}, 3);%nms_option=3
        if ~isempty(aboxes_v3{i})
            [~, scores_ind] = sort(aboxes_v3{i}(:,5), 'descend');
            aboxes_v3{i} = aboxes_v3{i}(scores_ind, :);
        end
        aboxes_v4{i} = pseudoNMS_v8_twopath(aboxes_v4{i}, 3);%nms_option=3
        if ~isempty(aboxes_v4{i})
            [~, scores_ind] = sort(aboxes_v4{i}(:,5), 'descend');
            aboxes_v4{i} = aboxes_v4{i}(scores_ind, :);
        end
        aboxes_v5{i} = pseudoNMS_v8_twopath(aboxes_v5{i}, 3);%nms_option=3
        if ~isempty(aboxes_v5{i})
            [~, scores_ind] = sort(aboxes_v5{i}(:,5), 'descend');
            aboxes_v5{i} = aboxes_v5{i}(scores_ind, :);
        end
        aboxes_v6{i} = pseudoNMS_v8_twopath(aboxes_v6{i}, 3);%nms_option=3
        if ~isempty(aboxes_v6{i})
            [~, scores_ind] = sort(aboxes_v6{i}(:,5), 'descend');
            aboxes_v6{i} = aboxes_v6{i}(scores_ind, :);
        end
        aboxes_v7{i} = pseudoNMS_v8_twopath(aboxes_v7{i}, 3);%nms_option=3
        if ~isempty(aboxes_v7{i})
            [~, scores_ind] = sort(aboxes_v7{i}(:,5), 'descend');
            aboxes_v7{i} = aboxes_v7{i}(scores_ind, :);
        end
        aboxes_v8{i} = pseudoNMS_v8_twopath(aboxes_v8{i}, 3);%nms_option=3
        if ~isempty(aboxes_v8{i})
            [~, scores_ind] = sort(aboxes_v8{i}(:,5), 'descend');
            aboxes_v8{i} = aboxes_v8{i}(scores_ind, :);
        end
        aboxes_v9{i} = pseudoNMS_v8_twopath(aboxes_v9{i}, 3);%nms_option=3
        if ~isempty(aboxes_v9{i})
            [~, scores_ind] = sort(aboxes_v9{i}(:,5), 'descend');
            aboxes_v9{i} = aboxes_v9{i}(scores_ind, :);
        end
    end
% 0206 added
    start_thresh = 5; %5
    thresh_interval = 3;%3
    thresh_end = 500; % 500
    [gt_num_all, gt_recall_all, gt_num_pool, gt_recall_pool] = Get_Detector_Recall_finegrained(roidb, aboxes_old, start_thresh,thresh_interval,thresh_end);
    save(fullfile(cache_dir,'recall_vector_rpn.mat'),'gt_num_all', 'gt_recall_all', 'gt_num_pool', 'gt_recall_pool');
    fprintf('rpn all scales: gt recall rate = %d / %d = %.4f\n', gt_recall_all, gt_num_all, gt_recall_all/gt_num_all);
    
    [gt_num_all, gt_recall_all, gt_num_pool, gt_recall_pool] = Get_Detector_Recall_finegrained(roidb, aboxes_new, start_thresh,thresh_interval,thresh_end);
    save(fullfile(cache_dir,'recall_vector_fastrcnn.mat'),'gt_num_all', 'gt_recall_all', 'gt_num_pool', 'gt_recall_pool');
    fprintf('fastrcnn all scales: gt recall rate = %d / %d = %.4f\n', gt_recall_all, gt_num_all, gt_recall_all/gt_num_all);
    
    [gt_num_all, gt_recall_all, gt_num_pool, gt_recall_pool] = Get_Detector_Recall_finegrained(roidb, aboxes_v1, start_thresh,thresh_interval,thresh_end);
    save(fullfile(cache_dir,'recall_vector_v1.mat'),'gt_num_all', 'gt_recall_all', 'gt_num_pool', 'gt_recall_pool');
    fprintf('fastrcnn_v1 all scales: gt recall rate = %d / %d = %.4f\n', gt_recall_all, gt_num_all, gt_recall_all/gt_num_all);
    
    [gt_num_all, gt_recall_all, gt_num_pool, gt_recall_pool] = Get_Detector_Recall_finegrained(roidb, aboxes_v2, start_thresh,thresh_interval,thresh_end);
    save(fullfile(cache_dir,'recall_vector_v2.mat'),'gt_num_all', 'gt_recall_all', 'gt_num_pool', 'gt_recall_pool');
    fprintf('fastrcnn_v2 all scales: gt recall rate = %d / %d = %.4f\n', gt_recall_all, gt_num_all, gt_recall_all/gt_num_all);
    
    [gt_num_all, gt_recall_all, gt_num_pool, gt_recall_pool] = Get_Detector_Recall_finegrained(roidb, aboxes_v3, start_thresh,thresh_interval,thresh_end);
    save(fullfile(cache_dir,'recall_vector_v3.mat'),'gt_num_all', 'gt_recall_all', 'gt_num_pool', 'gt_recall_pool');
    fprintf('fastrcnn_v3 all scales: gt recall rate = %d / %d = %.4f\n', gt_recall_all, gt_num_all, gt_recall_all/gt_num_all);
    
    [gt_num_all, gt_recall_all, gt_num_pool, gt_recall_pool] = Get_Detector_Recall_finegrained(roidb, aboxes_v4, start_thresh,thresh_interval,thresh_end);
    save(fullfile(cache_dir,'recall_vector_v4.mat'),'gt_num_all', 'gt_recall_all', 'gt_num_pool', 'gt_recall_pool');
    fprintf('fastrcnn_v4 all scales: gt recall rate = %d / %d = %.4f\n', gt_recall_all, gt_num_all, gt_recall_all/gt_num_all);
    
    [gt_num_all, gt_recall_all, gt_num_pool, gt_recall_pool] = Get_Detector_Recall_finegrained(roidb, aboxes_v5, start_thresh,thresh_interval,thresh_end);
    save(fullfile(cache_dir,'recall_vector_v5.mat'),'gt_num_all', 'gt_recall_all', 'gt_num_pool', 'gt_recall_pool');
    fprintf('fastrcnn_v5 all scales: gt recall rate = %d / %d = %.4f\n', gt_recall_all, gt_num_all, gt_recall_all/gt_num_all);
    
    [gt_num_all, gt_recall_all, gt_num_pool, gt_recall_pool] = Get_Detector_Recall_finegrained(roidb, aboxes_v6, start_thresh,thresh_interval,thresh_end);
    save(fullfile(cache_dir,'recall_vector_v6.mat'),'gt_num_all', 'gt_recall_all', 'gt_num_pool', 'gt_recall_pool');
    fprintf('fastrcnn_v6 all scales: gt recall rate = %d / %d = %.4f\n', gt_recall_all, gt_num_all, gt_recall_all/gt_num_all);
    
    [gt_num_all, gt_recall_all, gt_num_pool, gt_recall_pool] = Get_Detector_Recall_finegrained(roidb, aboxes_v7, start_thresh,thresh_interval,thresh_end);
    save(fullfile(cache_dir,'recall_vector_v7.mat'),'gt_num_all', 'gt_recall_all', 'gt_num_pool', 'gt_recall_pool');
    fprintf('fastrcnn_v7 all scales: gt recall rate = %d / %d = %.4f\n', gt_recall_all, gt_num_all, gt_recall_all/gt_num_all);
    
    [gt_num_all, gt_recall_all, gt_num_pool, gt_recall_pool] = Get_Detector_Recall_finegrained(roidb, aboxes_v8, start_thresh,thresh_interval,thresh_end);
    save(fullfile(cache_dir,'recall_vector_v8.mat'),'gt_num_all', 'gt_recall_all', 'gt_num_pool', 'gt_recall_pool');
    fprintf('fastrcnn_v8 all scales: gt recall rate = %d / %d = %.4f\n', gt_recall_all, gt_num_all, gt_recall_all/gt_num_all);
    
    [gt_num_all, gt_recall_all, gt_num_pool, gt_recall_pool] = Get_Detector_Recall_finegrained(roidb, aboxes_v9, start_thresh,thresh_interval,thresh_end);
    save(fullfile(cache_dir,'recall_vector_v9.mat'),'gt_num_all', 'gt_recall_all', 'gt_num_pool', 'gt_recall_pool');
    fprintf('fastrcnn_v9 all scales: gt recall rate = %d / %d = %.4f\n', gt_recall_all, gt_num_all, gt_recall_all/gt_num_all);

    mAP = 0.0;
    diary off;
end

function [gt_num_all, gt_recall_all, gt_num_pool, gt_recall_pool] = Get_Detector_Recall_finegrained(roidb, aboxes, start_thresh,thresh_interval,thresh_end)
    gt_num_all = 0;
    gt_recall_all = 0;
    % 1229 added
    gt_num_pool = zeros(1, length(start_thresh:thresh_interval:thresh_end)+1);  % 4 ~ 14, 14~24, ...
    gt_recall_pool = gt_num_pool;

    %0110 added
    for i = 1:length(roidb.rois)
%         if ~isempty(aboxes{i})
%             aboxes{i} = aboxes{i}(end:-1:1,:);
%             aboxes{i}(:,5) = 1 - aboxes{i}(:,5);
%         end
        %gts = roidb.rois(i).boxes; % for widerface, no ignored bboxes
        gts = roidb.rois(i).boxes(roidb.rois(i).gt,:); % for widerface, no ignored bboxes
        if ~isempty(gts)
            % only leave 2x gt_num detected bboxes
            gt_num = size(gts, 1);
            %rois = aboxes{i}(:, 1:4);
            this_pred_num = min(2*gt_num, size(aboxes{i}, 1));
            if this_pred_num ~= 0
                rois = aboxes{i}(1:this_pred_num, 1:4);
            else
                rois = [];
            end
            
            face_height = gts(:,4) - gts(:,2) + 1;
            % total statistics
            idx_all = (face_height>= start_thresh);  % all:4-inf
            gts_all = gts(idx_all, :);
            if ~isempty(gts_all)
                gt_num_all = gt_num_all + size(gts_all, 1);
                if ~isempty(rois)
                    max_ols = max(boxoverlap(rois, gts_all));
                    if ~isempty(max_ols)
                        gt_recall_all = gt_recall_all + sum(max_ols >= 0.5);
                    end
                end
            end
                
            % segment statistics
            cnt = 0;
            for k = start_thresh:thresh_interval:thresh_end
                cnt = cnt + 1;
                part_idx = (face_height>= k) & (face_height < k + thresh_interval); % eg.:4~14
                part_gts = gts(part_idx, :);
            
                if ~isempty(part_gts)
                    gt_num_pool(cnt) = gt_num_pool(cnt) + size(part_gts, 1);
                    if ~isempty(rois)
                        max_ols = max(boxoverlap(rois, part_gts));  
                        if ~isempty(max_ols)
                            gt_recall_pool(cnt) = gt_recall_pool(cnt) + sum(max_ols >= 0.5);
                        end
                    end
                end
            end
            %0206 for 300-inf
            cnt = cnt + 1;
            part_idx = (face_height>= k + thresh_interval); % 300~inf
            part_gts = gts(part_idx, :);

            if ~isempty(part_gts)
                gt_num_pool(cnt) = gt_num_pool(cnt) + size(part_gts, 1);
                if ~isempty(rois)
                    max_ols = max(boxoverlap(rois, part_gts));
                    if ~isempty(max_ols)
                        gt_recall_pool(cnt) = gt_recall_pool(cnt) + sum(max_ols >= 0.5);
                    end
                end
            end
        end
    end
    %save('recall_conv2.mat','gt_num_all', 'gt_recall_all', 'gt_num_pool', 'gt_recall_pool');
    %fprintf('For all scales: gt recall rate = %d / %d = %.4f\n', gt_recall_all, gt_num_all, gt_recall_all/gt_num_all);
end

function max_rois_num = check_gpu_memory(conf, caffe_net)
%%  try to determine the maximum number of rois

    max_rois_num = 0;
    %for rois_num = 500:500:5000
    %for rois_num = 500:500:2000
    for rois_num = 100:100:1000  %0320 for conv2
        % generate pseudo testing data with max size
        im_blob = single(zeros(conf.max_size, conf.max_size, 3, 1));
        rois_blob = single(repmat([0; 0; 0; conf.max_size-1; conf.max_size-1], 1, rois_num));
        rois_blob = permute(rois_blob, [3, 4, 1, 2]);
		%0323 added
		rois_cxt_blob = single(repmat([0; 0; 0; conf.max_size-1; conf.max_size-1], 1, rois_num));
        rois_cxt_blob = permute(rois_cxt_blob, [3, 4, 1, 2]);

        net_inputs = {im_blob, rois_blob, rois_cxt_blob};

        % Reshape net's input blobs
        caffe_net.reshape_as_input(net_inputs);

        caffe_net.forward(net_inputs);
        gpuInfo = gpuDevice();

        max_rois_num = rois_num;
            
        if gpuInfo.FreeMemory < 2 * 10^9  % 2GB for safety
            break;
        end
    end

end


% ------------------------------------------------------------------------
function [boxes, box_inds, thresh] = keep_top_k(boxes, box_inds, end_at, top_k, thresh)
% ------------------------------------------------------------------------
    % Keep top K
    X = cat(1, boxes{1:end_at});
    if isempty(X)
        return;
    end
    scores = sort(X(:,end), 'descend');
    thresh = scores(min(length(scores), top_k));
    for image_index = 1:end_at
        if ~isempty(boxes{image_index})
            bbox = boxes{image_index};
            keep = find(bbox(:,end) >= thresh);
            boxes{image_index} = bbox(keep,:);
            box_inds{image_index} = box_inds{image_index}(keep);
        end
    end
end
