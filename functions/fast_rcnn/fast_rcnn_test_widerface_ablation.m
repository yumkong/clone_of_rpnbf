function mAP = fast_rcnn_test_widerface_ablation(conf, imdb, roidb, varargin)
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
    num_classes = imdb.num_classes;
    
    try
      aboxes = cell(num_classes, 1);
      if opts.ignore_cache
          throw('');
      end
      for i = 1:num_classes
        load(fullfile(cache_dir, [imdb.classes{i} '_boxes_' imdb.name opts.suffix]));
        aboxes{i} = boxes;
      end
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
        max_rois_num_in_gpu = check_gpu_memory(conf, caffe_net);

        disp('opts:');
        disp(opts);
        disp('conf:');
        disp(conf);
        
        %heuristic: keep an average of 40 detections per class per images prior to NMS
        max_per_set = 40 * num_images;
        % heuristic: keep at most 100 detection per class per image prior to NMS
        max_per_image = 100;
        % detection thresold for each class (this is adaptively set based on the max_per_set constraint)
        thresh = -inf * ones(num_classes, 1);
        % top_scores will hold one minheap of scores per class (used to enforce the max_per_set constraint)
        top_scores = cell(num_classes, 1);
        % all detections are collected into:
        %    all_boxes[cls][image] = N x 5 array of detections in
        %    (x1, y1, x2, y2, score)
%         aboxes = cell(num_classes, 1);
%         box_inds = cell(num_classes, 1);
%         for i = 1:num_classes
%             aboxes{i} = cell(length(imdb.image_ids), 1);
%             box_inds{i} = cell(length(imdb.image_ids), 1);
%         end
        
        aboxes_old = cell(length(imdb.image_ids), 1);
        aboxes_new = cell(length(imdb.image_ids), 1);

        count = 0;
        t_start = tic;
        for i = 1:num_images
            count = count + 1;
            fprintf('%s: test (%s) %d/%d ', procid(), imdb.name, count, num_images);
            th = tic;
            d = roidb.rois(i);
            im = imread(imdb.image_at(i));

            %[boxes, scores] = fast_rcnn_im_detect_widerface_ablation(conf, caffe_net, im, d.boxes, max_rois_num_in_gpu);
            scores = fast_rcnn_im_detect_widerface_ablation(conf, caffe_net, im, d.boxes, max_rois_num_in_gpu);
            
            %inds = find(~d.gt);
            aboxes_old{i} = [d.boxes(~d.gt, :) d.scores(~d.gt, :)];
            aboxes_new{i} = [d.boxes(~d.gt, :) scores(~d.gt, :)];
            aboxes_old{i} = pseudoNMS_v8_twopath(aboxes_old{i}, 3);%nms_option=3
            %0226 added: sort by score ()pseudoNMS may make the box
            %not sorted in descending order of scores
            if ~isempty(aboxes_old{i})
                [~, scores_ind] = sort(aboxes_old{i}(:,5), 'descend');
                aboxes_old{i} = aboxes_old{i}(scores_ind, :);
            end
            aboxes_new{i} = pseudoNMS_v8_twopath(aboxes_new{i}, 3);%nms_option=3
            %0226 added: sort by score ()pseudoNMS may make the box
            %not sorted in descending order of scores
            if ~isempty(aboxes_new{i})
                [~, scores_ind] = sort(aboxes_new{i}(:,5), 'descend');
                aboxes_new{i} = aboxes_new{i}(scores_ind, :);
            end
            
%             for j = 1:num_classes
%                 inds = find(~d.gt & scores(:, j) > thresh(j));
%                 if ~isempty(inds)
%                     [~, ord] = sort(scores(inds, j), 'descend');
%                     ord = ord(1:min(length(ord), max_per_image));
%                     inds = inds(ord);
%                     cls_boxes = boxes(inds, (1+(j-1)*4):((j)*4));
%                     cls_scores = scores(inds, j);
%                     aboxes{j}{i} = [aboxes{j}{i}; cat(2, single(cls_boxes), single(cls_scores))];
%                     box_inds{j}{i} = [box_inds{j}{i}; inds];
%                 else
%                     aboxes{j}{i} = [aboxes{j}{i}; zeros(0, 5, 'single')];
%                     box_inds{j}{i} = box_inds{j}{i};
%                 end
%             end

            fprintf(' time: %.3fs\n', toc(th));  

%             if mod(count, 1000) == 0
%                 for j = 1:num_classes
%                 [aboxes{j}, box_inds{j}, thresh(j)] = ...
%                     keep_top_k(aboxes{j}, box_inds{j}, i, max_per_set, thresh(j));
%                 end
%                 disp(thresh);
%             end    
        end
        save_file = fullfile(cache_dir, [imdb.classes{i} '_boxes_' imdb.name opts.suffix]);
        save(save_file, 'aboxes_old', 'aboxes_new');

%         for j = 1:num_classes
%             [aboxes{j}, box_inds{j}, thresh(j)] = ...
%                 keep_top_k(aboxes{j}, box_inds{j}, i, max_per_set, thresh(j));
%         end
%         disp(thresh);

        %for i = 1:num_images

%             top_scores{i} = sort(top_scores{i}, 'descend');  
%             if (length(top_scores{i}) > max_per_set)
%                 thresh(i) = top_scores{i}(max_per_set);
%             end
% 
%             % go back through and prune out detections below the found threshold
%             for j = 1:length(imdb.image_ids)
%                 if ~isempty(aboxes{i}{j})
%                     I = find(aboxes{i}{j}(:,end) < thresh(i));
%                     aboxes{i}{j}(I,:) = [];
%                     box_inds{i}{j}(I,:) = [];
%                 end
%             end
% 
%             save_file = fullfile(cache_dir, [imdb.classes{i} '_boxes_' imdb.name opts.suffix]);
%             boxes = aboxes{i};
%             inds = box_inds{i};
%             save(save_file, 'boxes', 'inds');
%             clear boxes inds;
        %end
        fprintf('test all images in %f seconds.\n', toc(t_start));
        
        caffe.reset_all(); 
        rng(prev_rng);
    end
% 0206 added
    start_thresh = 5; %5
    thresh_interval = 3;%3
    thresh_end = 500; % 500
    [gt_num_all, gt_recall_all, gt_num_pool, gt_recall_pool] = Get_Detector_Recall_finegrained(roidb, aboxes_old, start_thresh,thresh_interval,thresh_end);
    save(fullfile(cache_dir,'recall_vector_old.mat'),'gt_num_all', 'gt_recall_all', 'gt_num_pool', 'gt_recall_pool');
    fprintf('OLD all scales: gt recall rate = %d / %d = %.4f\n', gt_recall_all, gt_num_all, gt_recall_all/gt_num_all);
    
    [gt_num_all, gt_recall_all, gt_num_pool, gt_recall_pool] = Get_Detector_Recall_finegrained(roidb, aboxes_new, start_thresh,thresh_interval,thresh_end);
    save(fullfile(cache_dir,'recall_vector_old.mat'),'gt_num_all', 'gt_recall_all', 'gt_num_pool', 'gt_recall_pool');
    fprintf('NEW all scales: gt recall rate = %d / %d = %.4f\n', gt_recall_all, gt_num_all, gt_recall_all/gt_num_all);
    % ------------------------------------------------------------------------
    % Peform AP evaluation
    % ------------------------------------------------------------------------

%     if isequal(imdb.eval_func, @imdb_eval_voc)
%         for model_ind = 1:num_classes
%           cls = imdb.classes{model_ind};
%           res(model_ind) = imdb.eval_func(cls, aboxes{model_ind}, imdb, opts.cache_name, opts.suffix);
%         end
%     else
%     % ilsvrc
%         res = imdb.eval_func(aboxes, imdb, opts.cache_name, opts.suffix);
%     end
% 
%     if ~isempty(res)
%         fprintf('\n~~~~~~~~~~~~~~~~~~~~\n');
%         fprintf('Results:\n');
%         aps = [res(:).ap]' * 100;
%         disp(aps);
%         disp(mean(aps));
%         fprintf('~~~~~~~~~~~~~~~~~~~~\n');
%         mAP = mean(aps);
%     else
%         mAP = nan;
%     end
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
    for rois_num = 500:500:5000
        % generate pseudo testing data with max size
        im_blob = single(zeros(conf.max_size, conf.max_size, 3, 1));
        rois_blob = single(repmat([0; 0; 0; conf.max_size-1; conf.max_size-1], 1, rois_num));
        rois_blob = permute(rois_blob, [3, 4, 1, 2]);

        net_inputs = {im_blob, rois_blob};

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