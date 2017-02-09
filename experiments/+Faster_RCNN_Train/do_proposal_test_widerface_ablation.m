function aboxes = do_proposal_test_widerface_ablation(conf, model_stage, imdb, roidb, nms_option)
    aboxes                      = proposal_test_widerface_ablation(conf, imdb, ...
                                        'net_def_file',     model_stage.test_net_def_file, ...
                                        'net_file',         model_stage.output_model_file, ...
                                        'cache_name',       model_stage.cache_name, ...
                                        'three_scales',     false); %0124: root switch of test with 1 scale or 3 scales
    cache_dir = fullfile(pwd, 'output', conf.exp_name, 'rpn_cachedir', model_stage.cache_name, imdb.name);
	%cache_dir = fullfile(pwd, 'output', opts.cache_name, imdb.name);
    try
        % try to load cache
        box_nms_name = fullfile(cache_dir, ['proposal_boxes_afterNMS_' imdb.name]);
        ld = load(box_nms_name);
        aboxes = ld.aboxes;
    catch 
        fprintf('Doing nms ... ');   
        % liu@1001: model_stage.nms.after_nms_topN functions as a threshold, indicating how many boxes will be preserved on average
        ave_per_image_topN = model_stage.nms.after_nms_topN;
        model_stage.nms.after_nms_topN = -1;
        aboxes                      = boxes_filter(aboxes, model_stage.nms.per_nms_topN, model_stage.nms.nms_overlap_thres, model_stage.nms.after_nms_topN, conf.use_gpu);      
        fprintf(' Done.\n');  

        % only use the first max_sample_num images to compute an "expected" lower bound thresh
        max_sample_num = 5000;
        sample_aboxes = aboxes(randperm(length(aboxes), min(length(aboxes), max_sample_num)));
        scores = zeros(ave_per_image_topN*length(sample_aboxes), 1);
        for i = 1:length(sample_aboxes)
            s_scores = sort([scores; sample_aboxes{i}(:, end)], 'descend');
            scores = s_scores(1:ave_per_image_topN*length(sample_aboxes));
        end
        score_thresh = scores(end);
        fprintf('score_threshold = %f\n', score_thresh);
        % drop the boxes which scores are lower than the threshold
        for i = 1:length(aboxes)
            fprintf('Doing pseudo nms for image %d\n', i);   
            % 0206: keep an average of the first 300 boxes
            aboxes{i} = aboxes{i}(aboxes{i}(:, end) >= score_thresh, :);
            % 0206: psudoNms
            aboxes{i} = pseudoNMS_v8_twopath(aboxes{i}, nms_option);%4
        end
        save(box_nms_name, 'aboxes');
    end

    % 0206 added
    start_thresh = 5;
    thresh_interval = 10;
    thresh_end = 300;
    [gt_num_all, gt_recall_all, gt_num_pool, gt_recall_pool] = Get_Detector_Recall_finegrained(roidb, aboxes, start_thresh,thresh_interval,thresh_end);
    save(fullfile(cache_dir,'recall_vector.mat'),'gt_num_all', 'gt_recall_all', 'gt_num_pool', 'gt_recall_pool');
    fprintf('For all scales: gt recall rate = %d / %d = %.4f\n', gt_recall_all, gt_num_all, gt_recall_all/gt_num_all);
    h = figure(2);
    h = sfigure(h, 2.5, 2);
    rotation_plot(gt_recall_pool./(gt_num_pool+eps), start_thresh, thresh_interval, thresh_end);
    %save plot here
    plotSaveName = fullfile(cache_dir,'recall_plot');
    export_fig(plotSaveName, '-png', '-a1', '-native');
end

function [gt_num_all, gt_recall_all, gt_num_pool, gt_recall_pool] = Get_Detector_Recall_finegrained(roidb, aboxes, start_thresh,thresh_interval,thresh_end)
    gt_num_all = 0;
    gt_recall_all = 0;
    % 1229 added
    gt_num_pool = zeros(1, length(start_thresh:thresh_interval:thresh_end)+1);  % 4 ~ 14, 14~24, ...
    gt_recall_pool = gt_num_pool;

    %0110 added
    for i = 1:length(roidb.rois)
        gts = roidb.rois(i).boxes; % for widerface, no ignored bboxes
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
            if (~isempty(gts_all)) && (~isempty(rois))
                max_ols = max(boxoverlap(rois, gts_all));
                gt_num_all = gt_num_all + size(gts_all, 1);
                if ~isempty(max_ols)
                    gt_recall_all = gt_recall_all + sum(max_ols >= 0.5);
                end
            end
                
            % segment statistics
            cnt = 0;
            for k = start_thresh:thresh_interval:thresh_end
                cnt = cnt + 1;
                part_idx = (face_height>= k) & (face_height < k + thresh_interval); % eg.:4~14
                part_gts = gts(part_idx, :);
            
                if (~isempty(part_gts)) && (~isempty(rois))
                    max_ols = max(boxoverlap(rois, part_gts));
                    gt_num_pool(cnt) = gt_num_pool(cnt) + size(part_gts, 1);
                    if ~isempty(max_ols)
                        gt_recall_pool(cnt) = gt_recall_pool(cnt) + sum(max_ols >= 0.5);
                    end
                end
            end
            %0206 for 300-inf
            cnt = cnt + 1;
            part_idx = (face_height>= k + thresh_interval); % 300~inf
            part_gts = gts(part_idx, :);

            if (~isempty(part_gts)) && (~isempty(rois))
                max_ols = max(boxoverlap(rois, part_gts));
                gt_num_pool(cnt) = gt_num_pool(cnt) + size(part_gts, 1);
                if ~isempty(max_ols)
                    gt_recall_pool(cnt) = gt_recall_pool(cnt) + sum(max_ols >= 0.5);
                end
            end
        end
    end
    %save('recall_conv2.mat','gt_num_all', 'gt_recall_all', 'gt_num_pool', 'gt_recall_pool');
    %fprintf('For all scales: gt recall rate = %d / %d = %.4f\n', gt_recall_all, gt_num_all, gt_recall_all/gt_num_all);
end

function aboxes = boxes_filter(aboxes, per_nms_topN, nms_overlap_thres, after_nms_topN, use_gpu)
    % to speed up nms
    % liu@1001: get the first per_nms_topN bboxes
    if per_nms_topN > 0
        aboxes = cellfun(@(x) x(1:min(size(x, 1), per_nms_topN), :), aboxes, 'UniformOutput', false);
    end
    % do nms
    if nms_overlap_thres > 0 && nms_overlap_thres < 1
        if 0
            for i = 1:length(aboxes)
                tic_toc_print('weighted ave nms: %d / %d \n', i, length(aboxes));
                aboxes{i} = get_keep_boxes(aboxes{i}, 0, nms_overlap_thres, 0.7);
            end 
        else
            if use_gpu
                for i = 1:length(aboxes)
                    tic_toc_print('nms: %d / %d \n', i, length(aboxes));
                    aboxes{i} = aboxes{i}(nms(aboxes{i}, nms_overlap_thres, use_gpu), :);
                end
            else
                parfor i = 1:length(aboxes)
                    aboxes{i} = aboxes{i}(nms(aboxes{i}, nms_overlap_thres), :);
                end
            end
        end
    end
    aver_boxes_num = mean(cellfun(@(x) size(x, 1), aboxes, 'UniformOutput', true));
    fprintf('aver_boxes_num = %d, select top %d\n', round(aver_boxes_num), after_nms_topN);
    if after_nms_topN > 0
        aboxes = cellfun(@(x) x(1:min(size(x, 1), after_nms_topN), :), aboxes, 'UniformOutput', false);
    end
end

