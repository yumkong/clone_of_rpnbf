function do_proposal_test_widerface_ablation_final(conf, model_stage, imdb, roidb, nms_option)
    % share the test with final3 for they have the same test network struct
    [aboxes_conv4, aboxes_conv5, aboxes_conv6]     = proposal_test_widerface_ablation_final(conf, imdb, ...
                                        'net_def_file',     model_stage.test_net_def_file, ...
                                        'net_file',         model_stage.output_model_file, ...
                                        'cache_name',       model_stage.cache_name, ...
                                        'suffix',           '_thr_10_10_10'); 
    cache_dir = fullfile(pwd, 'output', conf.exp_name, 'rpn_cachedir', model_stage.cache_name, imdb.name);
	%cache_dir = fullfile(pwd, 'output', opts.cache_name, imdb.name);
    try
        % try to load cache
        box_nms_name = fullfile(cache_dir, ['proposal_boxes_afterNMS_' imdb.name]);
        ld = load(box_nms_name);
        aboxes_s4 = ld.aboxes_s4;
        aboxes_s8 = ld.aboxes_s8;
        aboxes_s16 = ld.aboxes_s16;
    catch 
        fprintf('Doing nms ... ');   
        % liu@1001: model_stage.nms.after_nms_topN functions as a threshold, indicating how many boxes will be preserved on average
        ave_per_image_topN_conv4 = model_stage.nms.after_nms_topN; % conv4
        ave_per_image_topN_conv5 = model_stage.nms.after_nms_topN; % conv5
        ave_per_image_topN_conv6 = model_stage.nms.after_nms_topN; % conv6
        model_stage.nms.after_nms_topN_conv34 = -1;
        model_stage.nms.after_nms_topN_conv5 = -1;
        model_stage.nms.after_nms_topN_conv6 = -1;
        aboxes_conv4              = boxes_filter(aboxes_conv4, model_stage.nms.per_nms_topN, model_stage.nms.nms_overlap_thres, model_stage.nms.after_nms_topN, conf.use_gpu);
        aboxes_conv5              = boxes_filter(aboxes_conv5, model_stage.nms.per_nms_topN, model_stage.nms.nms_overlap_thres, model_stage.nms.after_nms_topN, conf.use_gpu);
        aboxes_conv6              = boxes_filter(aboxes_conv6, model_stage.nms.per_nms_topN, model_stage.nms.nms_overlap_thres, model_stage.nms.after_nms_topN, conf.use_gpu);
        fprintf(' Done.\n');  

        % only use the first max_sample_num images to compute an "expected" lower bound thresh
        max_sample_num = 3000;

        score_thresh_conv4 = 0;
        score_thresh_conv5 = 0;
        score_thresh_conv6 = 0;
    %     % conv4
    %     sample_aboxes = aboxes_conv4(randperm(length(aboxes_conv4), min(length(aboxes_conv4), max_sample_num)));  % conv4 and conv5 are the same, so just use conv4
    %     scores = zeros(ave_per_image_topN_conv4*length(sample_aboxes), 1);
    %     for i = 1:length(sample_aboxes)
    %         s_scores = sort([scores; sample_aboxes{i}(:, end)], 'descend');
    %         scores = s_scores(1:ave_per_image_topN_conv4*length(sample_aboxes));
    %     end
    %     score_thresh_conv4 = scores(end);
    %     
    %     % conv5
    %     sample_aboxes = aboxes_conv5(randperm(length(aboxes_conv5), min(length(aboxes_conv5), max_sample_num)));  % conv4 and conv5 are the same, so just use conv4
    %     scores = zeros(ave_per_image_topN_conv5*length(sample_aboxes), 1);
    %     for i = 1:length(sample_aboxes)
    %         s_scores = sort([scores; sample_aboxes{i}(:, end)], 'descend');
    %         scores = s_scores(1:ave_per_image_topN_conv5*length(sample_aboxes));
    %     end
    %     score_thresh_conv5 = scores(end);
    %     
    %     % conv6
    %     sample_aboxes = aboxes_conv6(randperm(length(aboxes_conv6), min(length(aboxes_conv6), max_sample_num)));  % conv4 and conv5 are the same, so just use conv4
    %     scores = zeros(ave_per_image_topN_conv6*length(sample_aboxes), 1);
    %     for i = 1:length(sample_aboxes)
    %         s_scores = sort([scores; sample_aboxes{i}(:, end)], 'descend');
    %         scores = s_scores(1:ave_per_image_topN_conv6*length(sample_aboxes));
    %     end
    %     score_thresh_conv6 = scores(end);

        fprintf('score_threshold s4 = %f, s8 = %f, s16 = %f\n', score_thresh_conv4, score_thresh_conv5, score_thresh_conv6);

        %1007 tempararily use another cell to save bbox after nms
        aboxes_s4 = cell(length(aboxes_conv4), 1);
        aboxes_s8 = cell(length(aboxes_conv5), 1);
        aboxes_s16 = cell(length(aboxes_conv6), 1);

        %0103 added
        aboxes = cell(length(aboxes_conv5), 1);  % conv4 and conv6 are also ok
        aboxes_nms = cell(length(aboxes_conv5), 1);
        for i = 1:length(aboxes_conv4)

            %aboxes_nms{i} = cat(1, aboxes_conv4{i}(aboxes_conv4{i}(:, end) > score_thresh_conv4, :),...
            %                       aboxes_conv5{i}(aboxes_conv5{i}(:, end) > score_thresh_conv5, :));
            aboxes_conv4{i} = aboxes_conv4{i}(aboxes_conv4{i}(:, end) > score_thresh_conv4, :);
            aboxes_conv5{i} = aboxes_conv5{i}(aboxes_conv5{i}(:, end) > score_thresh_conv5, :);
            aboxes_conv6{i} = aboxes_conv6{i}(aboxes_conv6{i}(:, end) > score_thresh_conv6, :);
            aboxes{i} = cat(1, aboxes_conv4{i}, aboxes_conv5{i}, aboxes_conv6{i});

            %1006 added to do NPD-style nms
            time = tic;
            aboxes_s4{i} = pseudoNMS_v8_twopath(aboxes_conv4{i}, nms_option);
            aboxes_s8{i} = pseudoNMS_v8_twopath(aboxes_conv5{i}, nms_option);
            aboxes_s16{i} = pseudoNMS_v8_twopath(aboxes_conv6{i}, nms_option);
            aboxes_nms{i} = pseudoNMS_v8_twopath(aboxes{i}, nms_option);
            fprintf('PseudoNMS for image %d cost %.1f seconds\n', i, toc(time));
            %0226 added: sort by score
            if ~isempty(aboxes_s4{i})
                [~, scores_ind] = sort(aboxes_s4{i}(:,5), 'descend');
                aboxes_s4{i} = aboxes_s4{i}(scores_ind, :);
            end
            if ~isempty(aboxes_s8{i})
                [~, scores_ind] = sort(aboxes_s8{i}(:,5), 'descend');
                aboxes_s8{i} = aboxes_s8{i}(scores_ind, :);
            end
            if ~isempty(aboxes_s16{i})
                [~, scores_ind] = sort(aboxes_s16{i}(:,5), 'descend');
                aboxes_s16{i} = aboxes_s16{i}(scores_ind, :);
            end
        end
        
        save(box_nms_name, 'aboxes_s4', 'aboxes_s8', 'aboxes_s16');
    end
    % 0206 added
    start_thresh = 5; %5
    thresh_interval = 3;%3
    thresh_end = 500; % 500
    [gt_num_all, gt_recall_all, gt_num_pool, gt_recall_pool] = Get_Detector_Recall_finegrained(roidb, aboxes_s4,aboxes_s8,aboxes_s16, start_thresh,thresh_interval,thresh_end);
    save(fullfile(cache_dir,'recall_vector_fine.mat'),'gt_num_all', 'gt_recall_all', 'gt_num_pool', 'gt_recall_pool');
    fprintf('For all scales: gt recall rate = %d / %d = %.4f\n', gt_recall_all, gt_num_all, gt_recall_all/gt_num_all);
end

function [gt_num_all, gt_recall_all, gt_num_pool, gt_recall_pool] = Get_Detector_Recall_finegrained(roidb, aboxes_s4,aboxes_s8,aboxes_s16, start_thresh,thresh_interval,thresh_end)
    gt_num_all = 0;
    gt_recall_all = 0;
    % 1229 added
    gt_num_pool = zeros(1, length(start_thresh:thresh_interval:thresh_end)+1);  % 4 ~ 14, 14~24, ...
    gt_recall_pool = gt_num_pool;

    %0110 added
    for i = 1:length(roidb.rois)
        gts = roidb.rois(i).boxes; % for widerface, no ignored bboxes
        if ~isempty(gts)
%             % only leave 2x gt_num detected bboxes
%             gt_num = size(gts, 1);
%             %rois = aboxes{i}(:, 1:4);
%             this_pred_num = min(2*gt_num, size(aboxes{i}, 1));
%             if this_pred_num ~= 0
%                 rois = aboxes{i}(1:this_pred_num, 1:4);
%             else
%                 rois = [];
%             end
            
            face_height = gts(:,4) - gts(:,2) + 1;
            idx_s4 = (face_height>= start_thresh) & (face_height< 11);
            gt_num_s4 = sum(idx_s4);
            this_pred_num = min(2*gt_num_s4, size(aboxes_s4{i}, 1));
            if this_pred_num ~= 0
                rois_s4 = aboxes_s4{i}(1:this_pred_num, 1:4);
            else
                rois_s4 = [];
            end
            idx_s8 = (face_height>= 11) & (face_height< 128);%65
            gt_num_s8 = sum(idx_s8);
            this_pred_num = min(2*gt_num_s8, size(aboxes_s8{i}, 1));
            if this_pred_num ~= 0
                rois_s8 = aboxes_s8{i}(1:this_pred_num, 1:4);
            else
                rois_s8 = [];
            end
            idx_s16 = (face_height>= 128); %65
            gt_num_s16 = sum(idx_s16);
            this_pred_num = min(2*gt_num_s16, size(aboxes_s16{i}, 1));
            if this_pred_num ~= 0
                rois_s16 = aboxes_s16{i}(1:this_pred_num, 1:4);
            else
                rois_s16 = [];
            end
            %0226 rois_all
            rois = cat(1, rois_s4, rois_s8, rois_s16);
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
            % stride 4
            for k = start_thresh:thresh_interval:11-1
                cnt = cnt + 1;
                part_idx = (face_height>= k) & (face_height < k + thresh_interval); % eg.:4~14
                part_gts = gts(part_idx, :);
            
                if ~isempty(part_gts)
                    gt_num_pool(cnt) = gt_num_pool(cnt) + size(part_gts, 1);
                    if ~isempty(rois_s4)
                        max_ols = max(boxoverlap(rois_s4, part_gts));  
                        if ~isempty(max_ols)
                            gt_recall_pool(cnt) = gt_recall_pool(cnt) + sum(max_ols >= 0.5);
                        end
                    end
                end
            end
            %0226: stride 8 65-->128
            for k = 11:thresh_interval:128-1
                cnt = cnt + 1;
                part_idx = (face_height>= k) & (face_height < k + thresh_interval); % eg.:4~14
                part_gts = gts(part_idx, :);
            
                if ~isempty(part_gts)
                    gt_num_pool(cnt) = gt_num_pool(cnt) + size(part_gts, 1);
                    if ~isempty(rois_s8)
                        max_ols = max(boxoverlap(rois_s8, part_gts));  
                        if ~isempty(max_ols)
                            gt_recall_pool(cnt) = gt_recall_pool(cnt) + sum(max_ols >= 0.5);
                        end
                    end
                end
            end
            %0226: stride 16 65-->128
            for k = 128:thresh_interval:thresh_end
                cnt = cnt + 1;
                part_idx = (face_height>= k) & (face_height < k + thresh_interval); % eg.:4~14
                part_gts = gts(part_idx, :);
            
                if ~isempty(part_gts)
                    gt_num_pool(cnt) = gt_num_pool(cnt) + size(part_gts, 1);
                    if ~isempty(rois_s16)
                        max_ols = max(boxoverlap(rois_s16, part_gts));  
                        if ~isempty(max_ols)
                            gt_recall_pool(cnt) = gt_recall_pool(cnt) + sum(max_ols >= 0.5);
                        end
                    end
                end
            end
            %0226 for 300-inf
            cnt = cnt + 1;
            part_idx = (face_height>= k + thresh_interval); % 300~inf
            part_gts = gts(part_idx, :);

            if ~isempty(part_gts)
                gt_num_pool(cnt) = gt_num_pool(cnt) + size(part_gts, 1);
                if ~isempty(rois_s16)
                    max_ols = max(boxoverlap(rois_s16, part_gts));
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

