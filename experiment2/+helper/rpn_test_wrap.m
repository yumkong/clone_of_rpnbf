function aboxes = rpn_test_wrap(conf, model_stage, imdb, roidb, nms_option)
    cache_dir = fullfile(pwd, 'output2', conf.exp_name, 'rpn_cache', model_stage.cache_name, imdb.name);
    aboxes                      = helper.rpn_test(conf, imdb, cache_dir, ...
                                        'net_def_file',     model_stage.test_net_def_file, ...
                                        'net_file',         model_stage.output_model_file, ...
                                        'cache_name',       model_stage.cache_name, ...
                                        'three_scales',     false); %0124: root switch of test with 1 scale or 3 scales
    
	%cache_dir = fullfile(pwd, 'output', opts.cache_name, imdb.name);
    try
        % try to load cache
        box_nms_name = fullfile(cache_dir, ['proposal_boxes_pseudonms_' imdb.name]);
        ld = load(box_nms_name);
        aboxes = ld.aboxes;
    catch 
        fprintf('Doing nms ... ');   
        % liu@1001: model_stage.nms.after_nms_topN functions as a threshold, indicating how many boxes will be preserved on average
        ave_per_image_topN = model_stage.nms.after_nms_topN;
        model_stage.nms.after_nms_topN = -1;
        aboxes = boxes_filter(aboxes, model_stage.nms.per_nms_topN, model_stage.nms.nms_overlap_thres, model_stage.nms.after_nms_topN, conf.use_gpu);      
        fprintf('Done.\n');  

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
            aboxes{i} = helper.nms_pseudo(aboxes{i}, nms_option);%4
            %0226 added: sort by score ()pseudoNMS may make the box
            %not sorted in descending order of scores
            if ~isempty(aboxes{i})
                [~, scores_ind] = sort(aboxes{i}(:,5), 'descend');
                aboxes{i} = aboxes{i}(scores_ind, :);
            end
            
        end
        save(box_nms_name, 'aboxes');
    end
    %%%%% 0714 show some results! %%%%%
    show_result = false;
    if show_result
        for i = 1:length(aboxes)
            im = imread(imdb.image_at(i));
            figure(1),clf;
            imshow(im);
            keep = helper.nms(aboxes{i}, 0.3);
            bbs_show = aboxes{i}(keep, :);
            if ~isempty(bbs_show)
                bbs_show = bbs_show(bbs_show(:,5) >= 0.8, :);
                bbs_show(:, 3) = bbs_show(:, 3) - bbs_show(:, 1) + 1;
                bbs_show(:, 4) = bbs_show(:, 4) - bbs_show(:, 2) + 1;
                bbApply('draw',bbs_show,'g');
            end
        end
    end
    %%%%%%%%%%

    % 0714 added [8 12]
    thresh_beg = 8;
    thresh_end = 12;
    [gt_num, recall_num] = get_detector_recall(roidb, aboxes, thresh_beg, thresh_end);
    fprintf('gt = %d, recall = %d, recall_rate = %.4f\n', gt_num, recall_num, recall_num/gt_num);
end

function [gt_num, recall_num] = get_detector_recall(roidb, aboxes, thresh_beg, thresh_end)
    gt_num = 0;
    recall_num = 0;
    %0713 changed
    %for i = 1:length(roidb.rois)
    for i = 1:length(aboxes)
        gts = roidb.rois(i).boxes; % for widerface, no ignored bboxes
        if ~isempty(gts)
            face_height = gts(:,4) - gts(:,2) + 1;
            % total statistics
            idx_all = (face_height >= thresh_beg & face_height <= thresh_end);  % all:4-inf
            gts_all = gts(idx_all, :);
            
            % only leave 2x gt_num (in range [8 12]) detected bboxes
            gt_num_this = size(gts_all, 1);
            %rois = aboxes{i}(:, 1:4);
            this_pred_num = min(2*gt_num_this, size(aboxes{i}, 1));
            if this_pred_num ~= 0
                rois = aboxes{i}(1:this_pred_num, 1:4);
            else
                rois = [];
            end
            
            if ~isempty(gts_all)
                gt_num = gt_num + size(gts_all, 1);
                if ~isempty(rois)
                    max_ols = max(helper.boxoverlap(rois, gts_all));
                    if ~isempty(max_ols)
                        recall_num = recall_num + sum(max_ols >= 0.5);
                    end
                end
            end
        end
    end
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
                    helper.tic_toc_print('nms: %d / %d \n', i, length(aboxes));
                    aboxes{i} = aboxes{i}(helper.nms(aboxes{i}, nms_overlap_thres, use_gpu), :);
                end
            else
                parfor i = 1:length(aboxes)
                    aboxes{i} = aboxes{i}(helper.nms(aboxes{i}, nms_overlap_thres), :);
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

