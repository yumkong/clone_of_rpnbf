function aboxes = do_proposal_test_widerface_conv3(conf, model_stage, imdb, roidb, cache_name, method_name, nms_option)
    aboxes                      = proposal_test_widerface_conv3(conf, imdb, ...
                                        'net_def_file',     model_stage.test_net_def_file, ...
                                        'net_file',         model_stage.output_model_file, ...
                                        'cache_name',       model_stage.cache_name); 
                               
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
        aboxes{i} = aboxes{i}(aboxes{i}(:, end) >= 0.7, :);  %score_thresh_conv4
    end

    % 0103 added
    thr1 = 32;
    thr2 = 100;
    thr3 = 300;
    thr4 = 500;
    %thr5 = 900;
    fprintf('For det-conv3:\n');
    Get_Detector_Recall(roidb, aboxes, thr1, thr2, thr3, thr4);
  
end

function Get_Detector_Recall(roidb, aboxes, thr1, thr2, thr3, thr4)
    gt_num = 0;
    gt_recall_num = 0;
    % 1229 added
    gt_num_det1 = 0;  % 6 ~ 32
    gt_recall_num_det1 = 0;  
    gt_num_det2 = 0;  % 33 ~ 100
    gt_recall_num_det2 = 0;
    gt_num_det3 = 0;  % 101 ~ 300
    gt_recall_num_det3 = 0;
    gt_num_det4 = 0;  % 301 ~ 500
    gt_recall_num_det4 = 0;  
    gt_num_det5 = 0;  % 501 ~ inf
    gt_recall_num_det5 = 0;

    %0110 added
    gt_num_detall_total = 0;
    gt_num_det1_total = 0;
    gt_num_det2_total = 0;
    gt_num_det3_total = 0;
    gt_num_det4_total = 0;
    gt_num_det5_total = 0;
    for i = 1:length(roidb.rois)
        gts = roidb.rois(i).boxes; % for widerface, no ignored bboxes
        face_height = gts(:,4) - gts(:,2) + 1;
        idx_all = (face_height>= 6);  % all: 6-inf
        idx_det1 = (face_height>= 6) & (face_height <= thr1); % 6-32
        idx_det2 = (face_height> thr1) & (face_height <= thr2);%33-100
        idx_det3 = (face_height> thr2) & (face_height <= thr3);%101-300
        idx_det4 = (face_height> thr3) & (face_height <= thr4);%301-500
        idx_det5 = (face_height> thr4); %500- inf
        gts_all = gts(idx_all, :);
        gts_det1 = gts(idx_det1, :);
        gts_det2 = gts(idx_det2, :);
        gts_det3 = gts(idx_det3, :);
        gts_det4 = gts(idx_det4, :);
        gts_det5 = gts(idx_det5, :);
        
        rois = aboxes{i}(:, 1:4);
        if ~isempty(gts_all)
            max_ols = max(boxoverlap(rois, gts_all));
            gt_num = gt_num + size(gts_all, 1);
            if ~isempty(max_ols)
                gt_recall_num = gt_recall_num + sum(max_ols >= 0.5);
                %0110 added
                gt_num_detall_total = gt_num_detall_total + size(rois, 1);
            end
        end
        if ~isempty(gts_det1)
            max_ols = max(boxoverlap(rois, gts_det1));
            gt_num_det1 = gt_num_det1 + size(gts_det1, 1);
            if ~isempty(max_ols)
                gt_recall_num_det1 = gt_recall_num_det1 + sum(max_ols >= 0.5);
                %0110 added
                tmp_idx = (rois(:,4) - rois(:,2)) >=6 & (rois(:,4) - rois(:,2)) <= thr1;
                gt_num_det1_total = gt_num_det1_total + sum(tmp_idx);
            end
        end
        if ~isempty(gts_det2)
            max_ols = max(boxoverlap(rois, gts_det2));
            gt_num_det2 = gt_num_det2 + size(gts_det2, 1);
            if ~isempty(max_ols)
                gt_recall_num_det2 = gt_recall_num_det2 + sum(max_ols >= 0.5);
                %0110 added
                tmp_idx = (rois(:,4) - rois(:,2)) > thr1 & (rois(:,4) - rois(:,2)) <= thr2;
                gt_num_det2_total = gt_num_det2_total + sum(tmp_idx);
            end
        end
        if ~isempty(gts_det3)
            max_ols = max(boxoverlap(rois, gts_det3));
            gt_num_det3 = gt_num_det3 + size(gts_det3, 1);
            if ~isempty(max_ols)
                gt_recall_num_det3 = gt_recall_num_det3 + sum(max_ols >= 0.5);
                %0110 added
                tmp_idx = (rois(:,4) - rois(:,2)) > thr2 & (rois(:,4) - rois(:,2)) <= thr3;
                gt_num_det3_total = gt_num_det3_total + sum(tmp_idx);
            end
        end
        if ~isempty(gts_det4)
            max_ols = max(boxoverlap(rois, gts_det4));
            gt_num_det4 = gt_num_det4 + size(gts_det4, 1);
            if ~isempty(max_ols)
                gt_recall_num_det4 = gt_recall_num_det4 + sum(max_ols >= 0.5);
                %0110 added
                tmp_idx = (rois(:,4) - rois(:,2)) > thr3 & (rois(:,4) - rois(:,2)) <= thr4;
                gt_num_det4_total = gt_num_det4_total + sum(tmp_idx);
            end
        end
        if ~isempty(gts_det5)
            max_ols = max(boxoverlap(rois, gts_det5));
            gt_num_det5 = gt_num_det5 + size(gts_det5, 1);
            if ~isempty(max_ols)
                gt_recall_num_det5 = gt_recall_num_det5 + sum(max_ols >= 0.5);
                %0110 added
                tmp_idx = (rois(:,4) - rois(:,2)) > thr4;
                gt_num_det5_total = gt_num_det5_total + sum(tmp_idx);
            end
        end
    end
    fprintf('All scales: gt recall num = %d, gt_num = %d, total num = %d\n', gt_recall_num, gt_num, gt_num_detall_total);
    fprintf('6-32: gt recall num = %d, gt_num = %d, total num = %d\n', gt_recall_num_det1, gt_num_det1, gt_num_det1_total);
    fprintf('33-100: gt recall num = %d, gt_num = %d, total num = %d\n', gt_recall_num_det2, gt_num_det2, gt_num_det2_total);
    fprintf('101-300: gt recall num = %d, gt_num = %d, total num = %d\n', gt_recall_num_det3, gt_num_det3, gt_num_det3_total);
    fprintf('301-500: gt recall num = %d, gt_num = %d, total num = %d\n', gt_recall_num_det4, gt_num_det4, gt_num_det4_total);
    fprintf('501-inf: gt recall num = %d, gt_num = %d, total num = %d\n', gt_recall_num_det5, gt_num_det5, gt_num_det5_total);
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

