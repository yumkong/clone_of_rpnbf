%originally from do_proposal_test_widerface_multibox_ohem_happy_flip
function do_proposal_test_MPRPN(conf, model_stage, imdb, roidb, cache_name, method_name, nms_option)
    % share the test with final3 for they have the same test network struct
    [aboxes_conv4, aboxes_conv5, aboxes_conv6]     = proposal_test_MPRPN(conf, imdb, ...
                                        'net_def_file',     model_stage.test_net_def_file, ...
                                        'net_file',         model_stage.output_model_file, ...
                                        'cache_name',       model_stage.cache_name, ...
                                        'suffix',           '_thr10percent'); 
                               
    fprintf('Doing nms ... ');   
    % liu@1001: model_stage.nms.after_nms_topN functions as a threshold, indicating how many boxes will be preserved on average
    ave_per_image_topN_conv4 = model_stage.nms.after_nms_topN_conv34; % conv4
    ave_per_image_topN_conv5 = model_stage.nms.after_nms_topN_conv5; % conv5
    ave_per_image_topN_conv6 = model_stage.nms.after_nms_topN_conv6; % conv6
    model_stage.nms.after_nms_topN_conv34 = -1;
    model_stage.nms.after_nms_topN_conv5 = -1;
    model_stage.nms.after_nms_topN_conv6 = -1;
    aboxes_conv4              = boxes_filter(aboxes_conv4, model_stage.nms.per_nms_topN, model_stage.nms.nms_overlap_thres, model_stage.nms.after_nms_topN_conv34, conf.use_gpu);
    aboxes_conv5              = boxes_filter(aboxes_conv5, model_stage.nms.per_nms_topN, model_stage.nms.nms_overlap_thres, model_stage.nms.after_nms_topN_conv5, conf.use_gpu);
    aboxes_conv6              = boxes_filter(aboxes_conv6, model_stage.nms.per_nms_topN, model_stage.nms.nms_overlap_thres, model_stage.nms.after_nms_topN_conv6, conf.use_gpu);
    fprintf(' Done.\n');  
    
    % only use the first max_sample_num images to compute an "expected" lower bound thresh
    max_sample_num = 5000;
    
    % conv4
    sample_aboxes = aboxes_conv4(randperm(length(aboxes_conv4), min(length(aboxes_conv4), max_sample_num)));  % conv4 and conv5 are the same, so just use conv4
    scores = zeros(ave_per_image_topN_conv4*length(sample_aboxes), 1);
    for i = 1:length(sample_aboxes)
        s_scores = sort([scores; sample_aboxes{i}(:, end)], 'descend');
        scores = s_scores(1:ave_per_image_topN_conv4*length(sample_aboxes));
    end
    score_thresh_conv4 = scores(end);
    
    % conv5
    sample_aboxes = aboxes_conv5(randperm(length(aboxes_conv5), min(length(aboxes_conv5), max_sample_num)));  % conv4 and conv5 are the same, so just use conv4
    scores = zeros(ave_per_image_topN_conv5*length(sample_aboxes), 1);
    for i = 1:length(sample_aboxes)
        s_scores = sort([scores; sample_aboxes{i}(:, end)], 'descend');
        scores = s_scores(1:ave_per_image_topN_conv5*length(sample_aboxes));
    end
    score_thresh_conv5 = scores(end);
    
    % conv6
    sample_aboxes = aboxes_conv6(randperm(length(aboxes_conv6), min(length(aboxes_conv6), max_sample_num)));  % conv4 and conv5 are the same, so just use conv4
    scores = zeros(ave_per_image_topN_conv6*length(sample_aboxes), 1);
    for i = 1:length(sample_aboxes)
        s_scores = sort([scores; sample_aboxes{i}(:, end)], 'descend');
        scores = s_scores(1:ave_per_image_topN_conv6*length(sample_aboxes));
    end
    score_thresh_conv6 = scores(end);
    
    fprintf('score_threshold conv4 = %f, conv5 = %f, conv6 = %f\n', score_thresh_conv4, score_thresh_conv5, score_thresh_conv6);
    % drop the boxes which scores are lower than the threshold
%     show_image = true;
%     save_result = true;
    % path to save file
    cache_dir = fullfile(pwd, 'output', conf.exp_name, 'rpn_cachedir', cache_name, method_name);
    mkdir_if_missing(cache_dir);
    
    %1007 tempararily use another cell to save bbox after nms
    
    
    % 1121: add these 3 lines for drawing
    addpath(fullfile('external','export_fig'));
    res_dir = fullfile(pwd, 'output', conf.exp_name, 'rpn_cachedir','res_pic');
    mkdir_if_missing(res_dir);
    %1126 added to refresh figure
    close all;
    aboxes = cell(length(aboxes_conv5), 1);  % conv4 and conv6 are also ok
    for i = 1:length(aboxes_conv4)
        
        %aboxes_nms{i} = cat(1, aboxes_conv4{i}(aboxes_conv4{i}(:, end) > score_thresh_conv4, :),...
        %                       aboxes_conv5{i}(aboxes_conv5{i}(:, end) > score_thresh_conv5, :));
        aboxes_conv4{i} = aboxes_conv4{i}(aboxes_conv4{i}(:, end) > score_thresh_conv4, :);
        aboxes_conv5{i} = aboxes_conv5{i}(aboxes_conv5{i}(:, end) > score_thresh_conv5, :);
        aboxes_conv6{i} = aboxes_conv6{i}(aboxes_conv6{i}(:, end) > score_thresh_conv6, :);
        aboxes{i} = cat(1, aboxes_conv4{i}, aboxes_conv5{i}, aboxes_conv6{i});
    end

    % 0103 added
    thr1 = 32;
    thr2 = 450;
	fprintf('For det-4:\n');
    Get_Detector_Recall(roidb, aboxes_conv4, thr1, thr2);
    fprintf('For det-16:\n');
    Get_Detector_Recall(roidb, aboxes_conv5, thr1, thr2);
    fprintf('For det-32:\n');
    Get_Detector_Recall(roidb, aboxes_conv6, thr1, thr2);
    fprintf('For det-all:\n');
    Get_Detector_Recall(roidb, aboxes, thr1, thr2);
    
% eval the gt recall
%     gt_num = 0;
%     gt_recall_num = 0;
%     % 1229 added
%     gt_num_det1 = 0;  % 8 ~ 32
%     gt_recall_num_det1 = 0;  
%     gt_num_det2 = 0;  % 33 ~ 360
%     gt_recall_num_det2 = 0;
%     gt_num_det3 = 0;  % 361 ~ 900
%     gt_recall_num_det3 = 0;
%     for i = 1:length(roidb.rois)
%         gts = roidb.rois(i).boxes; % for widerface, no ignored bboxes
%         face_height = gts(:,4) - gts(:,2) + 1;
%         idx_all = (face_height>= 8) & (face_height <= 900);
%         idx_det1 = (face_height>= 8) & (face_height <= 32);
%         idx_det2 = (face_height> 32) & (face_height <= 360);
%         idx_det3 = (face_height> 360) & (face_height <= 900);
%         gts_all = gts(idx_all, :);
%         gts_det1 = gts(idx_det1, :);
%         gts_det2 = gts(idx_det2, :);
%         gts_det3 = gts(idx_det3, :);
%         
%         rois = aboxes{i}(:, 1:4);
%         if ~isempty(gts_all)
%             max_ols = max(boxoverlap(rois, gts_all));
%             gt_num = gt_num + size(gts_all, 1);
%             if ~isempty(max_ols)
%                 gt_recall_num = gt_recall_num + sum(max_ols >= 0.5);
%             end
%         end
%         if ~isempty(gts_det1)
%             max_ols = max(boxoverlap(rois, gts_det1));
%             gt_num_det1 = gt_num_det1 + size(gts_det1, 1);
%             if ~isempty(max_ols)
%                 gt_recall_num_det1 = gt_recall_num_det1 + sum(max_ols >= 0.5);
%             end
%         end
%         if ~isempty(gts_det2)
%             max_ols = max(boxoverlap(rois, gts_det2));
%             gt_num_det2 = gt_num_det2 + size(gts_det2, 1);
%             if ~isempty(max_ols)
%                 gt_recall_num_det2 = gt_recall_num_det2 + sum(max_ols >= 0.5);
%             end
%         end
%         if ~isempty(gts_det3)
%             max_ols = max(boxoverlap(rois, gts_det3));
%             gt_num_det3 = gt_num_det3 + size(gts_det3, 1);
%             if ~isempty(max_ols)
%                 gt_recall_num_det3 = gt_recall_num_det3 + sum(max_ols >= 0.5);
%             end
%         end
%     end
%     fprintf('For det-4:\n');
%     fprintf('All scales: gt recall rate = %.4f\n', gt_recall_num / gt_num);
%     fprintf('8-32: gt recall rate = %.4f\n', gt_recall_num_det1 / gt_num_det1);
%     fprintf('33-360: gt recall rate = %.4f\n', gt_recall_num_det2 / gt_num_det2);
%     fprintf('361-900: gt recall rate = %.4f\n', gt_recall_num_det3 / gt_num_det3);
end

function Get_Detector_Recall(roidb, aboxes, thr1, thr2)
    gt_num = 0;
    gt_recall_num = 0;
    % 1229 added
    gt_num_det1 = 0;  % 8 ~ 32
    gt_recall_num_det1 = 0;  
    gt_num_det2 = 0;  % 33 ~ 360
    gt_recall_num_det2 = 0;
    gt_num_det3 = 0;  % 361 ~ 900
    gt_recall_num_det3 = 0;
    for i = 1:length(roidb.rois)
        gts = roidb.rois(i).boxes; % for widerface, no ignored bboxes
        face_height = gts(:,4) - gts(:,2) + 1;
        idx_all = (face_height>= 8) & (face_height <= 900);
        idx_det1 = (face_height>= 8) & (face_height <= thr1);
        idx_det2 = (face_height> thr1) & (face_height <= thr2);%360-->500
        idx_det3 = (face_height> thr2) & (face_height <= 900); %360 --> 500
        gts_all = gts(idx_all, :);
        gts_det1 = gts(idx_det1, :);
        gts_det2 = gts(idx_det2, :);
        gts_det3 = gts(idx_det3, :);
        
        rois = aboxes{i}(:, 1:4);
        if ~isempty(gts_all)
            max_ols = max(boxoverlap(rois, gts_all));
            gt_num = gt_num + size(gts_all, 1);
            if ~isempty(max_ols)
                gt_recall_num = gt_recall_num + sum(max_ols >= 0.5);
            end
        end
        if ~isempty(gts_det1)
            max_ols = max(boxoverlap(rois, gts_det1));
            gt_num_det1 = gt_num_det1 + size(gts_det1, 1);
            if ~isempty(max_ols)
                gt_recall_num_det1 = gt_recall_num_det1 + sum(max_ols >= 0.5);
            end
        end
        if ~isempty(gts_det2)
            max_ols = max(boxoverlap(rois, gts_det2));
            gt_num_det2 = gt_num_det2 + size(gts_det2, 1);
            if ~isempty(max_ols)
                gt_recall_num_det2 = gt_recall_num_det2 + sum(max_ols >= 0.5);
            end
        end
        if ~isempty(gts_det3)
            max_ols = max(boxoverlap(rois, gts_det3));
            gt_num_det3 = gt_num_det3 + size(gts_det3, 1);
            if ~isempty(max_ols)
                gt_recall_num_det3 = gt_recall_num_det3 + sum(max_ols >= 0.5);
            end
        end
    end
    fprintf('All scales: gt recall rate = %.4f\n', gt_recall_num / gt_num);
    fprintf('8-32: gt recall rate = %.4f\n', gt_recall_num_det1 / gt_num_det1);
    fprintf('33-360: gt recall rate = %.4f\n', gt_recall_num_det2 / gt_num_det2);
    fprintf('361-900: gt recall rate = %.4f\n', gt_recall_num_det3 / gt_num_det3);
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

