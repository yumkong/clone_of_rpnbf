function aboxes = do_proposal_test_widerface_conv3plus4(conf, model_stage, imdb, roidb, cache_name, method_name, nms_option)
    aboxes                      = proposal_test_widerface_conv3plus4(conf, imdb, ...
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
    show_image = true;
    save_result = true;
    % path to save file
    cache_dir = fullfile(pwd, 'output', conf.exp_name, 'rpn_cachedir', cache_name, method_name);
    mkdir_if_missing(cache_dir);
    
    %1007 tempararily use another cell to save bbox after nms
    aboxes_nms = cell(length(aboxes), 1);
    %nms_option = 3; %1, 2, 3
    %aboxes_nms2 = cell(length(aboxes), 1);
    %nms_option2 = 2; %1, 2, 3
    
    % 1121: add these 3 lines for drawing
    addpath(fullfile('external','export_fig'));
    res_dir = fullfile(pwd, 'output', conf.exp_name, 'rpn_cachedir','res_pic');
    mkdir_if_missing(res_dir);
	%1126 added to refresh figure
    close all;
    
    for i = 1:length(aboxes)
        
        aboxes{i} = aboxes{i}(aboxes{i}(:, end) > score_thresh, :);
        
        % draw boxes after 'naive' thresholding
        if show_image
            img = imread(imdb.image_at(i));  
            %draw before NMS
            bbs = aboxes{i};
            if ~isempty(bbs)
              bbs(:, 3) = bbs(:, 3) - bbs(:, 1) + 1;
              bbs(:, 4) = bbs(:, 4) - bbs(:, 2) + 1;
              %I=imread(imgNms{i});
              figure(1); 
              imshow(img);  %im(img)
              bbApply('draw',bbs);
            end
        end
        
        %1006 added to do NPD-style nms
        time = tic;
        aboxes_nms{i} = pseudoNMS_v6(aboxes{i}, nms_option);
        
        fprintf('PseudoNMS for image %d cost %.1f seconds\n', i, toc(time));
        if show_image
            %draw boxes after 'smart' NMS
            bbs = aboxes_nms{i};
            
            %1121 also draw gt boxes
            bbs_gt = roidb.rois(i).boxes;
            bbs_gt = max(bbs_gt, 1); % if any elements <=0, raise it to 1
            bbs_gt(:, 3) = bbs_gt(:, 3) - bbs_gt(:, 1) + 1;
            bbs_gt(:, 4) = bbs_gt(:, 4) - bbs_gt(:, 2) + 1;
            % if a box has only 1 pixel in either size, remove it
            invalid_idx = (bbs_gt(:, 3) <= 1) | (bbs_gt(:, 4) <= 1);
            bbs_gt(invalid_idx, :) = [];
            
            figure(2); 
            imshow(img);  %im(img)
            hold on
            if ~isempty(bbs)
              bbs(:, 3) = bbs(:, 3) - bbs(:, 1) + 1;
              bbs(:, 4) = bbs(:, 4) - bbs(:, 2) + 1;
              %I=imread(imgNms{i});
              bbApply('draw',bbs, 'g');
            end
            if ~isempty(bbs_gt)
              bbApply('draw',bbs_gt,'r');
            end
            hold off
            % 1121: save result
            if save_result
                strs = strsplit(imdb.image_at(i), '/');
                saveName = sprintf('%s/res_%s',res_dir, strs{end}(1:end-4));
                export_fig(saveName, '-png', '-a1', '-native');
                fprintf('image %d saved.\n', i);
            end
        end
    end
    
    % save bbox before nms
    bbox_save_name = fullfile(cache_dir, sprintf('VGG16_all-ave-%d.txt', ave_per_image_topN));
    save_bbox_to_txt(aboxes, imdb.image_ids, bbox_save_name);
    % save bbox after nms
    bbox_save_name = fullfile(cache_dir, sprintf('VGG16_all-ave-%d-nms-op%d.txt', ave_per_image_topN, nms_option));
    save_bbox_to_txt(aboxes_nms, imdb.image_ids, bbox_save_name);
	
    % eval the gt recall
    gt_num = 0;
    gt_re_num = 0;
    %1007 added
    gt_num_nms = 0;
    gt_re_num_nms = 0;
    for i = 1:length(roidb.rois)
        %gts = roidb.rois(i).boxes(roidb.rois(i).ignores~=1, :);
        %if i == 245
        %disp(i);
        %end
        gts = roidb.rois(i).boxes; % for widerface, no ignored bboxes
        if ~isempty(gts)
            rois = aboxes{i}(:, 1:4);
            max_ols = max(boxoverlap(rois, gts));
            gt_num = gt_num + size(gts, 1);
            if ~isempty(max_ols)
                gt_re_num = gt_re_num + sum(max_ols >= 0.5);
            end
            %1007 added
            if ~isempty(aboxes_nms{i})
                rois_nms = aboxes_nms{i}(:, 1:4);
                max_ols_nms = max(boxoverlap(rois_nms, gts));
                gt_num_nms = gt_num_nms + size(gts, 1);
                if ~isempty(max_ols_nms)
                    gt_re_num_nms = gt_re_num_nms + sum(max_ols_nms >= 0.5);
                end
            end
        end
    end
    fprintf('gt recall rate = %.4f\n', gt_re_num / gt_num);
    % 1007 added
    fprintf('gt recall rate after nms-%d = %.4f\n', nms_option, gt_re_num_nms / gt_num_nms);
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

