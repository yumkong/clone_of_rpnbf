function roidb_new = do_proposal_test_my(conf, model_stage, imdb, roidb, nms_option)
    % get all pred bboxes
    aboxes                      = proposal_test_widerface(conf, imdb, ...
                                        'net_def_file',     model_stage.test_net_def_file, ...
                                        'net_file',         model_stage.output_model_file, ...
                                        'cache_name',       model_stage.cache_name);      
    % do nms
    %nms_option = 3; %nms level: 1, 2, 3
    % path to save file
    cache_dir = fullfile(pwd, 'output', conf.exp_name, 'rpn_cachedir', model_stage.cache_name, imdb.name);
    mkdir_if_missing(cache_dir);
    bbox_nms_save_name = fullfile(cache_dir, ['bbox_after_nms_rpn1_' imdb.name '.mat']);
    try 
        load(bbox_nms_save_name);  %aboxes_nms and thresholded 'aboxes' ==> to overwrite 'aboxes' which is too many boxes to evaluate
    catch
        fprintf('Doing nms ... ');   
        % liu@1001: model_stage.nms.after_nms_topN functions as a threshold, indicating how many boxes will be preserved on average
        ave_per_image_topN = model_stage.nms.after_nms_topN;
        model_stage.nms.after_nms_topN = -1;
        aboxes                      = boxes_filter(aboxes, model_stage.nms.per_nms_topN, model_stage.nms.nms_overlap_thres, model_stage.nms.after_nms_topN, conf.use_gpu);  
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
        show_image = false;

        %1007 tempararily use another cell to save bbox after nms
        aboxes_nms = cell(length(aboxes), 1);

        for i = 1:length(aboxes)
            aboxes{i} = aboxes{i}(aboxes{i}(:, end) > score_thresh, :);
            %1006 added to do NPD-style nms
            time = tic;
            aboxes_nms{i} = pseudoNMS_v3(aboxes{i}, nms_option);

            fprintf('PseudoNMS for image %d cost %.1f seconds\n', i, toc(time));
            % (optional) show image + bbox after NMS
            if show_image
                %draw boxes after 'smart' NMS
                bbs = aboxes_nms{i};
                if ~isempty(bbs)
                  bbs(:, 3) = bbs(:, 3) - bbs(:, 1) + 1;
                  bbs(:, 4) = bbs(:, 4) - bbs(:, 2) + 1;
                  %I=imread(imgNms{i});
                  figure(2); 
                  im(img);  %im(I)
                  bbApply('draw',bbs);
                end
            end
        end
        % save to mat file
        save(bbox_nms_save_name, 'aboxes_nms', 'aboxes');
    end
    
    % eval the gt recall
    gt_num = 0;
    gt_re_num = 0;
    %1007 added
    gt_num_nms = 0;
    gt_re_num_nms = 0;
    for i = 1:length(roidb.rois)
        %gts = roidb.rois(i).boxes(roidb.rois(i).ignores~=1, :);
        gts = roidb.rois(i).boxes; % for widerface, no ignored bboxes
        if ~isempty(gts)
            rois = aboxes{i}(:, 1:4);
            max_ols = max(boxoverlap(rois, gts));
            gt_num = gt_num + size(gts, 1);
            gt_re_num = gt_re_num + sum(max_ols >= 0.5);
            %1007 added
            if ~isempty(aboxes_nms{i})
                rois_nms = aboxes_nms{i}(:, 1:4);
                max_ols_nms = max(boxoverlap(rois_nms, gts));
                gt_num_nms = gt_num_nms + size(gts, 1);
                gt_re_num_nms = gt_re_num_nms + sum(max_ols_nms >= 0.5);
            end
        end
    end
    fprintf('gt recall rate = %.4f\n', gt_re_num / gt_num);
    fprintf('gt recall rate after nms-%d = %.4f\n', nms_option, gt_re_num_nms / gt_num_nms);
    
    %roidb_regions               = make_roidb_regions(aboxes, imdb.image_ids);  
    roidb_regions               = make_roidb_regions(aboxes_nms, imdb.image_ids);  
    
    roidb_new                   = roidb_from_proposal(imdb, roidb, roidb_regions, ...
                                        'keep_raw_proposal', false); % 1015: false --> true, to get more fgs
end

function aboxes = boxes_filter(aboxes, per_nms_topN, nms_overlap_thres, after_nms_topN, use_gpu)
    % to speed up nms
    if per_nms_topN > 0
        aboxes = cellfun(@(x) x(1:min(length(x), per_nms_topN), :), aboxes, 'UniformOutput', false);
    end
    % do nms
    if nms_overlap_thres > 0 && nms_overlap_thres < 1
        if use_gpu
            for i = 1:length(aboxes)
                aboxes{i} = aboxes{i}(nms(aboxes{i}, nms_overlap_thres, use_gpu), :);
            end 
        else
            parfor i = 1:length(aboxes)
                aboxes{i} = aboxes{i}(nms(aboxes{i}, nms_overlap_thres), :);
            end       
        end
    end
    aver_boxes_num = mean(cellfun(@(x) size(x, 1), aboxes, 'UniformOutput', true));
    fprintf('aver_boxes_num = %d, select top %d\n', round(aver_boxes_num), after_nms_topN);
    if after_nms_topN > 0
        aboxes = cellfun(@(x) x(1:min(length(x), after_nms_topN), :), aboxes, 'UniformOutput', false);
    end
end

function regions = make_roidb_regions(aboxes, images)
    regions.boxes = aboxes;
    regions.images = images;
end
