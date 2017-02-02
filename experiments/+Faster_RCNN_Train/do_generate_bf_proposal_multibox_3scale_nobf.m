function roidb_BF = do_generate_bf_proposal_multibox_3scale_nobf(conf, model_stage, imdb, roidb, is_test)
%function roidb_BF = do_generate_bf_proposal_multibox_ohem_happy_3scale(conf, model_stage, imdb, roidb, is_test, start_num)
   
    % **** 1201 ***** currently only do BF for conv4 
    % aboxes_conv4 is the raw bbox output of conv4
    %[aboxes_conv4, aboxes_conv5, aboxes_conv6]     = proposal_test_widerface_multibox_happy_flip_3scale(conf, imdb, ...
    [aboxes_conv4, aboxes_conv5, aboxes_conv6]     = proposal_test_widerface_multibox_happy_flip_3scale_new(conf, imdb, ...
                                        'net_def_file',     model_stage.test_net_def_file, ...
                                        'net_file',         model_stage.output_model_file, ...
                                        'cache_name',       model_stage.cache_name, ...
                                        'suffix',  '_thr_60_60_60', ...
                                        'start_num' , 1); %start_num
    
    fprintf('Doing nms ... ');   
    % liu@1001: model_stage.nms.after_nms_topN functions as a threshold, indicating how many boxes will be preserved on average
    ave_per_image_topN_conv4 = model_stage.nms.after_nms_topN_conv34; % conv4
    ave_per_image_topN_conv5 = model_stage.nms.after_nms_topN_conv5; % conv5
    ave_per_image_topN_conv6 = model_stage.nms.after_nms_topN_conv6; % conv6
    model_stage.nms.after_nms_topN_conv34 = -1;
    model_stage.nms.after_nms_topN_conv5 = -1;
    model_stage.nms.after_nms_topN_conv6 = -1;
    aboxes_conv4              = boxes_filter(aboxes_conv4, model_stage.nms.per_nms_topN, model_stage.nms.nms_overlap_thres_conv4, model_stage.nms.after_nms_topN_conv34, conf.use_gpu);
    aboxes_conv5              = boxes_filter(aboxes_conv5, model_stage.nms.per_nms_topN, model_stage.nms.nms_overlap_thres_conv5, model_stage.nms.after_nms_topN_conv5, conf.use_gpu);
    aboxes_conv6              = boxes_filter(aboxes_conv6, model_stage.nms.per_nms_topN, model_stage.nms.nms_overlap_thres_conv6, model_stage.nms.after_nms_topN_conv6, conf.use_gpu);
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
    
    % 1122 added to save combined results of conv4, conv5 and conv6
    aboxes = cell(length(aboxes_conv4), 1);  % conv5 and conv6 are also ok
    aboxes_nms = cell(length(aboxes_conv4), 1);
    nms_option = 3;
    % eval the gt recall
    gt_num = 0;
    gt_re_num_5 = 0;
    gt_re_num_7 = 0;
    gt_re_num_8 = 0;
    gt_re_num_9 = 0;
    for i = 1:length(roidb.rois)
        %aboxes{i} = cat(1, aboxes_conv4{i}, aboxes_conv5{i}, aboxes_conv6{i});
        aboxes{i} = cat(1, aboxes_conv4{i}(aboxes_conv4{i}(:, end) > 0.65, :),...
                           aboxes_conv5{i}(aboxes_conv5{i}(:, end) > 0.7, :),...
                           aboxes_conv6{i}(aboxes_conv6{i}(:, end) > 0.7, :));
                       
        if 1
            img = imread(imdb.image_at(i));  
            %draw before NMS
            bbs_conv4 = aboxes_conv4{i};
            bbs_conv5 = aboxes_conv5{i};
            bbs_conv6 = aboxes_conv6{i};
            figure(1); 
            imshow(img);  %im(img)
            hold on
            if ~isempty(bbs_conv4)
              bbs_conv4(:, 3) = bbs_conv4(:, 3) - bbs_conv4(:, 1) + 1;
              bbs_conv4(:, 4) = bbs_conv4(:, 4) - bbs_conv4(:, 2) + 1;
              bbApply('draw',bbs_conv4,'g');
            end
            if ~isempty(bbs_conv5)
              bbs_conv5(:, 3) = bbs_conv5(:, 3) - bbs_conv5(:, 1) + 1;
              bbs_conv5(:, 4) = bbs_conv5(:, 4) - bbs_conv5(:, 2) + 1;
              bbApply('draw',bbs_conv5,'c');
            end
            if ~isempty(bbs_conv6)
              bbs_conv6(:, 3) = bbs_conv6(:, 3) - bbs_conv6(:, 1) + 1;
              bbs_conv6(:, 4) = bbs_conv6(:, 4) - bbs_conv6(:, 2) + 1;
              bbApply('draw',bbs_conv6,'m');
            end
            hold off
        end
        aboxes_nms{i} = pseudoNMS_v8(aboxes{i}, nms_option);
        fprintf('PseudoNms for image %d / %d\n', i, length(roidb.rois));
        if 1      
            %1121 also draw gt boxes
            bbs_gt = roidb.rois(i).boxes;
            bbs_gt = max(bbs_gt, 1); % if any elements <=0, raise it to 1
            bbs_gt(:, 3) = bbs_gt(:, 3) - bbs_gt(:, 1) + 1;
            bbs_gt(:, 4) = bbs_gt(:, 4) - bbs_gt(:, 2) + 1;
            % if a box has only 1 pixel in either size, remove it
            invalid_idx = (bbs_gt(:, 3) <= 1) | (bbs_gt(:, 4) <= 1);
            bbs_gt(invalid_idx, :) = [];
            
            bbs_all = aboxes_nms{i};
            figure(2); 
            imshow(img);  %im(img)
            hold on

            if ~isempty(bbs_all)
                  bbs_all(:, 3) = bbs_all(:, 3) - bbs_all(:, 1) + 1;
                  bbs_all(:, 4) = bbs_all(:, 4) - bbs_all(:, 2) + 1;
                  bbApply('draw',bbs_all,'g');
            end
            if ~isempty(bbs_gt)
              bbApply('draw',bbs_gt,'r');
            end
            hold off
        end
        
        %gts = roidb.rois(i).boxes(roidb.rois(i).ignores~=1, :);
        gts = roidb.rois(i).boxes;
        if ~isempty(gts)
            gt_num = gt_num + size(gts, 1);
            if ~isempty(aboxes{i})
                rois = aboxes{i}(:, 1:4);
                max_ols = max(boxoverlap(rois, gts));
                gt_re_num_5 = gt_re_num_5 + sum(max_ols >= 0.5);
                gt_re_num_7 = gt_re_num_7 + sum(max_ols >= 0.7);
                gt_re_num_8 = gt_re_num_8 + sum(max_ols >= 0.8);
                gt_re_num_9 = gt_re_num_9 + sum(max_ols >= 0.9);
            end
        end
    end
    fprintf('gt recall rate (ol >0.5) = %.4f\n', gt_re_num_5 / gt_num);
    fprintf('gt recall rate (ol >0.7) = %.4f\n', gt_re_num_7 / gt_num);
    fprintf('gt recall rate (ol >0.8) = %.4f\n', gt_re_num_8 / gt_num);
    fprintf('gt recall rate (ol >0.9) = %.4f\n', gt_re_num_9 / gt_num);

    aboxes_nms = boxes_filter(aboxes_nms, -1, 0.33, -1, conf.use_gpu); %0.5
    
    show_image = false;
    save_result = false;
    SUBMIT_cachedir = fullfile(pwd, 'output', conf.exp_name, 'submit_mprpn_cachedir');
    mkdir_if_missing(SUBMIT_cachedir);
    %0201 added
    for i = 1:length(aboxes_conv4)
        % draw boxes after 'naive' thresholding
        sstr = strsplit(imdb.image_ids{i}, filesep);
        event_name = sstr{1};
        event_dir = fullfile(SUBMIT_cachedir, event_name);
        mkdir_if_missing(event_dir);
        fid = fopen(fullfile(event_dir, [sstr{2} '.txt']), 'w');
        fprintf(fid, '%s\n', [imdb.image_ids{i} '.jpg']);
        bbs_all = aboxes_nms{i};
        
        fprintf(fid, '%d\n', size(bbs_all, 1));
        if ~isempty(bbs_all)
            for j = 1:size(bbs_all,1)
                %each row: [x1 y1 w h score]
                fprintf(fid, '%d %d %d %d %f\n', round([bbs_all(j,1) bbs_all(j,2) bbs_all(j,3)-bbs_all(j,1)+1 bbs_all(j,4)-bbs_all(j,2)+1]), bbs_all(j, 5));
            end
        end
        fclose(fid);
        fprintf('Done with saving image %d bboxes.\n', i);
        
        if show_image      
            %1121 also draw gt boxes
            img = imread(imdb.image_at(i));  
            bbs_gt = roidb.rois(i).boxes;
            bbs_gt = max(bbs_gt, 1); % if any elements <=0, raise it to 1
            bbs_gt(:, 3) = bbs_gt(:, 3) - bbs_gt(:, 1) + 1;
            bbs_gt(:, 4) = bbs_gt(:, 4) - bbs_gt(:, 2) + 1;
            % if a box has only 1 pixel in either size, remove it
            invalid_idx = (bbs_gt(:, 3) <= 1) | (bbs_gt(:, 4) <= 1);
            bbs_gt(invalid_idx, :) = [];
            
            figure(3); 
            imshow(img);  %im(img)
            hold on

            if ~isempty(bbs_all)
                  bbs_all(:, 3) = bbs_all(:, 3) - bbs_all(:, 1) + 1;
                  bbs_all(:, 4) = bbs_all(:, 4) - bbs_all(:, 2) + 1;
                  bbApply('draw',bbs_all,'g');
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
end

function aboxes = boxes_filter(aboxes, per_nms_topN, nms_overlap_thres, after_nms_topN, use_gpu)
    % to speed up nms
    % liu@1001: get the first per_nms_topN bboxes
    % it is no good to get per_nms_topN samples from each image since some
    % images have more hard samples which should be sampled more while others contain very 
    % easy samples which should be sampled less
    if per_nms_topN > 0
        aboxes = cellfun(@(x) x(1:min(size(x, 1), per_nms_topN), :), aboxes, 'UniformOutput', false);
    end
    % do nms
    if nms_overlap_thres > 0 && nms_overlap_thres < 1  
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
    if after_nms_topN > 0
        aboxes = cellfun(@(x) x(1:min(size(x, 1), after_nms_topN), :), aboxes, 'UniformOutput', false);
    end
end

