function do_proposal_test_widerface_ablation_final_mpfvn(conf, model_stage, imdb, roidb, nms_option)
    % share the test with final3 for they have the same test network struct
    %[aboxes_res23, aboxes_res45]  = proposal_test_widerface_twopath_happy_flip(conf, imdb, ...
    %0129 added scale3 version
    [aboxes_s4, aboxes_s8, aboxes_s16]  = proposal_test_widerface_ablation_final_scale3(conf, imdb, ...
                                        'net_def_file',     model_stage.test_net_def_file, ...
                                        'net_file',         model_stage.output_model_file, ...
                                        'cache_name',       model_stage.cache_name, ...
                                        'suffix',           '_thr_60_60_60'); 
                               
    fprintf('Doing nms ... ');   
    % liu@1001: model_stage.nms.after_nms_topN functions as a threshold, indicating how many boxes will be preserved on average
    %ave_per_image_topN_s4 = model_stage.nms.after_nms_topN_s4;
    %ave_per_image_topN_s8 = model_stage.nms.after_nms_topN_s8;
    %ave_per_image_topN_s16 = model_stage.nms.after_nms_topN_s16;
    model_stage.nms.after_nms_topN_s4 = -1;
    model_stage.nms.after_nms_topN_s8 = -1;
    model_stage.nms.after_nms_topN_s16 = -1;
    aboxes_s4              = boxes_filter(aboxes_s4, model_stage.nms.per_nms_topN, model_stage.nms.nms_overlap_thres, model_stage.nms.after_nms_topN_s4, conf.use_gpu);
    aboxes_s8              = boxes_filter(aboxes_s8, model_stage.nms.per_nms_topN, model_stage.nms.nms_overlap_thres, model_stage.nms.after_nms_topN_s8, conf.use_gpu);
    aboxes_s16              = boxes_filter(aboxes_s16, model_stage.nms.per_nms_topN, model_stage.nms.nms_overlap_thres, model_stage.nms.after_nms_topN_s16, conf.use_gpu);
    fprintf(' Done.\n');  
    
    % only use the first max_sample_num images to compute an "expected" lower bound thresh
%     max_sample_num = 2000;
%     
%     % s4
%     sample_aboxes = aboxes_s4(randperm(length(aboxes_s4), min(length(aboxes_s4), max_sample_num)));  % conv4 and conv5 are the same, so just use conv4
%     scores = zeros(ave_per_image_topN_s4*length(sample_aboxes), 1);
%     for i = 1:length(sample_aboxes)
%         s_scores = sort([scores; sample_aboxes{i}(:, end)], 'descend');
%         scores = s_scores(1:ave_per_image_topN_s4*length(sample_aboxes));
%     end
%     score_thresh_s4 = scores(end);
%     
%     % s8
%     sample_aboxes = aboxes_s8(randperm(length(aboxes_s8), min(length(aboxes_s8), max_sample_num)));  % conv4 and conv5 are the same, so just use conv4
%     scores = zeros(ave_per_image_topN_s8*length(sample_aboxes), 1);
%     for i = 1:length(sample_aboxes)
%         s_scores = sort([scores; sample_aboxes{i}(:, end)], 'descend');
%         scores = s_scores(1:ave_per_image_topN_s8*length(sample_aboxes));
%     end
%     score_thresh_s8 = scores(end);
%     
%     % s16
%     sample_aboxes = aboxes_s16(randperm(length(aboxes_s16), min(length(aboxes_s16), max_sample_num)));  % conv4 and conv5 are the same, so just use conv4
%     scores = zeros(ave_per_image_topN_s16*length(sample_aboxes), 1);
%     for i = 1:length(sample_aboxes)
%         s_scores = sort([scores; sample_aboxes{i}(:, end)], 'descend');
%         scores = s_scores(1:ave_per_image_topN_s16*length(sample_aboxes));
%     end
%     score_thresh_s16 = scores(end);
% 
%     fprintf('score_threshold s4 = %f, s8 = %f , s16 = %f\n', score_thresh_s4, score_thresh_s8, score_thresh_s16);
    % drop the boxes which scores are lower than the threshold
    show_image = true;
    save_result = false;
    
    % 1122 added to save combined results of conv4 and conv5
    aboxes = cell(length(aboxes_s4), 1);  % conv4 and conv6 are also ok
    aboxes_nms = cell(length(aboxes_s4), 1);
    
    % 1121: add these 3 lines for drawing
    addpath(fullfile('external','export_fig'));
    res_dir = fullfile(pwd, 'output', conf.exp_name, 'rpn_cachedir','res_pic');
    mkdir_if_missing(res_dir);
    %1126 added to refresh figure
    close all;

    SUBMIT_cachedir = fullfile(pwd, 'output', conf.exp_name, 'submit_BP-FPN_cachedir');
    mkdir_if_missing(SUBMIT_cachedir);
    
    for i = 1:length(aboxes_s4)
        aboxes_s4{i} = aboxes_s4{i}(aboxes_s4{i}(:, end) > 0.99, :);  %0.99
        aboxes_s8{i} = aboxes_s8{i}(aboxes_s8{i}(:, end) > 0.99, :);%0.99
        aboxes_s16{i} = aboxes_s16{i}(aboxes_s16{i}(:, end) > 0.99, :);%0.99
        aboxes{i} = cat(1, aboxes_s4{i}, aboxes_s8{i}, aboxes_s16{i});
        
        if show_image
            img = imread(imdb.image_at(i));  
            %draw before NMS
            bbs_conv4 = aboxes_s4{i};
            bbs_conv5 = aboxes_s8{i};
            bbs_conv6 = aboxes_s16{i};
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
              bbApply('draw',bbs_conv6,'c');
            end
            hold off
        end
        
        %1006 added to do NPD-style nms
        time = tic;
        %0321: pseudoNMS_v8_ablation == pseudoNMS_v8_twopath
        aboxes_nms{i} = pseudoNMS_v8_ablation(aboxes{i}, nms_option);%4
        
        fprintf('PseudoNMS for image %d cost %.1f seconds\n', i, toc(time));
        
        if show_image    
            %1121 also draw gt boxes
            bbs_gt = roidb.rois(i).boxes;
            %0127 added if condition in case of empty gt boxes
            if ~isempty(bbs_gt)
                bbs_gt = max(bbs_gt, 1); % if any elements <=0, raise it to 1
                bbs_gt(:, 3) = bbs_gt(:, 3) - bbs_gt(:, 1) + 1;
                bbs_gt(:, 4) = bbs_gt(:, 4) - bbs_gt(:, 2) + 1;
                % if a box has only 1 pixel in either size, remove it
                invalid_idx = (bbs_gt(:, 3) <= 1) | (bbs_gt(:, 4) <= 1);
                bbs_gt(invalid_idx, :) = [];
            end
            img = imread(imdb.image_at(i)); 
            figure(2); 
            imshow(img);  %im(img)
            hold on
            bbs_all = aboxes_nms{i};
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
                strs = strsplit(imdb.image_at(i), filesep);
                saveName = sprintf('%s%cres_%s',res_dir, filesep, strs{end}(1:end-4));
                export_fig(saveName, '-png', '-a1', '-native');
                fprintf('image %d saved.\n', i);
            end
        end
    end
    
    aboxes_nms = boxes_filter(aboxes_nms, -1, 0.33, -1, conf.use_gpu); %0.5
    for i = 1:length(aboxes_s4)
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
        
        if 0      
            %1121 also draw gt boxes
            img = imread(imdb.image_at(i));  
            bbs_gt = roidb.rois(i).boxes;
            bbs_gt = max(bbs_gt, 1); % if any elements <=0, raise it to 1
            bbs_gt(:, 3) = bbs_gt(:, 3) - bbs_gt(:, 1) + 1;
            bbs_gt(:, 4) = bbs_gt(:, 4) - bbs_gt(:, 2) + 1;
            % if a box has only 1 pixel in either size, remove it
            invalid_idx = (bbs_gt(:, 3) <= 1) | (bbs_gt(:, 4) <= 1);
            bbs_gt(invalid_idx, :) = [];
            
            figure(6); 
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
            if 0
                strs = strsplit(imdb.image_at(i), '/');
                saveName = sprintf('%s%cres_%s',res_dir, filesep, strs{end}(1:end-4));
                export_fig(saveName, '-png', '-a1', '-native');
                fprintf('image %d saved.\n', i);
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

