function do_proposal_test_widerface_ablation_total_2345_cxt(conf,conf_fast_rcnn, model_stage,model_stage_fast, imdb, roidb, nms_option)
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
    
    % drop the boxes which scores are lower than the threshold
    show_image = true;
    save_result = false;
    
    fopts.net_def_file = model_stage_fast.test_net_def_file;
    fopts.net_file = model_stage_fast.output_model_file;
    fopts.cache_name = model_stage_fast.cache_name;
    fopts.exp_name = conf_fast_rcnn.exp_name;
    fopts.suffix = '';
    cache_dir = fullfile(pwd, 'output', fopts.exp_name, 'fast_rcnn_cachedir', fopts.cache_name, imdb.name); 
    mkdir_if_missing(cache_dir);
    %  init log
    timestamp = datestr(datevec(now()), 'yyyymmdd_HHMMSS');
    mkdir_if_missing(fullfile(cache_dir, 'log'));
    log_file = fullfile(cache_dir, 'log', ['test_', timestamp, '.txt']);
    diary(log_file);
    
    num_images = length(imdb.image_ids);
    save_file = fullfile(cache_dir, ['aboxes_' imdb.name fopts.suffix]);
    
    % 1121: add these 3 lines for drawing
    addpath(fullfile('external','export_fig'));
    res_dir = fullfile(pwd, 'output', conf_fast_rcnn.exp_name, 'fast_rcnn_cachedir','res_pic');
    mkdir_if_missing(res_dir);
    %1126 added to refresh figure
    close all;
    
    SUBMIT_cachedir1 = fullfile(pwd, 'output', conf_fast_rcnn.exp_name, 'fast_rcnn_cachedir', 'submit_MPFVN_cachedir1');
    mkdir_if_missing(SUBMIT_cachedir1);
    SUBMIT_cachedir2 = fullfile(pwd, 'output', conf_fast_rcnn.exp_name, 'fast_rcnn_cachedir', 'submit_MPFVN_cachedir2');
    mkdir_if_missing(SUBMIT_cachedir2);
    SUBMIT_cachedir3 = fullfile(pwd, 'output', conf_fast_rcnn.exp_name, 'fast_rcnn_cachedir', 'submit_MPFVN_cachedir3');
    mkdir_if_missing(SUBMIT_cachedir3);
    SUBMIT_cachedir4 = fullfile(pwd, 'output', conf_fast_rcnn.exp_name, 'fast_rcnn_cachedir', 'submit_MPFVN_cachedir4');
    mkdir_if_missing(SUBMIT_cachedir4);
    SUBMIT_cachedir5 = fullfile(pwd, 'output', conf_fast_rcnn.exp_name, 'fast_rcnn_cachedir', 'submit_MPFVN_cachedir5');
    mkdir_if_missing(SUBMIT_cachedir5);
    try
        ld = load(save_file);%'aboxes_old', 'aboxes_new','score_ind_old', 'score_ind_new'
        aboxes_old = ld.aboxes_old;
        aboxes_new = ld.aboxes_new;
    catch    
%%      testing 
        % init caffe net
        caffe_log_file_base = fullfile(cache_dir, 'caffe_log');
        caffe.init_log(caffe_log_file_base);
        caffe_net = caffe.Net(fopts.net_def_file, 'test');
        caffe_net.copy_from(fopts.net_file);

        % set random seed
        prev_rng = seed_rand(conf_fast_rcnn.rng_seed);
        caffe.set_random_seed(conf_fast_rcnn.rng_seed);

        % set gpu/cpu
        if conf_fast_rcnn.use_gpu
            caffe.set_mode_gpu();
        else
            caffe.set_mode_cpu();
        end             

        % determine the maximum number of rois in testing 
        %max_rois_num_in_gpu = check_gpu_memory(conf, caffe_net);
        max_rois_num_in_gpu = 1000;

        disp('opts:');
        disp(fopts);
        disp('conf:');
        disp(conf_fast_rcnn);
        
        aboxes_old = cell(length(imdb.image_ids), 1);
        aboxes_new = cell(length(imdb.image_ids), 1);

        count = 0;
        t_start = tic;
        for i = 1:length(aboxes_s4)
            count = count + 1;
            fprintf('%s: test (%s) %d/%d ', procid(), imdb.name, count, num_images);
            th = tic;
            im = imread(imdb.image_at(i));
            % 0402 pad to 32x
            target_size = size(im);
            new_target_size = ceil(target_size / 32) * 32;
            botpad = new_target_size(1) - target_size(1);
            rightpad = new_target_size(2) - target_size(2);
            im = imPad(im , [0 botpad 0 rightpad], 0);
        
            aboxes_s4{i} = aboxes_s4{i}(aboxes_s4{i}(:, end) > 0.95, :);  %0.99
            aboxes_s8{i} = aboxes_s8{i}(aboxes_s8{i}(:, end) > 0.95, :);%0.99
            aboxes_s16{i} = aboxes_s16{i}(aboxes_s16{i}(:, end) > 0.95, :);%0.99
            aboxes_old{i} = cat(1, aboxes_s4{i}, aboxes_s8{i}, aboxes_s16{i});
            if ~isempty(aboxes_old)
                rpn_boxes = aboxes_old{i}(:, 1:4);
                rpn_score = aboxes_old{i}(:, 5);
                [~, scores_ind] = sort(rpn_score, 'descend');
                rpn_boxes = rpn_boxes(scores_ind, :);
                rpn_score = rpn_score(scores_ind, :);
            end
            % 0326 added: avoid gpu out of memory 
            % max_rois_num_in_gpu = 1000
            if size(rpn_boxes, 1) > max_rois_num_in_gpu
                rpn_boxes = rpn_boxes(1:max_rois_num_in_gpu, :);
                rpn_score = rpn_score(1:max_rois_num_in_gpu, :);
            end
            %0404 changed from mpfvn to conv2345
            %fastrcnn_score = fast_rcnn_im_detect_widerface_total_conv2345_nocxt
            fastrcnn_score = fast_rcnn_im_detect_widerface_total_conv2345_cxt(conf_fast_rcnn, caffe_net, im, rpn_boxes, max_rois_num_in_gpu);
			fastrcnn_score_pno = fastrcnn_score;
            
            if ~isempty(rpn_boxes)
                aboxes_old{i} = [rpn_boxes rpn_score];
                aboxes_new{i} = [rpn_boxes fastrcnn_score_pno];
            else
                aboxes_old{i} = [];
                aboxes_new{i} = [];
            end

            fprintf('costs %.1f seconds\n', toc(th));
        end
        %0331: only save these two file can reproduce all the results
        save(save_file, 'aboxes_old', 'aboxes_new');
        fprintf('test all images in %f seconds.\n', toc(t_start));
        
        caffe.reset_all(); 
        rng(prev_rng);
    end
    
    aboxes_v1 = cell(length(imdb.image_ids), 1);
    aboxes_v2 = cell(length(imdb.image_ids), 1);
    aboxes_v3 = cell(length(imdb.image_ids), 1);
    aboxes_v4 = cell(length(imdb.image_ids), 1);
    aboxes_v5 = cell(length(imdb.image_ids), 1);
    
    aa = cell2mat(aboxes_new);
    all_f_score = aa(:,5);
    max_all_f = max(all_f_score);
    min_all_f = min(all_f_score);
    count = 0;
    for i = 1:length(aboxes_s4)
        count = count + 1;
        fprintf('%s: test (%s) %d/%d \n', procid(), imdb.name, count, num_images);
        if ~isempty(aboxes_old{i})
            rpn_boxes = aboxes_old{i}(:, 1:4);
            rpn_score = aboxes_old{i}(:, 5);
            fastrcnn_score_raw = aboxes_new{i}(:, 5);
            %0328: cubic root - optimal by round 2
            fastrcnn_score = nthroot(fastrcnn_score_raw, 3);
            %fastrcnn_score = nthroot(fastrcnn_score_raw, 1);
            %fastrcnn_score = nthroot(fastrcnn_score_raw, 1);
            %0328 shrink to [0.8 1] - optimal by round 1
            fastrcnn_score1 = (fastrcnn_score - min_all_f)/(max_all_f - min_all_f)*0.1 + 0.9;
            fastrcnn_score2 = (fastrcnn_score - min_all_f)/(max_all_f - min_all_f)*0.2 + 0.8;
            fastrcnn_score3 = (fastrcnn_score - min_all_f)/(max_all_f - min_all_f)*0.3 + 0.7;
            fastrcnn_score4 = (fastrcnn_score - min_all_f)/(max_all_f - min_all_f)*0.4 + 0.6;
            fastrcnn_score5 = (fastrcnn_score - min_all_f)/(max_all_f - min_all_f)*0.5 + 0.5;
        end
        
        if ~isempty(rpn_boxes)
            aboxes_old{i} = [rpn_boxes rpn_score];
            aboxes_new{i} = [rpn_boxes fastrcnn_score];
            % 1 * rpn + 0.5 * fastrcnn_score is optimal by round 3&4
            aboxes_v1{i} = [rpn_boxes (rpn_score + 0.1 * fastrcnn_score1)];
            aboxes_v2{i} = [rpn_boxes (rpn_score + 0.1 * fastrcnn_score2)];
            aboxes_v3{i} = [rpn_boxes (rpn_score + 0.1 * fastrcnn_score3)];
            aboxes_v4{i} = [rpn_boxes (rpn_score + 0.1 * fastrcnn_score4)];
            aboxes_v5{i} = [rpn_boxes (rpn_score + 0.1 * fastrcnn_score5)];
        else
            aboxes_old{i} = [];
            aboxes_new{i} = [];
            aboxes_v1{i} = [];
            aboxes_v2{i} = [];
            aboxes_v3{i} = [];
            aboxes_v4{i} = [];
            aboxes_v5{i} = [];
        end
        %0321: pseudoNMS_v8_ablation == pseudoNMS_v8_twopath
        aboxes_v1{i} = pseudoNMS_v8_ablation(aboxes_v1{i}, nms_option);%4
        aboxes_v2{i} = pseudoNMS_v8_ablation(aboxes_v2{i}, nms_option);%4
        aboxes_v3{i} = pseudoNMS_v8_ablation(aboxes_v3{i}, nms_option);%4
        aboxes_v4{i} = pseudoNMS_v8_ablation(aboxes_v4{i}, nms_option);%4
        aboxes_v5{i} = pseudoNMS_v8_ablation(aboxes_v5{i}, nms_option);%4
        if ~isempty(aboxes_v1{i})
            [~, scores_ind] = sort(aboxes_v1{i}(:,5), 'descend');
            aboxes_v1{i} = aboxes_v1{i}(scores_ind, :);
        end
        if ~isempty(aboxes_v2{i})
            [~, scores_ind] = sort(aboxes_v2{i}(:,5), 'descend');
            aboxes_v2{i} = aboxes_v2{i}(scores_ind, :);
        end
        if ~isempty(aboxes_v3{i})
            [~, scores_ind] = sort(aboxes_v3{i}(:,5), 'descend');
            aboxes_v3{i} = aboxes_v3{i}(scores_ind, :);
        end
        if ~isempty(aboxes_v4{i})
            [~, scores_ind] = sort(aboxes_v4{i}(:,5), 'descend');
            aboxes_v4{i} = aboxes_v4{i}(scores_ind, :);
        end
        if ~isempty(aboxes_v5{i})
            [~, scores_ind] = sort(aboxes_v5{i}(:,5), 'descend');
            aboxes_v5{i} = aboxes_v5{i}(scores_ind, :);
        end
    end
    aboxes_v1 = boxes_filter(aboxes_v1, -1, 0.33, -1, conf.use_gpu); %0.5
    aboxes_v2 = boxes_filter(aboxes_v2, -1, 0.33, -1, conf.use_gpu); %0.5
    aboxes_v3 = boxes_filter(aboxes_v3, -1, 0.33, -1, conf.use_gpu); %0.5
    aboxes_v4 = boxes_filter(aboxes_v4, -1, 0.33, -1, conf.use_gpu); %0.5
    aboxes_v5 = boxes_filter(aboxes_v5, -1, 0.33, -1, conf.use_gpu); %0.5
    for i = 1:length(aboxes_s4)
        % draw boxes after 'naive' thresholding
        sstr = strsplit(imdb.image_ids{i}, filesep);
        event_name = sstr{1};
        event_dir1 = fullfile(SUBMIT_cachedir1, event_name);
        mkdir_if_missing(event_dir1);
        fid1 = fopen(fullfile(event_dir1, [sstr{2} '.txt']), 'w');
        fprintf(fid1, '%s\n', [imdb.image_ids{i} '.jpg']);
        
        event_dir2 = fullfile(SUBMIT_cachedir2, event_name);
        mkdir_if_missing(event_dir2);
        fid2 = fopen(fullfile(event_dir2, [sstr{2} '.txt']), 'w');
        fprintf(fid2, '%s\n', [imdb.image_ids{i} '.jpg']);
        
        event_dir3 = fullfile(SUBMIT_cachedir3, event_name);
        mkdir_if_missing(event_dir3);
        fid3 = fopen(fullfile(event_dir3, [sstr{2} '.txt']), 'w');
        fprintf(fid3, '%s\n', [imdb.image_ids{i} '.jpg']);
        
        event_dir4 = fullfile(SUBMIT_cachedir4, event_name);
        mkdir_if_missing(event_dir4);
        fid4 = fopen(fullfile(event_dir4, [sstr{2} '.txt']), 'w');
        fprintf(fid4, '%s\n', [imdb.image_ids{i} '.jpg']);
        
        event_dir5 = fullfile(SUBMIT_cachedir5, event_name);
        mkdir_if_missing(event_dir5);
        fid5 = fopen(fullfile(event_dir5, [sstr{2} '.txt']), 'w');
        fprintf(fid5, '%s\n', [imdb.image_ids{i} '.jpg']);
        
        bbs_all = aboxes_v1{i};
        fprintf(fid1, '%d\n', size(bbs_all, 1));
        if ~isempty(bbs_all)
            for j = 1:size(bbs_all,1)
                %each row: [x1 y1 w h score]
                fprintf(fid1, '%d %d %d %d %f\n', round([bbs_all(j,1) bbs_all(j,2) bbs_all(j,3)-bbs_all(j,1)+1 bbs_all(j,4)-bbs_all(j,2)+1]), bbs_all(j, 5));
            end
        end
        
        bbs_all = aboxes_v2{i};
        fprintf(fid2, '%d\n', size(bbs_all, 1));
        if ~isempty(bbs_all)
            for j = 1:size(bbs_all,1)
                %each row: [x1 y1 w h score]
                fprintf(fid2, '%d %d %d %d %f\n', round([bbs_all(j,1) bbs_all(j,2) bbs_all(j,3)-bbs_all(j,1)+1 bbs_all(j,4)-bbs_all(j,2)+1]), bbs_all(j, 5));
            end
        end
        
        bbs_all = aboxes_v3{i};
        fprintf(fid3, '%d\n', size(bbs_all, 1));
        if ~isempty(bbs_all)
            for j = 1:size(bbs_all,1)
                %each row: [x1 y1 w h score]
                fprintf(fid3, '%d %d %d %d %f\n', round([bbs_all(j,1) bbs_all(j,2) bbs_all(j,3)-bbs_all(j,1)+1 bbs_all(j,4)-bbs_all(j,2)+1]), bbs_all(j, 5));
            end
        end
        
        bbs_all = aboxes_v4{i};
        fprintf(fid4, '%d\n', size(bbs_all, 1));
        if ~isempty(bbs_all)
            for j = 1:size(bbs_all,1)
                %each row: [x1 y1 w h score]
                fprintf(fid4, '%d %d %d %d %f\n', round([bbs_all(j,1) bbs_all(j,2) bbs_all(j,3)-bbs_all(j,1)+1 bbs_all(j,4)-bbs_all(j,2)+1]), bbs_all(j, 5));
            end
        end
        
        bbs_all = aboxes_v5{i};
        fprintf(fid5, '%d\n', size(bbs_all, 1));
        if ~isempty(bbs_all)
            for j = 1:size(bbs_all,1)
                %each row: [x1 y1 w h score]
                fprintf(fid5, '%d %d %d %d %f\n', round([bbs_all(j,1) bbs_all(j,2) bbs_all(j,3)-bbs_all(j,1)+1 bbs_all(j,4)-bbs_all(j,2)+1]), bbs_all(j, 5));
            end
        end
        fclose(fid1);
        fclose(fid2);
        fclose(fid3);
        fclose(fid4);
        fclose(fid5);
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

