function roidb_BF = do_generate_bf_proposal_twopath(conf, model_stage, imdb, roidb, nms_option)
    
    cache_dir = fullfile(pwd, 'output', conf.exp_name, 'rpn_cachedir', model_stage.cache_name, imdb.name);
    %cache_dir = fullfile(pwd, 'output', model_stage.cache_name, imdb.name);
    %save_roidb_name = fullfile(cache_dir, [ 'roidb_' imdb.name '_BF.mat']);
    %1011 changed
    save_roidb_name = fullfile(cache_dir, [ 'roidb_' imdb.name '_BF.mat']);
    if exist(save_roidb_name, 'file')
        ld = load(save_roidb_name);
        roidb_BF = ld.roidb_BF;
        clear ld;
        return;
    end
    % share the test with final3 for they have the same test network struct
    %[aboxes_res23, aboxes_res45]  = proposal_test_widerface_twopath_happy_flip(conf, imdb, ...
    %0129 added scale3 version
    [aboxes_res23, aboxes_res45]  = proposal_test_widerface_twopath_happy_flip_vn7(conf, imdb, ...
                                        'net_def_file',     model_stage.test_net_def_file, ...
                                        'net_file',         model_stage.output_model_file, ...
                                        'cache_name',       model_stage.cache_name, ...
                                        'suffix',           '_thr_60_60'); 
                               
    fprintf('Doing nms ... ');   
    % liu@1001: model_stage.nms.after_nms_topN functions as a threshold, indicating how many boxes will be preserved on average
    ave_per_image_topN_res23 = model_stage.nms.after_nms_topN_res23;
    ave_per_image_topN_res45 = model_stage.nms.after_nms_topN_res45;
    model_stage.nms.after_nms_topN_res23 = -1;
    model_stage.nms.after_nms_topN_res45 = -1;
    aboxes_res23              = boxes_filter(aboxes_res23, model_stage.nms.per_nms_topN, model_stage.nms.nms_overlap_thres, model_stage.nms.after_nms_topN_res23, conf.use_gpu);
    aboxes_res45              = boxes_filter(aboxes_res45, model_stage.nms.per_nms_topN, model_stage.nms.nms_overlap_thres, model_stage.nms.after_nms_topN_res45, conf.use_gpu);
    fprintf(' Done.\n');  
    
    % only use the first max_sample_num images to compute an "expected" lower bound thresh
    max_sample_num = 5000;
    
    % res23
    sample_aboxes = aboxes_res23(randperm(length(aboxes_res23), min(length(aboxes_res23), max_sample_num)));  % conv4 and conv5 are the same, so just use conv4
    scores = zeros(ave_per_image_topN_res23*length(sample_aboxes), 1);
    for i = 1:length(sample_aboxes)
        s_scores = sort([scores; sample_aboxes{i}(:, end)], 'descend');
        scores = s_scores(1:ave_per_image_topN_res23*length(sample_aboxes));
    end
    score_thresh_res23 = scores(end);
    
    % res45
    sample_aboxes = aboxes_res45(randperm(length(aboxes_res45), min(length(aboxes_res45), max_sample_num)));  % conv4 and conv5 are the same, so just use conv4
    scores = zeros(ave_per_image_topN_res45*length(sample_aboxes), 1);
    for i = 1:length(sample_aboxes)
        s_scores = sort([scores; sample_aboxes{i}(:, end)], 'descend');
        scores = s_scores(1:ave_per_image_topN_res45*length(sample_aboxes));
    end
    score_thresh_res45 = scores(end);

    fprintf('score_threshold res23 = %f, res45 = %f\n', score_thresh_res23, score_thresh_res45);
    
    % 1122 added to save combined results of conv4 and conv5
    aboxes = cell(length(aboxes_res23), 1);  % conv4 and conv6 are also ok
    aboxes_nms = cell(length(aboxes_res23), 1);
    
    for i = 1:length(aboxes_res23)
        aboxes_res23{i} = aboxes_res23{i}(aboxes_res23{i}(:, end) > 0.99, :);%0.65
        aboxes_res45{i} = aboxes_res45{i}(aboxes_res45{i}(:, end) > 0.99, :);%0.7
        aboxes{i} = cat(1, aboxes_res23{i}, aboxes_res45{i});
        
        %1006 added to do NPD-style nms
        time = tic;
        aboxes_nms{i} = pseudoNMS_v8_twopath(aboxes{i}, nms_option);
        
        fprintf('PseudoNMS for image %d cost %.1f seconds\n', i, toc(time));
        
    end
    
    aboxes_nms = boxes_filter(aboxes_nms, -1, 0.33, -1, conf.use_gpu); %0.5
    
    %roidb_regions.boxes = aboxes;
    roidb_regions.boxes = aboxes_nms;
    roidb_regions.images = imdb.image_ids;
    % concatenate gt boxes and high-scoring dt boxes
    roidb_BF                   = roidb_from_proposal_score(imdb, roidb, roidb_regions, ...
            'keep_raw_proposal', false);
        
    save(save_roidb_name, 'roidb_BF', '-v7.3');
%     for i = 1:length(aboxes_res23)
%         % draw boxes after 'naive' thresholding
%         sstr = strsplit(imdb.image_ids{i}, filesep);
%         event_name = sstr{1};
%         event_dir = fullfile(SUBMIT_cachedir, event_name);
%         mkdir_if_missing(event_dir);
%         fid = fopen(fullfile(event_dir, [sstr{2} '.txt']), 'w');
%         fprintf(fid, '%s\n', [imdb.image_ids{i} '.jpg']);
%         bbs_all = aboxes_nms{i};
%         
%         fprintf(fid, '%d\n', size(bbs_all, 1));
%         if ~isempty(bbs_all)
%             for j = 1:size(bbs_all,1)
%                 %each row: [x1 y1 w h score]
%                 fprintf(fid, '%d %d %d %d %f\n', round([bbs_all(j,1) bbs_all(j,2) bbs_all(j,3)-bbs_all(j,1)+1 bbs_all(j,4)-bbs_all(j,2)+1]), bbs_all(j, 5));
%             end
%         end
%         fclose(fid);
%         fprintf('Done with saving image %d bboxes.\n', i);
%         
%     end	
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

