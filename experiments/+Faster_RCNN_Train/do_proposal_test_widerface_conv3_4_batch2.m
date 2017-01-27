function roidb_fastrcnn = do_proposal_test_widerface_conv3_4_batch2(conf, model_stage, imdb, roidb)
    
    cache_dir = fullfile(pwd, 'output', conf.exp_name, 'rpn_cachedir', model_stage.cache_name, imdb.name);
    %cache_dir = fullfile(pwd, 'output', model_stage.cache_name, imdb.name);
    %save_roidb_name = fullfile(cache_dir, [ 'roidb_' imdb.name '_BF.mat']);
    %1011 changed
    save_roidb_name = fullfile(cache_dir, [ 'roidb_' imdb.name '_fastrcnn.mat']);
    if exist(save_roidb_name, 'file')
        ld = load(save_roidb_name);
        roidb_fastrcnn = ld.roidb_fastrcnn;
        clear ld;
        return;
    end
    % **** 1201 ***** currently only do BF for conv4 
    % aboxes_conv4 is the raw bbox output of conv4
    %0125 added extract score_maps
    %aboxes                      = proposal_test_widerface_conv3_4(conf, imdb, ...
    [aboxes, score_feat_maps]    = proposal_test_widerface_conv3_4_feat(conf, imdb, ...
                                        'net_def_file',     model_stage.test_net_def_file, ...
                                        'net_file',         model_stage.output_model_file, ...
                                        'cache_name',       model_stage.cache_name, ...
                                        'three_scales',     false); %0124: 1(false) for next step training, 3(true) for complete test
                               
    fprintf('Doing nms ... ');  
    % liu@1001: model_stage.nms.after_nms_topN functions as a threshold, indicating how many boxes will be preserved on average
    ave_per_image_topN = model_stage.nms.after_nms_topN;
    model_stage.nms.after_nms_topN = -1;
    aboxes        = boxes_filter(aboxes, model_stage.nms.per_nms_topN, model_stage.nms.nms_overlap_thres, model_stage.nms.after_nms_topN, conf.use_gpu);
    fprintf(' Done.\n');  
    
    % only use the first max_sample_num images to compute an "expected" lower bound thresh
    max_sample_num = 5000;
    % conv4
    sample_aboxes = aboxes(randperm(length(aboxes), min(length(aboxes), max_sample_num)));  % conv4 and conv5 are the same, so just use conv4
    scores = zeros(ave_per_image_topN*length(sample_aboxes), 1);
    for i = 1:length(sample_aboxes)
        s_scores = sort([scores; sample_aboxes{i}(:, end)], 'descend');
        scores = s_scores(1:ave_per_image_topN*length(sample_aboxes));
    end
    score_thresh = scores(end);
    
    fprintf('score_threshold = %f\n', score_thresh);
    
    for i = 1:length(aboxes)
        aboxes{i} = aboxes{i}(aboxes{i}(:, end) >= score_thresh, :);  %0.7
    end

    roidb_regions.boxes = aboxes;
    roidb_regions.images = imdb.image_ids;
    % concatenate gt boxes and high-scoring dt boxes
    roidb_fastrcnn = roidb_from_proposal_score(imdb, roidb, roidb_regions, 'keep_raw_proposal', false);
    %0125 added
    roidb_fastrcnn.score_feat_maps = score_feat_maps;
        
    save(save_roidb_name, 'roidb_fastrcnn', '-v7.3');
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

