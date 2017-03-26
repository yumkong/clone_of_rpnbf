function roidb_fastrcnn = do_proposal_test_widerface_ablation_fastrcnn_flip(conf, model_stage, imdb, roidb, nms_option)

    % 0304: first judge if there is roidb_fastrcnn (for fastrcnn training)
    % if have, directly load and return, otherwise generate from scratch
    cache_dir = fullfile(pwd, 'output', conf.exp_name, 'rpn_cachedir', model_stage.cache_name, imdb.name);
    %1011 changed
    save_roidb_name = fullfile(cache_dir, [ 'roidb_' imdb.name '_fastrcnn.mat']);
    if exist(save_roidb_name, 'file')
        ld = load(save_roidb_name);
        roidb_fastrcnn = ld.roidb_fastrcnn;
        clear ld;
        return;
    end

    % 0304: if have aboxes that already passed nms, load it
    % otherwise regenerate it
    try
        % try to load cache
        box_nms_name = fullfile(cache_dir, ['proposal_boxes_fastrcnn_' imdb.name]);
        ld = load(box_nms_name);
        % 0304: currently do not need this three, unmask them when need
        %aboxes_s4 = ld.aboxes_s4;
        %aboxes_s8 = ld.aboxes_s8;
        %aboxes_s16 = ld.aboxes_s16;
        aboxes = ld.aboxes;
    catch 
        % 0304: if having nothing, generate all boxes from scratch
        [aboxes_conv4, aboxes_conv5, aboxes_conv6]     = proposal_test_widerface_ablation_final_flip(conf, imdb, ...
                                        'net_def_file',     model_stage.test_net_def_file, ...
                                        'net_file',         model_stage.output_model_file, ...
                                        'cache_name',       model_stage.cache_name, ...
                                        'suffix',           '_thr_10_10_10');
        fprintf('Doing nms ... ');   
        % liu@1001: model_stage.nms.after_nms_topN functions as a threshold, indicating how many boxes will be preserved on average
        ave_per_image_topN_conv4 = model_stage.nms.after_nms_topN_s4; % conv4
        ave_per_image_topN_conv5 = model_stage.nms.after_nms_topN_s8; % conv5
        ave_per_image_topN_conv6 = model_stage.nms.after_nms_topN_s16; % conv6
        model_stage.nms.after_nms_topN_s4 = -1;
        model_stage.nms.after_nms_topN_s8 = -1;
        model_stage.nms.after_nms_topN_s16 = -1;
        aboxes_conv4              = boxes_filter(aboxes_conv4, model_stage.nms.per_nms_topN, model_stage.nms.nms_overlap_thres, model_stage.nms.after_nms_topN_s4, conf.use_gpu);
        aboxes_conv5              = boxes_filter(aboxes_conv5, model_stage.nms.per_nms_topN, model_stage.nms.nms_overlap_thres, model_stage.nms.after_nms_topN_s8, conf.use_gpu);
        aboxes_conv6              = boxes_filter(aboxes_conv6, model_stage.nms.per_nms_topN, model_stage.nms.nms_overlap_thres, model_stage.nms.after_nms_topN_s16, conf.use_gpu);
        fprintf(' Done.\n');  

        % only use the first max_sample_num images to compute an "expected" lower bound thresh
        max_sample_num = 2000; %0304: 3000-->2000
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

        fprintf('score_threshold s4 = %f, s8 = %f, s16 = %f\n', score_thresh_conv4, score_thresh_conv5, score_thresh_conv6);

        %1007 tempararily use another cell to save bbox after nms
        aboxes_s4 = cell(length(aboxes_conv4), 1);
        aboxes_s8 = cell(length(aboxes_conv5), 1);
        aboxes_s16 = cell(length(aboxes_conv6), 1);

        %0103 added
        aboxes = cell(length(aboxes_conv5), 1);  % conv4 and conv6 are also ok
        %aboxes_nms = cell(length(aboxes_conv5), 1);
        for i = 1:length(aboxes_conv4)

            %aboxes_nms{i} = cat(1, aboxes_conv4{i}(aboxes_conv4{i}(:, end) > score_thresh_conv4, :),...
            %                       aboxes_conv5{i}(aboxes_conv5{i}(:, end) > score_thresh_conv5, :));
            aboxes_s4{i} = aboxes_conv4{i}(aboxes_conv4{i}(:, end) > score_thresh_conv4, :);
            aboxes_s8{i} = aboxes_conv5{i}(aboxes_conv5{i}(:, end) > score_thresh_conv5, :);
            aboxes_s16{i} = aboxes_conv6{i}(aboxes_conv6{i}(:, end) > score_thresh_conv6, :);
            aboxes{i} = cat(1, aboxes_s4{i}, aboxes_s8{i}, aboxes_s16{i});
            % 0304 masked: currently no pseudoNMS is included, because this
            % is not final result, just prepare train data for next step
            %1006 added to do NPD-style nms
%             time = tic;
%             aboxes_s4{i} = pseudoNMS_v8_twopath(aboxes_conv4{i}, nms_option);
%             aboxes_s8{i} = pseudoNMS_v8_twopath(aboxes_conv5{i}, nms_option);
%             aboxes_s16{i} = pseudoNMS_v8_twopath(aboxes_conv6{i}, nms_option);
%             aboxes_nms{i} = pseudoNMS_v8_twopath(aboxes{i}, nms_option);
%             fprintf('PseudoNMS for image %d cost %.1f seconds\n', i, toc(time));
%             %0226 added: sort by score
%             if ~isempty(aboxes_s4{i})
%                 [~, scores_ind] = sort(aboxes_s4{i}(:,5), 'descend');
%                 aboxes_s4{i} = aboxes_s4{i}(scores_ind, :);
%             end
%             if ~isempty(aboxes_s8{i})
%                 [~, scores_ind] = sort(aboxes_s8{i}(:,5), 'descend');
%                 aboxes_s8{i} = aboxes_s8{i}(scores_ind, :);
%             end
%             if ~isempty(aboxes_s16{i})
%                 [~, scores_ind] = sort(aboxes_s16{i}(:,5), 'descend');
%                 aboxes_s16{i} = aboxes_s16{i}(scores_ind, :);
%             end
        end
        
        save(box_nms_name, 'aboxes_s4', 'aboxes_s8', 'aboxes_s16', 'aboxes');
    end
    % 0206 added
    roidb_regions.boxes = aboxes;
    roidb_regions.images = imdb.image_ids;
    % concatenate gt boxes and high-scoring dt boxes
    roidb_fastrcnn = roidb_from_proposal_score(imdb, roidb, roidb_regions, 'keep_raw_proposal', false);
    %0125 added
    %roidb_fastrcnn.score_feat_maps = score_feat_maps;  
    save(save_roidb_name, 'roidb_fastrcnn', '-v7.3');
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

