function do_proposal_test_FDDB_multibox_ohem(conf, model_stage, cache_name, method_name, nms_option, detector, caffe_net)
    % share the test with final3 for they have the same test network struct
    [aboxes_conv4, aboxes_conv5, aboxes_conv6]     = proposal_test_FDDB_multibox(conf, ...
                                        'net_def_file',     model_stage.test_net_def_file, ...
                                        'net_file',         model_stage.output_model_file, ...
                                        'cache_name',       model_stage.cache_name, ...
                                        'data_dir',         model_stage.data_dir,...
										'suffix',           '_thr_50_55_55'); 
                               
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
    show_image = true;
    save_result = false;
    % path to save file
    cache_dir = fullfile(pwd, 'output', conf.exp_name, 'rpn_cachedir', cache_name, method_name);
    mkdir_if_missing(cache_dir);
    
    %1007 tempararily use another cell to save bbox after nms
%     aboxes_nms_conv4 = cell(length(aboxes_conv4), 1);
%     aboxes_nms_conv5 = cell(length(aboxes_conv5), 1);
%     aboxes_nms_conv6 = cell(length(aboxes_conv6), 1);
    
    % 1122 added to save combined results of conv4 and conv5
    aboxes = cell(length(aboxes_conv5), 1);  % conv4 and conv6 are also ok
    aboxes_nms = cell(length(aboxes_conv5), 1);
    %nms_option = 3; %1, 2, 3
    %aboxes_nms2 = cell(length(aboxes), 1);
    %nms_option2 = 2; %1, 2, 3
    
    % 1121: add these 3 lines for drawing
    addpath(fullfile('external','export_fig'));
    res_dir = fullfile(pwd, 'output', conf.exp_name, 'rpn_cachedir','res_pic');
    mkdir_if_missing(res_dir);
    %1126 added to refresh figure
    close all;
	
    %1216 added
    dataDir = model_stage.data_dir;
    imgDir = fullfile(dataDir, 'image','original');
    listFile = fullfile(dataDir, 'list', 'FDDB-list.txt');
    % read all image names
    [~, fileList] = FDDB_ReadList(listFile);
    % in window system, replace '/' with '\'
    % 0107: always make it '/' even for windows system
    %if ispc
    %    fileList = strrep(fileList, '/', '\');
    %end
    for i = 1:length(aboxes_conv4)
        
        %aboxes_nms{i} = cat(1, aboxes_conv4{i}(aboxes_conv4{i}(:, end) > score_thresh_conv4, :),...
        %                       aboxes_conv5{i}(aboxes_conv5{i}(:, end) > score_thresh_conv5, :));
        aboxes_conv4{i} = aboxes_conv4{i}(aboxes_conv4{i}(:, end) > 0.65, :);  %score_thresh_conv4, 0.7
        aboxes_conv5{i} = aboxes_conv5{i}(aboxes_conv5{i}(:, end) > 0.7, :);%score_thresh_conv5, 0.8
        aboxes_conv6{i} = aboxes_conv6{i}(aboxes_conv6{i}(:, end) > 0.7, :);%score_thresh_conv6, 0.8
        aboxes{i} = cat(1, aboxes_conv4{i}, aboxes_conv5{i}, aboxes_conv6{i});
        % draw boxes after 'naive' thresholding
        if 0
            %1216 changed
            imgFile = fullfile(imgDir, [fileList{i}, '.jpg']);
            img = imread(imgFile);
            %img = imread(imdb.image_at(i));  
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
        
        %1006 added to do NPD-style nms
        time = tic;
        aboxes_nms{i} = pseudoNMS_v8_FDDB(aboxes{i}, nms_option);
        bbs_all = aboxes_nms{i};
        fprintf('PseudoNMS for image %d cost %.1f seconds\n', i, toc(time));
        if 0         
            %1121 also draw gt boxes
            figure(2); 
            imshow(img);  %im(img)
            hold on
            if ~isempty(bbs_all)
                  bbs_all(:, 3) = bbs_all(:, 3) - bbs_all(:, 1) + 1;
                  bbs_all(:, 4) = bbs_all(:, 4) - bbs_all(:, 2) + 1;
                  bbApply('draw',bbs_all,'g');
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
    aboxes_nms = boxes_filter(aboxes_nms, -1, 0.33, -1, conf.use_gpu);  %0.5
	% 0106 write results according to FDDB format
	SUBMIT_cachedir = fullfile(pwd, 'output', conf.exp_name, 'submit_bf_FDDB_cachedir');  %submit_mprpn_FDDB_cachedir
    mkdir_if_missing(SUBMIT_cachedir);
    final_score_path = fullfile(pwd, 'output', conf.exp_name, 'rpn_cachedir', model_stage.cache_name, 'FDDB_test');
    mkdir_if_missing(final_score_path);
    final_score_file = fullfile(final_score_path, 'FDDB_box_score.mat');
	resultFile = fullfile(SUBMIT_cachedir,'results.txt');
	fout = fopen(resultFile, 'wt');
    try
        % try to load cache
        ld = load(final_score_file);
        bbs_repo = ld.bbs_repo;
        bf_score_min = ld.bf_score_min;
        bf_score_max = ld.bf_score_max;
        clear ld;
    catch   
        for i = 1:length(aboxes_conv4)
            % draw boxes after 'naive' thresholding
            fprintf('Computing BF scores for image %d / %d\n', i, length(aboxes_conv4));
            bbs = aboxes_nms{i};
            %do bf here
            if ~isempty(bbs)
                imgFile = fullfile(imgDir, [fileList{i}, '.jpg']);
                img = imread(imgFile);
                feat = rois_get_features_ratio_4x4_context_FDDB(conf, caffe_net, img, bbs, 3000, 2);   
                bf_scores = adaBoostApply(feat, detector.clf);
                mprpn_scores = bbs(:,5);
                bbs_all = [bbs(:,1:4) bf_scores mprpn_scores];
            else
                bbs_all = [];
            end
            bbs_repo{i} = bbs_all;
        end
        for i = 1:length(bbs_repo)
           if ~isa(bbs_repo{i}, 'single')
               bbs_repo{i} = single(bbs_repo{i});
           end
        end
        bbs_repo = bbs_repo';
        bbs_tmp = cell2mat(bbs_repo);
        bf_score_min = min(bbs_tmp(:,5));
        bf_score_max = max(bbs_tmp(:,5));
        save(final_score_file, 'bbs_repo','bf_score_min','bf_score_max');
        clear bbs_tmp;
    end
    % optional: cubic root of bf scores
    bf_score_min = nthroot(bf_score_min, 3);
    bf_score_max = nthroot(bf_score_max, 3);
    for i = 1:length(aboxes_conv4)
        bbs = bbs_repo{i};
		numFaces = size(bbs, 1);
        fprintf(fout, '%s\n%d\n', fileList{i}, numFaces);

        if ~isempty(bbs)
            % optinal: cubic root of bf scores
            bbs(:,5) = nthroot(bbs(:,5), 3);
            bbs(:,5) = (bbs(:,5) - bf_score_min) / (bf_score_max - bf_score_min);
            for j = 1:numFaces
                %each row: [x1 y1 w h score]
                %fprintf(fout, '%d %d %d %d %f\n', round([bbs_all(j,1) bbs_all(j,2) bbs_all(j,3)-bbs_all(j,1)+1 bbs_all(j,4)-bbs_all(j,2)+1]), bbs_all(j, 5));
                fprintf(fout, '%d %d %d %d %f\n', round([bbs(j,1) bbs(j,2) bbs(j,3)-bbs(j,1)+1 bbs(j,4)-bbs(j,2)+1]), (bbs(j, 5) + 2*bbs(j, 6))/3);
            end
        end
        
        fprintf('Done with saving image %d bboxes.\n', i);
        if 1        
            %1121 also draw gt boxes
            figure(3); 
            imgFile = fullfile(imgDir, [fileList{i}, '.jpg']);
            img = imread(imgFile);
            imshow(img);  %im(img)
            hold on
            if ~isempty(bbs)
                  bbs(:, 3) = bbs(:, 3) - bbs(:, 1) + 1;
                  bbs(:, 4) = bbs(:, 4) - bbs(:, 2) + 1;
                  bbs(:, 5) = (bbs(:, 5) + 2*bbs(:, 6))/3;
                  % only show boxes with a score >= 0.8
                  sel_idx = bbs(:, 5) >= 0.8;
                  bbs = bbs(sel_idx, :);
                  bbApply('draw',bbs,'g');
            end
            hold off
        end
    end
	fclose(fout);

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

