function aboxes = do_proposal_test_widerface_my(conf, model_stage, imdb, roidb, cache_name, method_name)
    aboxes                      = proposal_test_widerface(conf, imdb, ...
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
    fprintf('score_threshold:%f\n', score_thresh);
    % drop the boxes which scores are lower than the threshold
    show_image = true;
    % path to save file
    cache_dir = fullfile(pwd, 'output', conf.exp_name, 'rpn_cachedir', cache_name, method_name);
    mkdir_if_missing(cache_dir);
    
    %1007 tempararily use another cell to save bbox after nms
    aboxes_nms = cell(length(aboxes), 1);
    nms_option = 3; %1, 2, 3
    
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
              im(img);  %im(I)
              bbApply('draw',bbs);
            end
        end
        
        %1006 added to do NPD-style nms
        time = tic;
        % 1007 do nms
        %if i == 102
        %   fprintf('Hoori\n'); 
        %end
        aboxes_nms{i} = pseudoNMS_v3(aboxes{i}, nms_option);
        
        fprintf('PseudoNMS for image %d cost %.1f seconds\n', i, toc(time));
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
    
    % save bbox before nms
    bbox_save_name = fullfile(cache_dir, sprintf('VGG16_e1-e3-ave-%d.txt', ave_per_image_topN));
    save_bbox_to_txt(aboxes, imdb.image_ids, bbox_save_name);
    % save bbox after nms
    bbox_save_name = fullfile(cache_dir, sprintf('VGG16_e1-e3-ave-%d-nms-op%d.txt', ave_per_image_topN, nms_option));
    save_bbox_to_txt(aboxes_nms, imdb.image_ids, bbox_save_name);
	
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
    % 1007 added
    fprintf('gt recall rate after nms-%d = %.4f\n', nms_option, gt_re_num_nms / gt_num_nms);

    fprintf('Preparing the results for widerface Precision-Recall (VOC-style) evaluation ...');
    % first prepare for gt
%     cache_dir = fullfile(pwd, 'output', conf.exp_name, 'rpn_cachedir', cache_name, method_name);
%     mkdir_if_missing(cache_dir);
    
    annotation_save_name = fullfile(cache_dir, 'widerface_anno_e1-e3.mat');
    if ~exist(annotation_save_name, 'file')
        gt_im_num = length(roidb.rois);
        objects = cell(gt_im_num, 1);
        imgname = cell(gt_im_num, 1);
        for kk = 1:gt_im_num
            tmp_name = strsplit(imdb.image_ids{kk}, filesep);
            imgname{kk} = tmp_name{2};
            tmp_box = roidb.rois(kk).boxes;
            objects{kk} = [tmp_box zeros(size(tmp_box, 1), 2)];
        end
        Annotations = struct('imgname', imgname, 'objects', objects);
        save(annotation_save_name, 'Annotations');
    end
    % then prepare for dt
%     fid = fopen(fullfile(cache_dir, 'VGG16_e1-e3-pseudoNMS05-2.txt'), 'a');
%     assert(length(imdb.image_ids) == size(aboxes, 1));
%     for i = 1:size(aboxes, 1)
%         if ~isempty(aboxes{i})
%             sstr = strsplit(imdb.image_ids{i}, filesep);
%             % [x1 y1 x2 y2] pascal VOC style
%             for j = 1:size(aboxes{i}, 1)
%                 %each row: [image_name score x1 y1 x2 y2]
%                 fprintf(fid, '%s %f %d %d %d %d\n', sstr{2}, aboxes{i}(j, 5), round(aboxes{i}(j, 1:4)));
%             end
%         end
%     end
%     fclose(fid);
    fprintf('Done.\n');
    
%     fprintf('Preparing the results for Caltech evaluation ...');
%     %cache_dir = fullfile(pwd, 'output', conf.exp_name, 'rpn_cachedir', cache_name);
%     cache_dir = fullfile(pwd, 'output');
%     res_boxes = aboxes;
%     mkdir_if_missing(fullfile(cache_dir, method_name));
%     % remove all the former results
%     DIRS=dir(fullfile(fullfile(cache_dir, method_name))); 
%     n=length(DIRS);
%     for i=1:n
%         if (DIRS(i).isdir && ~strcmp(DIRS(i).name,'.') && ~strcmp(DIRS(i).name,'..') )
%             rmdir(fullfile(cache_dir, method_name ,DIRS(i).name),'s');
%         end
%     end
%     
%     assert(length(imdb.image_ids) == size(res_boxes, 1));
%     for i = 1:size(res_boxes, 1)
%         if ~isempty(res_boxes{i})
%             sstr = strsplit(imdb.image_ids{i}, '\');
%             mkdir_if_missing(fullfile(cache_dir, method_name, sstr{1}));
%             fid = fopen(fullfile(cache_dir, method_name, sstr{1}, [sstr{2} '.txt']), 'a');
%             % transform [x1 y1 x2 y2] to [x y w h], for matching the
%             % caltech evaluation protocol
%             res_boxes{i}(:, 3) = res_boxes{i}(:, 3) - res_boxes{i}(:, 1);
%             res_boxes{i}(:, 4) = res_boxes{i}(:, 4) - res_boxes{i}(:, 2);
%             for j = 1:size(res_boxes{i}, 1)
%                 fprintf(fid, '%d,%f,%f,%f,%f,%f\n', str2double(sstr{3}(2:end))+1, res_boxes{i}(j, :));
%             end
%             fclose(fid);
%         end
%     end
%     fprintf('Done.');
%     
%     % copy results to eval folder and run eval script to get figure.
%     folder1 = fullfile(pwd, 'output', conf.exp_name, 'rpn_cachedir', cache_name, method_name);
%     folder2 = fullfile(pwd, 'external', 'code3.2.1', 'data-USA', 'res', method_name);
%     mkdir_if_missing(folder2);
%     copyfile(folder1, folder2);
%     tmp_dir = pwd;
%     cd(fullfile(pwd, 'external', 'code3.2.1'));
%     dbEval_RPNBF;
%     cd(tmp_dir);
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

