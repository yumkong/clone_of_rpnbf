function do_proposal_test_widerface_ablation_total_2345_FDDB(conf,conf_fast_rcnn, model_stage, model_stage_fast, nms_option)
    % share the test with final3 for they have the same test network struct
    %[aboxes_res23, aboxes_res45]  = proposal_test_widerface_twopath_happy_flip(conf, imdb, ...
    %0129 added scale3 version
    [aboxes_s4, aboxes_s8, aboxes_s16]  = proposal_test_widerface_ablation_FDDB_scale3(conf, ...
                                        'net_def_file',     model_stage.test_net_def_file, ...
                                        'net_file',         model_stage.output_model_file, ...
                                        'cache_name',       model_stage.cache_name, ...
                                        'data_dir',         model_stage.data_dir,...
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
    %cache_dir = fullfile(pwd, 'output', fopts.exp_name, 'fast_rcnn_cachedir', fopts.cache_name, imdb.name); 
    cache_dir = fullfile(pwd, 'output', fopts.exp_name, 'fast_rcnn_cachedir', fopts.cache_name, 'FDDB_test'); 
    mkdir_if_missing(cache_dir);
    %  init log
    timestamp = datestr(datevec(now()), 'yyyymmdd_HHMMSS');
    mkdir_if_missing(fullfile(cache_dir, 'log'));
    log_file = fullfile(cache_dir, 'log', ['test_', timestamp, '.txt']);
    diary(log_file);
    %0409 masked, see line 60 instead
    %num_images = length(imdb.image_ids);
    %0409 changed
    %save_file = fullfile(cache_dir, ['aboxes_' imdb.name fopts.suffix]);
    save_file = fullfile(cache_dir, ['aboxes_FDDB' fopts.suffix]);
    
    % 1121: add these 3 lines for drawing
    addpath(fullfile('external','export_fig'));
    res_dir = fullfile(pwd, 'output', conf_fast_rcnn.exp_name, 'fast_rcnn_cachedir','res_pic');
    mkdir_if_missing(res_dir);
    %1126 added to refresh figure
    close all;
    
    %0409 added for FDDB list
    dataDir = model_stage.data_dir;
    imgDir = fullfile(dataDir, 'image','original');
    listFile = fullfile(dataDir, 'list', 'FDDB-list.txt');
    % read all image names
    [num_images, fileList] = FDDB_ReadList(listFile);
    
    %0409 changed
    %SUBMIT_cachedir1 = fullfile(pwd, 'output', conf_fast_rcnn.exp_name, 'fast_rcnn_cachedir', 'realtest_BP-FPN_cachedir1');
    %mkdir_if_missing(SUBMIT_cachedir1);
    % 0106 write results according to FDDB format
	SUBMIT_cachedir = fullfile(pwd, 'output', conf.exp_name, 'submit_mpfvn_FDDB_cachedir');  %submit_mprpn_FDDB_cachedir
    mkdir_if_missing(SUBMIT_cachedir);
    resultFile = fullfile(SUBMIT_cachedir,'results_tmp.txt');
	fout = fopen(resultFile, 'wt');
    
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
        
        %aboxes_old = cell(length(imdb.image_ids), 1);
        %aboxes_new = cell(length(imdb.image_ids), 1);
        aboxes_old = cell(num_images, 1);
        aboxes_new = cell(num_images, 1);

        count = 0;
        t_start = tic;
        for i = 1:length(aboxes_s4)
            count = count + 1;
            fprintf('%s: FDDB test %d/%d ', procid(), count, num_images);
            th = tic;
            %0409 changed
            imgFile = fullfile(imgDir, [fileList{i}, '.jpg']);
            im = imread(imgFile);
            %im = imread(imdb.image_at(i));
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
            %fastrcnn_score = fast_rcnn_im_detect_widerface_mpfvn_0402(conf_fast_rcnn, caffe_net, im, rpn_boxes);
            fastrcnn_score = fast_rcnn_im_detect_widerface_total_conv2345_nocxt(conf_fast_rcnn, caffe_net, im, rpn_boxes, max_rois_num_in_gpu);
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
    
    %aboxes_v1 = cell(length(imdb.image_ids), 1);
    aboxes_v1 = cell(num_images, 1);
    
    %0406 added to fix bug: some empty boxes are 'double' type, use the
    %following to make all cell units 'single'
    aboxes_new = cellfun(@(x) single(x), aboxes_new, 'UniformOutput', false);
    aa = cell2mat(aboxes_new);
    all_f_score = aa(:,5);
    max_all_f = max(all_f_score);
    min_all_f = min(all_f_score);
    count = 0;
    for i = 1:length(aboxes_s4)
        count = count + 1;
        fprintf('%s: FDDB test %d/%d \n', procid(), count, num_images);
        if ~isempty(aboxes_old{i})
            rpn_boxes = aboxes_old{i}(:, 1:4);
            rpn_score = aboxes_old{i}(:, 5);
            fastrcnn_score_raw = aboxes_new{i}(:, 5);
            %0328: cubic root - optimal by round 2
            fastrcnn_score = nthroot(fastrcnn_score_raw, 3);
            %fastrcnn_score = nthroot(fastrcnn_score_raw, 1);
            %0328 shrink to [0.8 1] - optimal by round 1
            fastrcnn_score1 = (fastrcnn_score - min_all_f)/(max_all_f - min_all_f)*0.1 + 0.9;
        end
        
        if ~isempty(rpn_boxes)
            aboxes_old{i} = [rpn_boxes rpn_score];
            aboxes_new{i} = [rpn_boxes fastrcnn_score];
            % 1 * rpn + 0.5 * fastrcnn_score is optimal by round 3&4
            % 0410 changed: 
            %aboxes_v1{i} = [rpn_boxes (rpn_score + 0.1 * fastrcnn_score1)];
            aboxes_v1{i} = [rpn_boxes (rpn_score + 0.1 * fastrcnn_score1 - 0.1)];
        else
            aboxes_old{i} = [];
            aboxes_new{i} = [];
            aboxes_v1{i} = [];
        end
        %0321: pseudoNMS_v8_ablation == pseudoNMS_v8_twopath
        aboxes_v1{i} = pseudoNMS_v8_ablation(aboxes_v1{i}, nms_option);%4

        if ~isempty(aboxes_v1{i})
            [~, scores_ind] = sort(aboxes_v1{i}(:,5), 'descend');
            aboxes_v1{i} = aboxes_v1{i}(scores_ind, :);
        end

    end
    aboxes_v1 = boxes_filter(aboxes_v1, -1, 0.33, -1, conf.use_gpu); %0.5
    
    for i = 1:length(aboxes_s4)
        
        bbs_all = aboxes_v1{i};
        %0410 added for removing face in face
        bbs_all = pseudoNMS_v8_MALF(bbs_all);
        %0409 added
        numFaces = size(bbs_all, 1);
        fprintf(fout, '%s\n%d\n', fileList{i}, numFaces);
        %fprintf(fid1, '%d\n', size(bbs_all, 1));
        if ~isempty(bbs_all)
            for j = 1:numFaces
                %each row: [x1 y1 w h score]
                fprintf(fout, '%d %d %d %d %f\n', round([bbs_all(j,1) bbs_all(j,2) bbs_all(j,3)-bbs_all(j,1)+1 bbs_all(j,4)-bbs_all(j,2)+1]), bbs_all(j, 5));
            end
        end

        fprintf('Done with saving image %d bboxes.\n', i);
        
        if 0      
            %1121 also draw gt boxes
            %0409 changed
            imgFile = fullfile(imgDir, [fileList{i}, '.jpg']);
            img = imread(imgFile);
            %img = imread(imdb.image_at(i));  
            
            figure(6); 
            imshow(img);  %im(img)
            hold on
            if ~isempty(bbs_all)
                % 0406:for displaying 'clean' image
                bbs_all = bbs_all(bbs_all(:,5)>=0.997,:);%1.097
            end
            if ~isempty(bbs_all)
                  % 0406:for displaying score <= 1
                  bbs_all(:, 5) = bbs_all(:, 5); 
                  bbs_all(:, 3) = bbs_all(:, 3) - bbs_all(:, 1) + 1;
                  bbs_all(:, 4) = bbs_all(:, 4) - bbs_all(:, 2) + 1;
                  bbApply('draw',bbs_all,'g');
            end
            hold off
        end
    end	
    fclose(fout);  %0409: for FDDB,all results saved in 1 file
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

