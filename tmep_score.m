        count = 0;
        t_start = tic;
        for i = 1:num_images
            count = count + 1;
            fprintf('%s: test (%s) %d/%d ', procid(), imdb.name, count, num_images);
            th = tic;
            d = roidb.rois(i);
            im = imread(imdb.image_at(i));

            rpn_boxes = d.boxes(~d.gt, :);
            rpn_score = 
			fastrcnn_score_pno = 
            
            if ~isempty(fastrcnn_score)
                %0328: cubic root
                fastrcnn_score_p2 = nthroot(fastrcnn_score, 2);
                fastrcnn_score_p3 = nthroot(fastrcnn_score, 3);

                %0328 shrink to [0.8 1]
                %fastrcnn_score_p9 = (fastrcnn_score - min(fastrcnn_score))/(max(fastrcnn_score) - min(fastrcnn_score))*0.1 + 0.9;
                fastrcnn_score_p1 = (fastrcnn_score_p1 - min(fastrcnn_score_p1))/(max(fastrcnn_score_p1) - min(fastrcnn_score_p1))*0.2 + 0.8;
                fastrcnn_score_p2 = (fastrcnn_score_p2 - min(fastrcnn_score_p2))/(max(fastrcnn_score_p2) - min(fastrcnn_score_p2))*0.2 + 0.8;
                fastrcnn_score_p3 = (fastrcnn_score_p3 - min(fastrcnn_score_p3))/(max(fastrcnn_score_p3) - min(fastrcnn_score_p3))*0.2 + 0.8;
                fastrcnn_score_p4 = (fastrcnn_score_p4 - min(fastrcnn_score_p4))/(max(fastrcnn_score_p4) - min(fastrcnn_score_p4))*0.2 + 0.8;
                fastrcnn_score_p5 = (fastrcnn_score_p5 - min(fastrcnn_score_p5))/(max(fastrcnn_score_p5) - min(fastrcnn_score_p5))*0.2 + 0.8;
                fastrcnn_score_p6 = (fastrcnn_score_p6 - min(fastrcnn_score_p6))/(max(fastrcnn_score_p6) - min(fastrcnn_score_p6))*0.2 + 0.8;
            end
            %tmp_boxes = d.boxes(~d.gt, :);
            %rpn_score = d.scores(~d.gt, :);
            %fastrcnn_score = scores(~d.gt, :);
            if ~isempty(rpn_boxes)
                aboxes_old{i} = [rpn_boxes rpn_score];
                aboxes_new{i} = [rpn_boxes fastrcnn_score_pno];
                %aboxes_2old_1new_pno{i} = [rpn_boxes (2*rpn_score+fastrcnn_score_pno)/3];
                aboxes_2old_1new_pno{i} = [rpn_boxes (2*rpn_score+fastrcnn_score_pno)/2];
                 aboxes_2old_1new_p1{i} = [rpn_boxes (2*rpn_score+fastrcnn_score_p1)/2];
                 aboxes_2old_1new_p2{i} = [rpn_boxes (2*rpn_score+fastrcnn_score_p2)/2];
                 aboxes_2old_1new_p3{i} = [rpn_boxes (2*rpn_score+fastrcnn_score_p3)/2];
                 aboxes_2old_1new_p4{i} = [rpn_boxes (2*rpn_score+fastrcnn_score_p4)/2];
                 aboxes_2old_1new_p5{i} = [rpn_boxes (2*rpn_score+fastrcnn_score_p5)/2];
                 aboxes_2old_1new_p6{i} = [rpn_boxes (2*rpn_score+fastrcnn_score_p6)/2];
            else
                aboxes_old{i} = [];
                aboxes_new{i} = [];
                aboxes_2old_1new_pno{i} = [];
%                 aboxes_2old_1new_p9{i} = [];
%                 aboxes_2old_1new_p8{i} = [];
%                 aboxes_2old_1new_p7{i} = [];
                aboxes_2old_1new_p6{i} = [];
                aboxes_2old_1new_p5{i} = [];
                aboxes_2old_1new_p4{i} = [];
                aboxes_2old_1new_p3{i} = [];
                aboxes_2old_1new_p2{i} = [];
                aboxes_2old_1new_p1{i} = [];
            end
            
            % 0310: for rpn score
            aboxes_old{i} = pseudoNMS_v8_twopath(aboxes_old{i}, 3);%nms_option=3
            if ~isempty(aboxes_old{i})
                [~, scores_ind] = sort(aboxes_old{i}(:,5), 'descend');
                aboxes_old{i} = aboxes_old{i}(scores_ind, :);
            end
            
            % 0310: for fastrcnn score
            aboxes_new{i} = pseudoNMS_v8_twopath(aboxes_new{i}, 3);%nms_option=3
            if ~isempty(aboxes_new{i})
                [~, scores_ind] = sort(aboxes_new{i}(:,5), 'descend');
                aboxes_new{i} = aboxes_new{i}(scores_ind, :);
            end
            
            %0329 added
            aboxes_2old_1new_pno{i} = pseudoNMS_v8_twopath(aboxes_2old_1new_pno{i}, 3);%nms_option=3
            if ~isempty(aboxes_2old_1new_pno{i})
                [~, scores_ind] = sort(aboxes_2old_1new_pno{i}(:,5), 'descend');
                aboxes_2old_1new_pno{i} = aboxes_2old_1new_pno{i}(scores_ind, :);
            end
            aboxes_2old_1new_p6{i} = pseudoNMS_v8_twopath(aboxes_2old_1new_p6{i}, 3);%nms_option=3
            if ~isempty(aboxes_2old_1new_p6{i})
                [~, scores_ind] = sort(aboxes_2old_1new_p6{i}(:,5), 'descend');
                aboxes_2old_1new_p6{i} = aboxes_2old_1new_p6{i}(scores_ind, :);
            end
            aboxes_2old_1new_p5{i} = pseudoNMS_v8_twopath(aboxes_2old_1new_p5{i}, 3);%nms_option=3
            if ~isempty(aboxes_2old_1new_p5{i})
                [~, scores_ind] = sort(aboxes_2old_1new_p5{i}(:,5), 'descend');
                aboxes_2old_1new_p5{i} = aboxes_2old_1new_p5{i}(scores_ind, :);
            end
            aboxes_2old_1new_p4{i} = pseudoNMS_v8_twopath(aboxes_2old_1new_p4{i}, 3);%nms_option=3
            if ~isempty(aboxes_2old_1new_p4{i})
                [~, scores_ind] = sort(aboxes_2old_1new_p4{i}(:,5), 'descend');
                aboxes_2old_1new_p4{i} = aboxes_2old_1new_p4{i}(scores_ind, :);
            end
            aboxes_2old_1new_p3{i} = pseudoNMS_v8_twopath(aboxes_2old_1new_p3{i}, 3);%nms_option=3
            if ~isempty(aboxes_2old_1new_p3{i})
                [~, scores_ind] = sort(aboxes_2old_1new_p3{i}(:,5), 'descend');
                aboxes_2old_1new_p3{i} = aboxes_2old_1new_p3{i}(scores_ind, :);
            end
            aboxes_2old_1new_p2{i} = pseudoNMS_v8_twopath(aboxes_2old_1new_p2{i}, 3);%nms_option=3
            if ~isempty(aboxes_2old_1new_p2{i})
                [~, scores_ind] = sort(aboxes_2old_1new_p2{i}(:,5), 'descend');
                aboxes_2old_1new_p2{i} = aboxes_2old_1new_p2{i}(scores_ind, :);
            end
            aboxes_2old_1new_p1{i} = pseudoNMS_v8_twopath(aboxes_2old_1new_p1{i}, 3);%nms_option=3
            if ~isempty(aboxes_2old_1new_p1{i})
                [~, scores_ind] = sort(aboxes_2old_1new_p1{i}(:,5), 'descend');
                aboxes_2old_1new_p1{i} = aboxes_2old_1new_p1{i}(scores_ind, :);
            end

            fprintf(' time: %.3fs\n', toc(th));     
        end
        
        start_thresh = 5; %5
    thresh_interval = 3;%3
    thresh_end = 500; % 500
    [gt_num_all, gt_recall_all, gt_num_pool, gt_recall_pool] = Get_Detector_Recall_finegrained(roidb, aboxes_old, start_thresh,thresh_interval,thresh_end);
    save(fullfile(cache_dir,'recall_vector_rpn.mat'),'gt_num_all', 'gt_recall_all', 'gt_num_pool', 'gt_recall_pool');
    fprintf('rpn all scales: gt recall rate = %d / %d = %.4f\n', gt_recall_all, gt_num_all, gt_recall_all/gt_num_all);
    
    [gt_num_all, gt_recall_all, gt_num_pool, gt_recall_pool] = Get_Detector_Recall_finegrained(roidb, aboxes_new, start_thresh,thresh_interval,thresh_end);
    save(fullfile(cache_dir,'recall_vector_fastrcnn.mat'),'gt_num_all', 'gt_recall_all', 'gt_num_pool', 'gt_recall_pool');
    fprintf('fastrcnn all scales: gt recall rate = %d / %d = %.4f\n', gt_recall_all, gt_num_all, gt_recall_all/gt_num_all);
    
    [gt_num_all, gt_recall_all, gt_num_pool, gt_recall_pool] = Get_Detector_Recall_finegrained(roidb, aboxes_2old_1new_pno, start_thresh,thresh_interval,thresh_end);
    save(fullfile(cache_dir,'recall_vector_2rpn_1fastrcnn_pno.mat'),'gt_num_all', 'gt_recall_all', 'gt_num_pool', 'gt_recall_pool');
    fprintf('pno all scales: gt recall rate = %d / %d = %.4f\n', gt_recall_all, gt_num_all, gt_recall_all/gt_num_all);
    
    [gt_num_all, gt_recall_all, gt_num_pool, gt_recall_pool] = Get_Detector_Recall_finegrained(roidb, aboxes_2old_1new_p6, start_thresh,thresh_interval,thresh_end);
    save(fullfile(cache_dir,'recall_vector_2rpn_1fastrcnn_p6.mat'),'gt_num_all', 'gt_recall_all', 'gt_num_pool', 'gt_recall_pool');
    fprintf('p6 all scales: gt recall rate = %d / %d = %.4f\n', gt_recall_all, gt_num_all, gt_recall_all/gt_num_all);
    
    [gt_num_all, gt_recall_all, gt_num_pool, gt_recall_pool] = Get_Detector_Recall_finegrained(roidb, aboxes_2old_1new_p5, start_thresh,thresh_interval,thresh_end);
    save(fullfile(cache_dir,'recall_vector_2rpn_1fastrcnn_p5.mat'),'gt_num_all', 'gt_recall_all', 'gt_num_pool', 'gt_recall_pool');
    fprintf('p5 all scales: gt recall rate = %d / %d = %.4f\n', gt_recall_all, gt_num_all, gt_recall_all/gt_num_all);
    
    [gt_num_all, gt_recall_all, gt_num_pool, gt_recall_pool] = Get_Detector_Recall_finegrained(roidb, aboxes_2old_1new_p4, start_thresh,thresh_interval,thresh_end);
    save(fullfile(cache_dir,'recall_vector_2rpn_1fastrcnn_p4.mat'),'gt_num_all', 'gt_recall_all', 'gt_num_pool', 'gt_recall_pool');
    fprintf('p4 all scales: gt recall rate = %d / %d = %.4f\n', gt_recall_all, gt_num_all, gt_recall_all/gt_num_all);
    
    [gt_num_all, gt_recall_all, gt_num_pool, gt_recall_pool] = Get_Detector_Recall_finegrained(roidb, aboxes_2old_1new_p3, start_thresh,thresh_interval,thresh_end);
    save(fullfile(cache_dir,'recall_vector_2rpn_1fastrcnn_p3.mat'),'gt_num_all', 'gt_recall_all', 'gt_num_pool', 'gt_recall_pool');
    fprintf('p3 all scales: gt recall rate = %d / %d = %.4f\n', gt_recall_all, gt_num_all, gt_recall_all/gt_num_all);
    
    [gt_num_all, gt_recall_all, gt_num_pool, gt_recall_pool] = Get_Detector_Recall_finegrained(roidb, aboxes_2old_1new_p2, start_thresh,thresh_interval,thresh_end);
    save(fullfile(cache_dir,'recall_vector_2rpn_1fastrcnn_p2.mat'),'gt_num_all', 'gt_recall_all', 'gt_num_pool', 'gt_recall_pool');
    fprintf('p2 all scales: gt recall rate = %d / %d = %.4f\n', gt_recall_all, gt_num_all, gt_recall_all/gt_num_all);
    
    [gt_num_all, gt_recall_all, gt_num_pool, gt_recall_pool] = Get_Detector_Recall_finegrained(roidb, aboxes_2old_1new_p1, start_thresh,thresh_interval,thresh_end);
    save(fullfile(cache_dir,'recall_vector_2rpn_1fastrcnn_p1.mat'),'gt_num_all', 'gt_recall_all', 'gt_num_pool', 'gt_recall_pool');
    fprintf('p1 all scales: gt recall rate = %d / %d = %.4f\n', gt_recall_all, gt_num_all, gt_recall_all/gt_num_all);