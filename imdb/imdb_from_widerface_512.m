function [imdb, roidb] = imdb_from_widerface_512(root_dir, image_set, flip, cache_dir, model_name_base, event_num)
%function imdb_from_widerface(devkit, 'trainval', use_flip)

switch image_set
    case {'trainval'}
        if event_num == -1
            data_num_str = 'all';
        elseif event_num > 0
            data_num_str = sprintf('e1-e%d', event_num);
        end
        cache_imdb = fullfile(cache_dir, sprintf('train_imdb_%s_%s_raw.mat',model_name_base, data_num_str));  %imdb
        cache_roidb = fullfile(cache_dir, sprintf('train_roidb_%s_%s_raw.mat', model_name_base, data_num_str));  %roidb
        devpath = fullfile('WIDER_train','images');
        doc_dir = fullfile('wider_face_split','wider_face_train');
        name = 'WIDERFACE_train';
    case {'test'}
        if event_num == -1
            data_num_str = 'all';
        elseif event_num > 0
            data_num_str = sprintf('e1-e%d', event_num);
        end
        cache_imdb = fullfile(cache_dir, sprintf('test_imdb_%s_%s_raw.mat',model_name_base, data_num_str));  %imdb
        cache_roidb = fullfile(cache_dir, sprintf('test_roidb_%s_%s_raw.mat', model_name_base, data_num_str));  %roidb
        devpath = fullfile('WIDER_val','images');
        doc_dir = fullfile('wider_face_split','wider_face_val');
        name = 'WIDERFACE_test';
    otherwise
        error('usage = ''trainval'' or ''test''');
end

try
    load(cache_imdb); %'dataset' -- struct, containing imdb
catch
    imdb = [];
    %(1)
    imdb.name = name;
    %(2)
    rootdir = root_dir;
    imdb.image_dir = fullfile(rootdir, devpath);

    % have 3 cell elements
    % event_list
    % face_bbx_list
    % file_list
    annodoc = load(fullfile(rootdir, doc_dir));

    imgsum_ = 0;
    if event_num <= 0
        eventNum = numel(annodoc.file_list);  % use all events
    else
        eventNum = event_num;  % use the first N events
    end
    % ###########################################
    for eventIter = 1:eventNum
       imgsum_ = imgsum_ + numel(annodoc.file_list{eventIter});
    end

    %(3)
    imdb.image_ids = cell(imgsum_,1);
    % self added to have a list of bboxes regardless of event folders
    tmpboxdb.image_boxes = cell(imgsum_,1); 
    %0121 added
    tmpboxdb.image_boxes_x1 = cell(imgsum_,1); 
    
    %eventNum = numel(event_list);
    cnt = 0;
    % crop the faces of the first ten events, and save the results
    %0807 changed to smaller dataset for faster training and debug
    % ###########################################
    for eventIter = 1:eventNum
       event_ = annodoc.event_list{eventIter};
       images_ = annodoc.file_list{eventIter};
       boxes_ = annodoc.face_bbx_list{eventIter};

       imageNum = numel(images_);
       for imgIter = 1:imageNum
           cnt = cnt + 1;
           %0805: can make directory irrelevant to os: linux a/b, win a\b
           imdb.image_ids{cnt} = fullfile(event_, images_{imgIter});
           tmpboxdb.image_boxes{cnt} = boxes_{imgIter};
       end
    end

    %(4)
    imdb.extension = 'jpg';
    %(5)
    imdb.flip = flip;
    %(6) -- no
    %0121: add cropping image here
    if 1
        image_at = @(i) sprintf('%s%c%s.%s', imdb.image_dir,filesep, imdb.image_ids{i}, imdb.extension);
        x1_image_at = @(i) sprintf('%s%c%s_x1.%s', imdb.image_dir,filesep, imdb.image_ids{i}, imdb.extension);
        % 1: fliplr, 2: flipud, 3: rot90 (counterclock-90), 4: rot90-lr (clockwise-90)
        if 1  %fix randi seed to make result repeatable
            rng_seed = 6;
            prev_rng = rng;
            rng(rng_seed, 'twister');
        end
        
        if 1
            rng(prev_rng);
        end
        
        for i = 1:length(imdb.image_ids)
            %if i == 35
            fprintf('Processing image %d... \n', i);
            %end
            
            % generate cropped image from x1
            %if ~exist(x1_image_at(i), 'file')
                im = imread(image_at(i));
                box_ = tmpboxdb.image_boxes{i};
                [im_crop_x1, final_bbox_x1] = cropImg_getNewbox(im, box_);
                if 0
                %show new bbox and new image
                bbs_show = final_bbox_x1;
                figure(1), clf;
                imshow(im_crop_x1);
                hold on
                if ~isempty(bbs_show)
                    %bbs_show(:, 3) = bbs_show(:, 3) - bbs_show(:, 1) + 1;
                    %bbs_show(:, 4) = bbs_show(:, 4) - bbs_show(:, 2) + 1;
                    bbApply('draw',bbs_show,'m');
                end
                hold off
                end
                %save new bbox and new image
                if 1
                tmpboxdb.image_boxes_x1{i} = final_bbox_x1;
                imwrite(im_crop_x1, x1_image_at(i));
                end
            %end
        end
        %0121: completely changed the name to '_x1'
        img_num = length(imdb.image_ids);
        image_ids = imdb.image_ids;
        imdb.image_ids(1:1:img_num) = cellfun(@(x) [x, '_x1'], image_ids, 'UniformOutput', false);
    end
    %(7)
    imdb.classes = {'face'};
    %(8)
    imdb.num_classes = 1;
    %(9)
    imdb.class_to_id = containers.Map(imdb.classes, 1:imdb.num_classes);
    %(10)
    imdb.class_ids = 1:imdb.num_classes;
    %(11) --no
    %dataset.details
    %(12)
    imdb.eval_func = @imdb_eval_voc;
    %(13)
    imdb.roidb_func = @roidb_from_wider;
    %(14)
    imdb.image_at = @(i) sprintf('%s%c%s.%s', imdb.image_dir,filesep, imdb.image_ids{i}, imdb.extension);

    %(15) height, width
    sizes = zeros(imgsum_, 2);
    for iter = 1:imgsum_
    %    event_ = traindoc.event_list{eventIter};
    %    images_ = traindoc.file_list{eventIter};
       im = imread(imdb.image_at(iter));
       sizes(iter, :) = [size(im, 1) size(im, 2)];
    end
    imdb.sizes = sizes;
    %save it
    save(cache_imdb, 'imdb', 'tmpboxdb', '-v7.3');
end

try
    load(cache_roidb); %'dataset' -- struct, containing imdb
catch
    % or can call imdb.roidb_func(imdb, tmpboxdb)
    roidb = roidb_from_wider(imdb, tmpboxdb);
    %save it
    save(cache_roidb, 'roidb', '-v7.3');
end

end  %of function

function [im_crop, final_bbox] = cropImg_getNewbox(im, box_)
    [im_hei, im_wid, ~] = size(im);
    % at least make 1 as the starting point
    h_sel_range = max(im_hei - 512 + 1, 1);
    w_sel_range = max(im_wid - 512 + 1, 1);
    compare_num = 50;
    if h_sel_range > compare_num
        h_sel_20 = randperm(h_sel_range,compare_num);
    else
        h_sel_20 = ceil(h_sel_range/2)*ones(1,compare_num);
    end
    if w_sel_range > compare_num
        w_sel_20 = randperm(w_sel_range,compare_num);
    else
        w_sel_20 = ceil(w_sel_range/2)*ones(1,compare_num);
    end
    % find the best cropping position
    % [x1 y1 x2 y2]
    box_st = round([box_(:,1) box_(:,2) box_(:,1)+box_(:,3)-1 box_(:,2)+box_(:,4)-1]);
    inside_box_num = zeros(1,compare_num);
    for kk = 1:compare_num
        sel_idx = (box_st(:,2) >= h_sel_20(kk)) & (box_st(:,4) <= h_sel_20(kk)+511) ...
                  &(box_st(:,1) >= w_sel_20(kk)) & (box_st(:,3) <= w_sel_20(kk)+511);
        inside_box_num(kk) = sum(sel_idx);
    end
    [max_val, max_idx] = max(inside_box_num);
    fprintf('The cropped image contains %d faces\n', max_val);
    y1_pos = h_sel_20(max_idx);
    y2_pos = min(h_sel_20(max_idx)+511, im_hei);
    x1_pos = w_sel_20(max_idx);
    x2_pos = min(w_sel_20(max_idx)+511, im_wid);
    im_crop = im(y1_pos:y2_pos, x1_pos:x2_pos, :);
    % padding zeros if less than 512 x 512
    if size(im_crop,1) < 512 || size(im_crop,2) < 512
        botpad = 512 - size(im_crop,1);
        rightpad = 512 - size(im_crop,2);
        im_crop = imPad(im_crop , [0 botpad 0 rightpad], 0);
    end
    % save box coords if the box center 70% in the cropped image
    inter_h = min(y2_pos, box_st(:,4)) - max(y1_pos, box_st(:,2)) + 1;
    inter_w = min(x2_pos, box_st(:,3)) - max(x1_pos, box_st(:,1)) + 1;
    s = max(inter_h,0) .* max(inter_w,0);
    box_area = box_(:,3) .* box_(:,4);
    % original face area should >= 16 (4 x 4)
    s = s(box_area >=16);
    box_area = box_area(box_area >=16);
    sel_box_idx = s./box_area >= 0.7;
    if (~isempty(sel_box_idx))&& sum(sel_box_idx)>0
        final_bbox = zeros(sum(sel_box_idx), 4);
        final_bbox(:,1) = box_(sel_box_idx, 1) - x1_pos + 1;
        final_bbox(:,2) = box_(sel_box_idx, 2) - y1_pos + 1;
        final_bbox(:,3) = box_(sel_box_idx, 3);
        final_bbox(:,4) = box_(sel_box_idx, 4);
    else
        final_bbox = [];
    end
end
% ==========sub-function: build roidb =====================
function roidb = roidb_from_wider(imdb, tmpboxdb)
    roidb.name = imdb.name;
    regions = [];
    %regions.boxes = cell(length(imdb.image_ids), 1);
    if imdb.flip
        regions.images = imdb.image_ids(1:2:end);
    else
        regions.images = imdb.image_ids;
    end
    
    if ~imdb.flip
        for i = 1:length(imdb.image_ids)
            %bboxes [col(x1) row(y1) width(x2-x1+1) height(y2-y1+1)] -->  [x1 y1 x2 y2]
            %0121 changed
            %box_ = tmpboxdb.image_boxes{i};
            box_ = tmpboxdb.image_boxes_x1{i};
            % round is because the widerfaces gt boxes are real numbers
            if ~isempty(box_)
                box_st = round([box_(:,1) box_(:,2) box_(:,1)+box_(:,3)-1 box_(:,2)+box_(:,4)-1]);
            else
                box_st = [];
            end
            
            if ~isempty(regions)
                [~, image_name1] = fileparts(imdb.image_ids{i});
                [~, image_name2] = fileparts(regions.images{i});
                assert(strcmp(image_name1, image_name2));
            end
            roidb.rois(i) = attach_proposals(box_st, imdb.sizes(i,:), imdb.class_to_id, false);
        end
    else
        for i = 1:length(imdb.image_ids)/2
            %bboxes [col(x1) row(y1) width(x2-x1+1) height(y2-y1+1)] -->  [x1 y1 x2 y2]
            box_ = imdb.image_boxes{i};
            % round is because the widerfaces gt boxes are real numbers
            box_st = round([box_(1) box_(2) box_(1)+box_(3)-1 box_(2)+box_(4)-1]);
            if ~isempty(regions)
                [~, image_name1] = fileparts(imdb.image_ids{i*2-1});
                [~, image_name2] = fileparts(regions.images{i});
                assert(strcmp(image_name1, image_name2));
                assert(imdb.flip_from(i*2) == i*2-1);
            end
            roidb.rois(i*2-1) = attach_proposals(box_st, imdb.sizes{i}, imdb.class_to_id, false);
            roidb.rois(i*2) = attach_proposals(box_st, imdb.sizes{i}, imdb.class_to_id, true);
        end
    end
end

function rec = attach_proposals(box_rec, im_size, class_to_id,  flip)
% ------------------------------------------------------------------------
    if flip
        box_rec(:, [1, 3]) = im_size(2) + 1 - box_rec(:, [3, 1]);
    end

    all_boxes = box_rec;
    gt_boxes = all_boxes;
    num_gt_boxes = size(gt_boxes, 1);

    rec.gt = true(num_gt_boxes, 1);
    rec.overlap = zeros(num_gt_boxes, class_to_id.Count, 'single');
    
    gt_classes = ones(num_gt_boxes, 1);  %all are 'face' classes
    for i = 1:num_gt_boxes
        rec.overlap(:, gt_classes(i)) = max(rec.overlap(:, gt_classes(i)), boxoverlap(all_boxes, gt_boxes(i, :)));
    end
    rec.boxes = single(all_boxes);
    rec.feat = [];
    rec.class = uint8(ones(num_gt_boxes, 1));
    
end