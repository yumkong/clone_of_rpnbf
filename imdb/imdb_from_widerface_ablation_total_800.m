function [imdb, roidb] = imdb_from_widerface_ablation_total_800(root_dir, image_set, flip, cache_dir, model_name_base, event_pool)
%function imdb_from_widerface(devkit, 'trainval', use_flip)

switch image_set
    case {'trainval'}
        data_num_str = 'allevents';
        cache_imdb = fullfile(cache_dir, sprintf('train_imdb_%s_%s',model_name_base, data_num_str));  %imdb
        cache_roidb = fullfile(cache_dir, sprintf('train_roidb_%s_%s', model_name_base, data_num_str));  %roidb
        %0205 changed
        devpath = fullfile('WIDER_train_ablation','images');
        doc_dir = fullfile('wider_face_split','wider_face_train');
        name = 'WIDERFACE_train';
    case {'test'}
        data_num_str = 'allevents';
        cache_imdb = fullfile(cache_dir, sprintf('test_imdb_%s_%s',model_name_base, data_num_str));  %imdb
        cache_roidb = fullfile(cache_dir, sprintf('test_roidb_%s_%s', model_name_base, data_num_str));  %roidb
        devpath = fullfile('WIDER_val_ablation','images');
        doc_dir = fullfile('wider_face_split','wider_face_val');
        name = 'WIDERFACE_test';
    otherwise
        error('usage = ''trainval'' or ''test''');
end

%1021 added
if flip
    cache_imdb = [cache_imdb, '_flip'];
    cache_roidb = [cache_roidb, '_flip'];
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
%     if event_num <= 0
%         eventNum = numel(annodoc.file_list);  % use all events
%     else
%         eventNum = event_num;  % use the first N events
%     end
    % ###########################################
    for eventIter = event_pool %1:eventNum
       imgsum_ = imgsum_ + numel(annodoc.file_list{eventIter});
    end

    %(3)
    imdb.image_ids = cell(imgsum_,1);
    % self added to have a list of bboxes regardless of event folders
    tmpboxdb.image_boxes = cell(imgsum_,1);
    % 0120 add image boxes for cropped images
    tmpboxdb.image_boxes_800 = cell(imgsum_,1);
    tmpboxdb.image_boxes_800_flip = cell(imgsum_,1);
    %tmpboxdb.image_boxes_flip = cell(imgsum_,1);
    
    %eventNum = numel(event_list);
    cnt = 0;
    % crop the faces of the first ten events, and save the results
    %0807 changed to smaller dataset for faster training and debug
    % ###########################################
    for eventIter = event_pool %1:eventNum
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
    %(6)  1020
    show_debug = true;
    if 1
        image_at = @(i) sprintf('%s%c%s.%s', imdb.image_dir,filesep, imdb.image_ids{i}, imdb.extension);
        image_at_800 = @(i) sprintf('%s%c%s_800.%s', imdb.image_dir,filesep, imdb.image_ids{i}, imdb.extension);
        image_at_800_flip = @(i) sprintf('%s%c%s_800_flip.%s', imdb.image_dir,filesep, imdb.image_ids{i}, imdb.extension);
        % 1: fliplr, 2: flipud, 3: rot90 (counterclock-90), 4: rot90-lr (clockwise-90)
        if 1  %fix randi seed to make result repeatable
            rng_seed = 6;
            prev_rng = rng;
            rng(rng_seed, 'twister');
        end
        %0120: use x0.5 x1 or x2?
        %flip_which_image_pool = randi(3,1,length(imdb.image_ids));
        %0309: {1,2,3,4,5} flip_lr, 6 - flipud, 7-rot90, 8-rot90_lr
        flip_type_pool = randi(8,1,length(imdb.image_ids));
        if 1
            rng(prev_rng);
        end
        
        for i = 1:length(imdb.image_ids)
            %if i == 35
            fprintf('Processing image %d... \n', i);
            %end
            % generate cropped image from x05
            if ~exist(image_at_800(i), 'file')
                im = imread(image_at(i));
                %[x1 y1 w h]
                box_ = tmpboxdb.image_boxes{i};
                [im_crop_800, final_bbox_800] = cropImg_getNewbox(im, box_);
                if show_debug
                %show new bbox and new image
                bbs_show = final_bbox_800;
                figure(1), clf;
                imshow(im_crop_800);
                hold on
                if ~isempty(bbs_show)
                    %bbs_show(:, 3) = bbs_show(:, 3) - bbs_show(:, 1) + 1;
                    %bbs_show(:, 4) = bbs_show(:, 4) - bbs_show(:, 2) + 1;
                    bbApply('draw',bbs_show,'m');
                end
                hold off
                end
                %save new bbox and new image
                if ~show_debug
                tmpboxdb.image_boxes_800{i} = final_bbox_800;
                imwrite(im_crop_800, image_at_800(i));  
                end
            end
            % 0309: decide whether x0.5/ x1/ x2
            [mval, ~] = max(box_(:,4));
            if mval >= 500
                sfactor = 0.5;
            elseif mval <= 250
                sfactor = 2;
            else
                sfactor = 1;
            end
            % generate cropped image from x1
            if ~exist(image_at_800_flip(i), 'file')
                im_ori = imread(image_at(i));
                im = imresize(im_ori, sfactor);
                box_ = tmpboxdb.image_boxes{i};
                %0309: scale the boxes
                box_(:,1) = (box_(:,1) - 1) * sfactor + 1;
                box_(:,2) = (box_(:,2) - 1) * sfactor + 1;
                box_(:,3) = box_(:,3) * sfactor;
                box_(:,4) = box_(:,4) * sfactor;
                [im_crop_800_flip, final_bbox_800_flip] = cropImg_getNewbox(im, box_);
                %0120 added
                box_rec = final_bbox_800_flip;
                im = im_crop_800_flip;
                if ~isempty(box_rec) 
                    box_ = box_rec;
                    box_rec = round([box_(:,1) box_(:,2) box_(:,1)+box_(:,3)-1 box_(:,2)+box_(:,4)-1]);
                    [hei, wid, ~] = size(im);
                    switch flip_type_pool(i)
                        case {1, 2, 3, 4, 5}
                            im_crop_800_flip = fliplr(im);
                            box_rec(:, [1, 3]) = wid + 1 - box_rec(:, [3, 1]); %width - [right left]
                        case 6
                            im_crop_800_flip = flipud(im);
                            box_rec(:, [2, 4]) = hei + 1 - box_rec(:, [4, 2]); %height - [bottom top]
                        case 7
                            im_crop_800_flip = rot90(im);
                            tmp_rec = box_rec(:, [2 1 4 3]);
                            tmp_rec(:, [2, 4]) = wid + 1 - box_rec(:, [3, 1]);
                            box_rec = tmp_rec;
                        case 8
                            im_crop_800_flip = fliplr(rot90(im));
                            box_rec = repmat([hei wid hei wid], size(box_rec, 1),1) - box_rec(:, [4 3 2 1]);
                        otherwise
                            disp('Unknown flip type.')
                    end
                else
                    switch flip_type_pool(i)
                        case {1, 2, 3, 4, 5}
                            im_crop_800_flip = fliplr(im);
                        case 6
                            im_crop_800_flip = flipud(im);
                        case 7
                            im_crop_800_flip = rot90(im);
                        case 8
                            im_crop_800_flip = fliplr(rot90(im));
                        otherwise
                            disp('Unknown flip type.')
                    end
                end
                final_bbox_800_flip = box_rec;
                % 0309 important: [x1, y1, x2, y2]->[x1, y1, w, h]
                final_bbox_800_flip(:, 3) = final_bbox_800_flip(:, 3) - final_bbox_800_flip(:, 1) + 1;
                final_bbox_800_flip(:, 4) = final_bbox_800_flip(:, 4) - final_bbox_800_flip(:, 2) + 1;
                
                if show_debug
                %show new bbox and new image
                bbs_show = final_bbox_800_flip;
                figure(2), clf;
                imshow(im_crop_800_flip);
                hold on
                if ~isempty(bbs_show)
                    %bbs_show(:, 3) = bbs_show(:, 3) - bbs_show(:, 1) + 1;
                    %bbs_show(:, 4) = bbs_show(:, 4) - bbs_show(:, 2) + 1;
                    bbApply('draw',bbs_show,'m');
                end
                hold off
                end
                %save new bbox and new image only when not debugging
                if ~show_debug
                tmpboxdb.image_boxes_800_flip{i} = final_bbox_800_flip;
                imwrite(im_crop_800_flip, image_at_800_flip(i));
                end
            end
        end
        img_num = length(imdb.image_ids)*2; %4
        image_ids = imdb.image_ids;
        %imdb.image_ids(1:4:img_num) = image_ids;
        imdb.image_ids(1:2:img_num) = cellfun(@(x) [x, '_800'], image_ids, 'UniformOutput', false);
        imdb.image_ids(2:2:img_num) = cellfun(@(x) [x, '_800_flip'], image_ids, 'UniformOutput', false);
        %imdb.image_ids(4:4:img_num) = cellfun(@(x) [x, '_flip512'], image_ids, 'UniformOutput', false);
        %imdb.flip_from = zeros(img_num, 1);
        %imdb.flip_from(4:4:img_num) = 1:4:img_num;
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
    sizes = zeros(length(imdb.image_ids), 2);
    for iter = 1:length(imdb.image_ids)
    %    event_ = traindoc.event_list{eventIter};
    %    images_ = traindoc.file_list{eventIter};
       im = imread(imdb.image_at(iter));
       sizes(iter, :) = [size(im, 1) size(im, 2)];
    end
    imdb.sizes = sizes;
    %save it
    save(cache_imdb, 'imdb','tmpboxdb','-v7.3');
end

try
    load(cache_roidb); %'dataset' -- struct, containing imdb
catch
    % or can call imdb.roidb_func(imdb, tmpboxdb)
    %1021 changed
    %roidb = roidb_from_wider(imdb, tmpboxdb);
    roidb = roidb_from_wider(imdb, tmpboxdb);
    %save it
    save(cache_roidb, 'roidb', '-v7.3');
end

end  %of function

function [im_crop, final_bbox] = cropImg_getNewbox(im, box_)
    [im_hei, im_wid, ~] = size(im);
    % at least make 1 as the starting point
    h_sel_range = max(im_hei - 800 + 1, 1);
    w_sel_range = max(im_wid - 800 + 1, 1);
    compare_num = 100; %50
    if h_sel_range > compare_num
        h_sel_20 = randperm(h_sel_range,compare_num);
    else
        %h_sel_20 = ceil(h_sel_range/2)*ones(1,compare_num);
        h_sel_20 = randi(h_sel_range,1,compare_num);
    end
    if w_sel_range > compare_num
        w_sel_20 = randperm(w_sel_range,compare_num);
    else
        %w_sel_20 = ceil(w_sel_range/2)*ones(1,compare_num);
        w_sel_20 = randi(w_sel_range,1,compare_num);
    end
    % find the best cropping position
    % [x1 y1 x2 y2]
    box_st = round([box_(:,1) box_(:,2) box_(:,1)+box_(:,3)-1 box_(:,2)+box_(:,4)-1]);
    inside_box_num = zeros(1,compare_num);
    for kk = 1:compare_num
        sel_idx = (box_st(:,2) >= h_sel_20(kk)) & (box_st(:,4) <= h_sel_20(kk)+799) ...
                  &(box_st(:,1) >= w_sel_20(kk)) & (box_st(:,3) <= w_sel_20(kk)+799);
        inside_box_num(kk) = sum(sel_idx);
    end
    [max_val, max_idx] = max(inside_box_num);
    fprintf('The cropped image contains %d faces\n', max_val);
    y1_pos = h_sel_20(max_idx);
    y2_pos = min(h_sel_20(max_idx)+799, im_hei);
    x1_pos = w_sel_20(max_idx);
    x2_pos = min(w_sel_20(max_idx)+799, im_wid);
    im_crop = im(y1_pos:y2_pos, x1_pos:x2_pos, :);
    % padding zeros if less than 512 x 512
    if size(im_crop,1) < 800 || size(im_crop,2) < 800
        botpad = 800 - size(im_crop,1);
        rightpad = 800 - size(im_crop,2);
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
    for i = 1:length(imdb.image_ids)/2  %4
        %bboxes [col(x1) row(y1) width(x2-x1+1) height(y2-y1+1)] -->  [x1 y1 x2 y2]
        box_ = tmpboxdb.image_boxes_800{i};
        % round is because the widerfaces gt boxes are real numbers
        if ~isempty(box_)
            box_800 = round([box_(:,1) box_(:,2) box_(:,1)+box_(:,3)-1 box_(:,2)+box_(:,4)-1]);
        else
            box_800 = [];
        end
        %bboxes [col(x1) row(y1) width(x2-x1+1) height(y2-y1+1)] -->  [x1 y1 x2 y2]
        box_ = tmpboxdb.image_boxes_800_flip{i};
        % round is because the widerfaces gt boxes are real numbers
        if ~isempty(box_)
            box_800_flip = round([box_(:,1) box_(:,2) box_(:,1)+box_(:,3)-1 box_(:,2)+box_(:,4)-1]);
        else
            box_800_flip = [];
        end

        roidb.rois(i*2-1) = attach_proposals(box_800, imdb.sizes(i*2-1, :), imdb.class_to_id);
        roidb.rois(i*2) = attach_proposals(box_800_flip, imdb.sizes(i*2, :), imdb.class_to_id);
    end
end

function rec = attach_proposals(box_rec, im_size, class_to_id)
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