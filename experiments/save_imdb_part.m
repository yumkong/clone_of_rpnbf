function save_imdb_part()

% load two structs: imdb, tmpboxdb and (maybe flip_type_pool => no use)
% ****** should set differently at each time
load('/usr/local/data/yuguang/git_all/RPN_BF_pedestrain/RPN_BF-RPN-pedestrian/output/Res16_widerface_twopath_happy_flip/rpn_cachedir/test_imdb_res101_twopath_all_raw.mat');
% ****** should set differently each time
part_num = 696; %11120;

imdbnew = imdb;
if 0
    imdbnew.image_ids = imdbnew.image_ids(1:part_num);
    imdbnew.flip_from = imdbnew.flip_from(1:part_num,:);
    imdbnew.sizes = imdbnew.sizes(1:part_num,:);
else
    imdbnew.image_ids = imdbnew.image_ids(1:part_num);
    %imdbnew.flip_from = imdbnew.flip_from(1:part_num,:);
    imdbnew.sizes = imdbnew.sizes(1:part_num,:);
end
tmpboxdbnew = tmpboxdb;
if 0
    tmpboxdbnew.image_boxes = tmpboxdbnew.image_boxes(1:part_num);
    tmpboxdbnew.image_boxes_x05 = tmpboxdbnew.image_boxes_x05(1:part_num);
    tmpboxdbnew.image_boxes_x1 = tmpboxdbnew.image_boxes_x1(1:part_num);
    tmpboxdbnew.image_boxes_x2 = tmpboxdbnew.image_boxes_x2(1:part_num);
    tmpboxdbnew.image_boxes_flip = tmpboxdbnew.image_boxes_flip(1:part_num);
else
    tmpboxdbnew.image_boxes = tmpboxdbnew.image_boxes(1:part_num);
    tmpboxdbnew.image_boxes_x1 = tmpboxdbnew.image_boxes_x1(1:part_num);
end
clear imdb tmpboxdb

imdb = imdbnew;
tmpboxdb = tmpboxdbnew;
% reset this functional handle
imdb.image_at = @(i) sprintf('%s%c%s.%s', imdb.image_dir,filesep, imdb.image_ids{i}, imdb.extension);

% ****** should set differently each time
save('test_imdb_res101_twopath_e1-e11_raw.mat', 'imdb','tmpboxdb');

if 0
    % verify by two images (### note that this only applies for training set)
    % 1
    figure(1),imshow(imdb.image_at(160)); %a random number of 4k (<= part_num)
    bbs_show = tmpboxdb.image_boxes_flip{40}; % accordingly: k
    % should do this for flip
    bbs_show(:, 3) = bbs_show(:, 3) - bbs_show(:, 1) + 1;
    bbs_show(:, 4) = bbs_show(:, 4) - bbs_show(:, 2) + 1;
    bbApply('draw',bbs_show,'m');
    %2
    figure(2),imshow(imdb.image_at(666)); %a random number of 4k+2 (<= part_num)
    bbs_show = tmpboxdb.image_boxes_flip{167}; % accordingly: k+1
    % should not do this for non-flip
    %bbs_show(:, 3) = bbs_show(:, 3) - bbs_show(:, 1) + 1;
    %bbs_show(:, 4) = bbs_show(:, 4) - bbs_show(:, 2) + 1;
    bbApply('draw',bbs_show,'m');
else
    % verify by 1 images (### note that this applies for test set)
    figure(1),imshow(imdb.image_at(160)); %a random number of 4k (<= part_num)
    bbs_show = tmpboxdb.image_boxes_x1{160}; % accordingly: k
    bbApply('draw',bbs_show,'m');
end
end