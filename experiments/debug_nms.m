function debug_nms()
clear
clc

% add all necessary path
run(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'startup'));
addpath(genpath('external/toolbox'));  % piotr's image and video toolbox
dataset                     = [];
% the root directory to hold any useful intermediate data during training process
cache_data_root = 'output';  %cache_data
mkdir_if_missing(cache_data_root);
% ###3/5### CHANGE EACH TIME*** use this to name intermediate data's mat files
model_name_base = 'vgg16_conv4';  % ZF, vgg16_conv5
%1009 change exp here for output
exp_name = 'VGG16_widerface_conv4'; %VGG16_widerface_twelve_anchors
% the dir holding intermediate data paticular
cache_data_this_model_dir = fullfile(cache_data_root, exp_name, 'rpn_cachedir');
mkdir_if_missing(cache_data_this_model_dir);

event_num                   = 11; %3
dataset                     = Dataset.widerface_all(dataset, 'test', false, event_num, cache_data_this_model_dir, model_name_base);
imdb = dataset.imdb_test;
%1018 added for windows
%if image path is of unix format but in windows platform, replace it with windows path
if ispc && ~isempty(strfind(imdb.image_at(1), '/'))  %unix path must have '/'
    %test set
    imdb.image_dir = 'D:\\datasets\\WIDERFACE';
    for k = 1:length(imdb.image_ids)
        imdb.image_ids{k} = strrep(imdb.image_ids{k}, '/', '\\'); % '/' --> '\'
    end
    test_root = fullfile(imdb.image_dir, 'WIDER_val','images');
    imdb.image_at = @(i) sprintf('%s\\%s.%s', test_root, imdb.image_ids{i}, imdb.extension); 
end

if ispc
    cd('D:\\RPN_BF_master');
elseif isunix
    cd('/usr/local/data/yuguang/git_all/RPN_BF_pedestrain/RPN_BF-RPN-pedestrian');
end

load('rpn_aboxes.mat'); %'aboxes'
load('bf_aboxes.mat'); %'bf_aboxes'
nms_aboxes = cell(length(aboxes), 1);
nms_option = 3;
show_image = true;
for i = 1:length(aboxes)
        
        % rpn boxes
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
        
        % bf boxes        
        if show_image
            %draw boxes after 'smart' NMS
            bbs = bf_aboxes{i};
            if ~isempty(bbs)
              bbs(:, 3) = bbs(:, 3) - bbs(:, 1) + 1;
              bbs(:, 4) = bbs(:, 4) - bbs(:, 2) + 1;
              %I=imread(imgNms{i});
              figure(2); 
              im(img);  %im(I)
              bbApply('draw',bbs);
            end
        end
        
        time = tic;
        nms_aboxes{i} = pseudoNMS_v6(bf_aboxes{i}, nms_option);
        
        fprintf('PseudoNMS for image %d cost %.1f seconds\n', i, toc(time));
        if show_image
            %draw boxes after 'smart' NMS
            bbs = nms_aboxes{i};
            if ~isempty(bbs)
              bbs(:, 3) = bbs(:, 3) - bbs(:, 1) + 1;
              bbs(:, 4) = bbs(:, 4) - bbs(:, 2) + 1;
              %I=imread(imgNms{i});
              figure(3); 
              im(img);  %im(I)
              bbApply('draw',bbs);
            end
        end
end
    
end