function script_VGG16_widerface_multibox_ablation_fastrcnn_singleconv()
% script_rpn_face_VGG16_widerface_multibox_ohem()
% --------------------------------------------------------
% Yuguang Liu
% 3 layers of loss output
% timing at 0304: batch2 512x512 image: 3434M, just ok with vn7
% --------------------------------------------------------

% ********** liu@1001: run this at root directory !!! *******************************

clc;
clear mex;
clear is_valid_handle; % to clear init_key
run(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'startup'));
%% -------------------- CONFIG --------------------
%0930 change caffe folder according to platform
if ispc
    opts.caffe_version          = 'caffe_faster_rcnn_win_cudnn_bn'; %'caffe_faster_rcnn_win_cudnn_final'
    cd('D:\\RPN_BF_master');
elseif isunix
    % caffe_faster_rcnn_rfcn is from caffe-rfcn-r-fcn_othersoft
    % caffe_faster_rcnn_rfcn_normlayer is also from
    % caffe-rfcn-r-fcn_othersoft with l2-normalization layer added
    opts.caffe_version          ='caffe_faster_rcnn_bn'; %'caffe_faster_rcnn_dilate_ohem';
    cd('/usr/local/data/yuguang/git_all/RPN_BF_pedestrain/RPN_BF-RPN-pedestrian');
end
opts.gpu_id                 = auto_select_gpu;
active_caffe_mex(opts.gpu_id, opts.caffe_version);

%0120 to use bbApply('draw',bbs_show,'m') in widerface_all_flip_512
addpath(genpath('external/toolbox'));  % piotr's image and video toolbox
addpath(genpath('external/export_fig'));  % save image to png

% 1009 use more anchors
exp_name = 'VGG16_widerface';

% do validation, or not 
opts.do_val                 = true; 
% model
model                       = Model.VGG16_for_multibox_ablation_fastrcnn_singleconv_0401(exp_name);
% cache base
cache_base_proposal         = 'rpn_widerface_VGG16';
%cache_base_fast_rcnn        = '';
% set cache folder for each stage
%model                       = Faster_RCNN_Train.set_cache_folder_widerface(cache_base_proposal,cache_base_fast_rcnn, model);
model                       = Faster_RCNN_Train.set_cache_folder_widerface(cache_base_proposal, model);
% train/test data
dataset                     = [];
% the root directory to hold any useful intermediate data during training process
cache_data_root = 'output';  %cache_data
mkdir_if_missing(cache_data_root);
% ###3/5### CHANGE EACH TIME*** use this to name intermediate data's mat files
model_name_base = 'VGG16_multibox_ablation';  % ZF, vgg16_conv5
%1009 change exp here for output
exp_name = 'VGG16_widerface_multibox_ablation_final_fastrcnn';
% the dir holding intermediate data paticular
cache_data_this_model_dir = fullfile(cache_data_root, exp_name, 'rpn_cachedir');
mkdir_if_missing(cache_data_this_model_dir);
use_flipped                 = false;  %true --> false
% 0127: in vn7 only use 11 event for demo
train_event_pool            = [1 61 3 5 6 9 11 12 14 33 37 38 45 51 56]; %-1
dataset                     = Dataset.widerface_ablation_512(dataset, 'train', use_flipped, train_event_pool, cache_data_this_model_dir, model_name_base);
%dataset                     = Dataset.widerface_all(dataset, 'test', false, event_num, cache_data_this_model_dir, model_name_base);
%0106 added all test images
test_event_pool             = 1:61;
dataset                     = Dataset.widerface_ablation_512(dataset, 'test', false, test_event_pool, cache_data_this_model_dir, model_name_base);
% 0206 added: adapt dataset created in puck to VN7
if ispc
    devkit = 'D:\\datasets\\WIDERFACE';
    %train
    dataset.imdb_train.image_dir = fullfile(devkit, 'WIDER_train_ablation', 'images');
    dataset.imdb_train.image_ids = cellfun(@(x) strrep(x,'/',filesep), dataset.imdb_train.image_ids, 'UniformOutput', false);
    dataset.imdb_train.image_at = @(i) sprintf('%s%c%s.%s', dataset.imdb_train.image_dir, filesep, dataset.imdb_train.image_ids{i}, dataset.imdb_train.extension);
    %val
    dataset.imdb_test.image_dir = fullfile(devkit, 'WIDER_val_ablation', 'images');
    dataset.imdb_test.image_ids = cellfun(@(x) strrep(x,'/',filesep), dataset.imdb_test.image_ids, 'UniformOutput', false);
    dataset.imdb_test.image_at = @(i) sprintf('%s%c%s.%s', dataset.imdb_test.image_dir, filesep, dataset.imdb_test.image_ids{i}, dataset.imdb_test.extension);
    %verify
    if 0
        %---train-----------
        im = imread(dataset.imdb_train.image_at(666));
        figure(1),imshow(im)
        box = dataset.roidb_train.rois(666).boxes;
        box(:,3) = box(:,3) - box(:,1) +1;
        box(:,4) = box(:,4) - box(:,2) +1;
        bbApply('draw', box, 'm');
        %----test---------
        im = imread(dataset.imdb_test.image_at(666));
        figure(2),imshow(im)
        box = dataset.roidb_test.rois(666).boxes;
        box(:,3) = box(:,3) - box(:,1) +1;
        box(:,4) = box(:,4) - box(:,2) +1;
        bbApply('draw', box, 'm');
    end
end
%0805 added, make sure imdb_train and roidb_train are of cell type
if ~iscell(dataset.imdb_train)
    dataset.imdb_train = {dataset.imdb_train};
end
if ~iscell(dataset.roidb_train)
    dataset.roidb_train = {dataset.roidb_train};
end

% %% -------------------- TRAIN --------------------
% conf
conf_proposal          = proposal_config_widerface_ablation_final('image_means', model.mean_image, 'feat_stride_s4', model.feat_stride_s4,...
                                                                    'feat_stride_s8', model.feat_stride_s8, 'feat_stride_s16', model.feat_stride_s16);
conf_fast_rcnn              = fast_rcnn_config_widerface_batch2('image_means', model.mean_image);
% generate anchors and pre-calculate output size of rpn network 

conf_proposal.exp_name = exp_name;
conf_fast_rcnn.exp_name = exp_name;
%[conf_proposal.anchors, conf_proposal.output_width_map, conf_proposal.output_height_map] ...
%                            = proposal_prepare_anchors(conf_proposal, model.stage1_rpn.cache_name, model.stage1_rpn.test_net_def_file);
% ###4/5### CHANGE EACH TIME*** : name of output map
output_map_name = 'output_map_conv2';  % output_map_conv4, output_map_conv5
output_map_save_name = fullfile(cache_data_this_model_dir, output_map_name);
[conf_proposal.output_width_s4, conf_proposal.output_height_s4, ...
    conf_proposal.output_width_s8, conf_proposal.output_height_s8, ...
    conf_proposal.output_width_s16, conf_proposal.output_height_s16] = proposal_calc_output_size_ablation_final(conf_proposal, ...
                                                                    model.stage1_rpn.test_net_def_file, output_map_save_name);
% 1209: no need to change: same with all multibox
[conf_proposal.anchors_s4,conf_proposal.anchors_s8, conf_proposal.anchors_s16] = proposal_generate_anchors_ablation_final(cache_data_this_model_dir, ...
                         'ratios', [1], 'scales',  2.^[-1:4], 'add_size', [480]);  %[8 16 32 64 128 256 480]
%1009: from 7 to 12 anchors
%1012: from 12 to 24 anchors
%conf_proposal.anchors = proposal_generate_24anchors(cache_data_this_model_dir, 'scales', [10 16 24 32 48 64 90 128 180 256 360 512 720]);
        
%%  train
fprintf('\n***************\nstage one RPN \n***************\n');
model.stage1_rpn            = Faster_RCNN_Train.do_proposal_train_widerface_ablation_final(conf_proposal, dataset, model.stage1_rpn, opts.do_val);

% 1020: currently do not consider test
nms_option_test = 3;
% 0129: use full-size validation images instead of 512x512
%dataset                     = Dataset.widerface_all(dataset, 'test', false, -1, cache_data_this_model_dir, model_name_base);
% 1207: use rpn's result to update roidb_train and roidb_test
dataset.roidb_train         = cellfun(@(x, y) Faster_RCNN_Train.do_proposal_test_widerface_ablation_fastrcnn(conf_proposal, model.stage1_rpn, x, y), ...
                                                                            dataset.imdb_train, dataset.roidb_train, 'UniformOutput', false);
dataset.roidb_test = Faster_RCNN_Train.do_proposal_test_widerface_ablation_fastrcnn(conf_proposal, model.stage1_rpn, dataset.imdb_test, dataset.roidb_test, nms_option_test);

% liu@0816 masked --> not necessary currently
%%  fast rcnn train
fprintf('\n***************\nstage one fast rcnn\n***************\n');
% train
%shared
model.stage1_fast_rcnn.init_net_file = model.stage1_rpn.output_model_file; % init with trained rpn model
%unshared
%0106 use all test set for final evaluation: dataset.imdb_realtest
%0125 added: training with score feat map
model.stage1_fast_rcnn      = Faster_RCNN_Train.do_fast_rcnn_train_widerface_ablation_fastrcnn(conf_fast_rcnn, dataset, model.stage1_fast_rcnn, opts.do_val);
% test
Faster_RCNN_Train.do_fast_rcnn_test_widerface_ablation_fastrcnn(conf_fast_rcnn, model.stage1_fast_rcnn, dataset.imdb_test, dataset.roidb_test);
end
