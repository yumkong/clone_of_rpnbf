function script_rpn_face_VGG16_widerface_conv4_yolo()
% script_rpn_face_VGG16_widerface_conv4()
% --------------------------------------------------------
% RPN_BF
% Copyright (c) 2016, Liliang Zhang
% Licensed under TByrhe MIT License [see LICENSE for details]
% --------------------------------------------------------

% ********** liu@1001: run this at root directory !!! *******************************

clc;
clear mex;
clear is_valid_handle; % to clear init_key
run(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'startup'));
%% -------------------- CONFIG --------------------
%0930 change caffe folder according to platform
if ispc
    opts.caffe_version          = 'caffe_faster_rcnn_win_cudnn';
    cd('D:\\RPN_BF_master');
elseif isunix
    opts.caffe_version          = 'caffe_faster_rcnn';
    cd('/usr/local/data/yuguang/git_all/RPN_BF_pedestrain/RPN_BF-RPN-pedestrian');
end
opts.gpu_id                 = auto_select_gpu;
active_caffe_mex(opts.gpu_id, opts.caffe_version);

% 1009 use more anchors
exp_name = 'VGG16_widerface';

% do validation, or not 
opts.do_val                 = true; 

% cache base
cache_base_proposal         = 'yolo_widerface_VGG16';
% model
model                       = Model.VGG16_for_rpn_widerface_conv4_yolo(exp_name, cache_base_proposal);

%cache_base_fast_rcnn        = '';
% set cache folder for each stage
%model                       = Faster_RCNN_Train.set_cache_folder_widerface(cache_base_proposal,cache_base_fast_rcnn, model);
% train/test data
dataset                     = [];
% the root directory to hold any useful intermediate data during training process
cache_data_root = 'output';  %cache_data
mkdir_if_missing(cache_data_root);
% ###3/5### CHANGE EACH TIME*** use this to name intermediate data's mat files
model_name_base = 'vgg16_conv4';  % ZF, vgg16_conv5
%1009 change exp here for output
exp_name = 'VGG16_widerface_conv4_yolo'; %VGG16_widerface_twelve_anchors
% the dir holding intermediate data paticular
cache_data_this_model_dir = fullfile(cache_data_root, exp_name, 'rpn_cachedir');
mkdir_if_missing(cache_data_this_model_dir);
use_flipped                 = false;  %true --> false
event_num                   = 11; %3
dataset                     = Dataset.widerface_all(dataset, 'train', use_flipped, event_num, cache_data_this_model_dir, model_name_base);
dataset                     = Dataset.widerface_all(dataset, 'test', false, event_num, cache_data_this_model_dir, model_name_base);

%0805 added, make sure imdb_train and roidb_train are of cell type
if ~iscell(dataset.imdb_train)
    dataset.imdb_train = {dataset.imdb_train};
end
if ~iscell(dataset.roidb_train)
    dataset.roidb_train = {dataset.roidb_train};
end

% %% -------------------- TRAIN --------------------
% conf
conf_proposal               = proposal_config_widerface_yolo('image_means', model.mean_image, 'feat_stride', model.feat_stride);
%conf_fast_rcnn              = fast_rcnn_config_widerface('image_means', model.mean_image);
% generate anchors and pre-calculate output size of rpn network 

conf_proposal.exp_name = exp_name;
%conf_fast_rcnn.exp_name = exp_name;
%[conf_proposal.anchors, conf_proposal.output_width_map, conf_proposal.output_height_map] ...
%                            = proposal_prepare_anchors(conf_proposal, model.stage1_rpn.cache_name, model.stage1_rpn.test_net_def_file);
% ###4/5### CHANGE EACH TIME*** : name of output map
output_map_name = 'output_map_conv4';  % output_map_conv4, output_map_conv5
output_map_save_name = fullfile(cache_data_this_model_dir, output_map_name);
[conf_proposal.output_width_map, conf_proposal.output_height_map] = proposal_calc_output_size(conf_proposal, ...
                                                                    model.stage1_rpn.test_net_def_file, output_map_save_name);
conf_proposal.anchors = proposal_generate_anchors(cache_data_this_model_dir, 'base_size', 8, 'ratios', [1], 'scales', [1]);  %scales: 2.^[-1:5]
%1009: from 7 to 12 anchors
%1012: from 12 to 24 anchors
%conf_proposal.anchors = proposal_generate_24anchors(cache_data_this_model_dir, 'scales', [10 16 24 32 48 64 90 128 180 256 360 512 720]);
        
%%  train
fprintf('\n***************\nstage one RPN \n***************\n');
model.stage1_rpn            = Faster_RCNN_Train.do_proposal_train_widerface_yolo(conf_proposal, dataset, model.stage1_rpn, opts.do_val);

%% test
% get predicted rois (bboxes) to be used in later stages
%nms_option_train = 0;
%nms_option_test = 3;
%dataset.roidb_train         = cellfun(@(x, y) Faster_RCNN_Train.do_proposal_test_my(conf_proposal, model.stage1_rpn, x, y, nms_option_train), dataset.imdb_train, dataset.roidb_train, 'UniformOutput', false);
%dataset.roidb_test       	= Faster_RCNN_Train.do_proposal_test_my(conf_proposal, model.stage1_rpn, dataset.imdb_test, dataset.roidb_test, nms_option_test);
cache_name = 'widerface';
method_name = 'RPN-ped';
nms_option_test = 3;
Faster_RCNN_Train.do_proposal_test_widerface_yolo(conf_proposal, model.stage1_rpn, dataset.imdb_test, dataset.roidb_test, cache_name, method_name, nms_option_test);

% %%  stage one fast rcnn
% fprintf('\n***************\nstage one fast rcnn\n***************\n');
% % train
% model.stage1_fast_rcnn      = Faster_RCNN_Train.do_fast_rcnn_train_widerface(conf_fast_rcnn, dataset, model.stage1_fast_rcnn, opts.do_val);
% % test
% opts.mAP                    = Faster_RCNN_Train.do_fast_rcnn_test_widerface(conf_fast_rcnn, model.stage1_fast_rcnn, dataset.imdb_test, dataset.roidb_test);
%  
% %%  stage two proposal
% % net proposal
% fprintf('\n***************\nstage two proposal\n***************\n');
% % train
% model.stage2_rpn.init_net_file = model.stage1_fast_rcnn.output_model_file;
% model.stage2_rpn            = Faster_RCNN_Train.do_proposal_train_widerface(conf_proposal, dataset, model.stage2_rpn, opts.do_val);
% % test
% dataset.roidb_train        	= cellfun(@(x, y) Faster_RCNN_Train.do_proposal_test_my(conf_proposal, model.stage2_rpn, x, y, nms_option_train), dataset.imdb_train, dataset.roidb_train, 'UniformOutput', false);
% dataset.roidb_test         	= Faster_RCNN_Train.do_proposal_test_my(conf_proposal, model.stage2_rpn, dataset.imdb_test, dataset.roidb_test, nms_option_test);
% 
% %%  stage two fast rcnn
% fprintf('\n***************\nstage two fast rcnn\n***************\n');
% % train
% model.stage2_fast_rcnn.init_net_file = model.stage1_fast_rcnn.output_model_file;
% model.stage2_fast_rcnn      = Faster_RCNN_Train.do_fast_rcnn_train_widerface(conf_fast_rcnn, dataset, model.stage2_fast_rcnn, opts.do_val);
% 
% %% final test
% fprintf('\n***************\nfinal test\n***************\n');
%      
% model.stage2_rpn.nms        = model.final_test.nms;
% dataset.roidb_test       	= Faster_RCNN_Train.do_proposal_test_my(conf_proposal, model.stage2_rpn, dataset.imdb_test, dataset.roidb_test, nms_option_test);
% opts.final_mAP              = Faster_RCNN_Train.do_fast_rcnn_test_widerface(conf_fast_rcnn, model.stage2_fast_rcnn, dataset.imdb_test, dataset.roidb_test);


end

% function [anchors, output_width_map, output_height_map] = proposal_prepare_anchors(conf, cache_name, test_net_def_file)
%     [output_width_map, output_height_map] ...                           
%                                 = proposal_calc_output_size(conf, test_net_def_file);
%     anchors                = proposal_generate_anchors(cache_name, ...
%                                     'scales',  2.^[-1:5]);%0820:2.^[3:5] -->  2.^[-1:5]
% end
