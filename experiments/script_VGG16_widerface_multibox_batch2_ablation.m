function script_VGG16_widerface_multibox_batch2_ablation()
% script_rpn_face_VGG16_widerface_multibox_ohem()
% --------------------------------------------------------
% Yuguang Liu
% 3 layers of loss output
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
model                       = Model.VGG16_for_multibox_batch2_ablation_conv2(exp_name);
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
exp_name = 'VGG16_widerface_multibox_ablation_conv2';
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

%0805 added, make sure imdb_train and roidb_train are of cell type
if ~iscell(dataset.imdb_train)
    dataset.imdb_train = {dataset.imdb_train};
end
if ~iscell(dataset.roidb_train)
    dataset.roidb_train = {dataset.roidb_train};
end

% %% -------------------- TRAIN --------------------
% conf
conf_proposal               = proposal_config_widerface_twopath_happy_batch2_vn7('image_means', model.mean_image, ...
                                                    'feat_stride_res23', model.feat_stride_res23, ...
                                                    'feat_stride_res45', model.feat_stride_res45);
%conf_fast_rcnn              = fast_rcnn_config_widerface('image_means', model.mean_image);
% generate anchors and pre-calculate output size of rpn network 

conf_proposal.exp_name = exp_name;
%conf_fast_rcnn.exp_name = exp_name;
%[conf_proposal.anchors, conf_proposal.output_width_map, conf_proposal.output_height_map] ...
%                            = proposal_prepare_anchors(conf_proposal, model.stage1_rpn.cache_name, model.stage1_rpn.test_net_def_file);
% ###4/5### CHANGE EACH TIME*** : name of output map
output_map_name = 'output_map_twopath_happy_full';  % output_map_conv4, output_map_conv5
output_map_save_name = fullfile(cache_data_this_model_dir, output_map_name);
[conf_proposal.output_width_res23, conf_proposal.output_height_res23, ...
 conf_proposal.output_width_res45, conf_proposal.output_height_res45]...
                             = proposal_calc_output_size_twopath_happy_vn7(conf_proposal, model.stage1_rpn.test_net_def_file, output_map_save_name);
% 1209: no need to change: same with all multibox
[conf_proposal.anchors_res23,conf_proposal.anchors_res45] = proposal_generate_anchors_twopath_flip(cache_data_this_model_dir, ...
                                                            'ratios', [1.25 0.8], 'scales',  2.^[-1:4], 'add_size', [432]);  %[8 16 32 64 128 256 360 512 720 900]
%1009: from 7 to 12 anchors
%1012: from 12 to 24 anchors
%conf_proposal.anchors = proposal_generate_24anchors(cache_data_this_model_dir, 'scales', [10 16 24 32 48 64 90 128 180 256 360 512 720]);
        
%%  train
fprintf('\n***************\nstage one RPN \n***************\n');
model.stage1_rpn            = Faster_RCNN_Train.do_proposal_train_widerface_twopath_happy_batch2(conf_proposal, dataset, model.stage1_rpn, opts.do_val);

% 1020: currently do not consider test
% cache_name = 'widerface';
% method_name = 'RPN-ped';
nms_option_test = 3;
% 0129: use full-size validation images instead of 512x512
%dataset                     = Dataset.widerface_all(dataset, 'test', false, -1, cache_data_this_model_dir, model_name_base);
Faster_RCNN_Train.do_proposal_test_widerface_twopath_happy_batch2_e1_e11(conf_proposal, model.stage1_rpn, dataset.imdb_test, dataset.roidb_test, nms_option_test);

%0106 use all test set for final evaluation: dataset.imdb_realtest

%dataset                     = Dataset.widerface_all(dataset, 'realtest', false, event_num, cache_data_this_model_dir, model_name_base);
%Faster_RCNN_Train.do_proposal_test_widerface_twopath_realtest(conf_proposal, model.stage1_rpn, dataset.imdb_realtest, cache_name, method_name, nms_option_test);

end
