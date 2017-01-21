function script_res_widerface_twopath_happy_flip()
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
    opts.caffe_version          = 'caffe_faster_rcnn_win_cudnn_final'; %'caffe_faster_rcnn_win_cudnn'
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

% 1009 use more anchors
exp_name = 'Res50_widerface';

% do validation, or not 
opts.do_val                 = true; 
% model
model                       = Model.Res101_for_rpn_widerface_twopath_happy_flip(exp_name);
% cache base
cache_base_proposal         = 'rpn_widerface_Res101';
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
model_name_base = 'res101_twopath';  % ZF, vgg16_conv5
%1009 change exp here for output
exp_name = 'Res16_widerface_twopath_happy_flip';
% the dir holding intermediate data paticular
cache_data_this_model_dir = fullfile(cache_data_root, exp_name, 'rpn_cachedir');
mkdir_if_missing(cache_data_this_model_dir);
use_flipped                 = true;  %true --> false
event_num                   = -1; %11
dataset                     = Dataset.widerface_all_flip_512(dataset, 'train', use_flipped, event_num, cache_data_this_model_dir, model_name_base);
%dataset                     = Dataset.widerface_all(dataset, 'test', false, event_num, cache_data_this_model_dir, model_name_base);
%0106 added all test images
dataset                     = Dataset.widerface_all_512(dataset, 'test', false, event_num, cache_data_this_model_dir, model_name_base);

% train_sel_idx_name = fullfile(cache_data_this_model_dir, 'sel_idx.mat');
% try
%     %load('output\train_roidb_event123.mat');
%     load(train_sel_idx_name);
% catch
%     example_num = length(dataset.imdb_train.image_ids);
%     half_example_num = example_num/2; %12880
%     % only select half of the flipped image for memory efficiency
%     %tmp_idx = round(rand([half_example_num,1]));
%     tmp_idx = (rand([half_example_num,1])>=0.8);  %1/3 are 1, 2/3 are 0
%     sel_idx = ones(example_num, 1); % all original images are set as 1
%     sel_idx(2:2:end) = tmp_idx;  % flipped images are randomly set
%     
%     test_num = length(dataset.imdb_test.image_ids);
%     if test_num > 500
%         sel_val_idx = randperm(test_num, 500);
%     else
%         sel_val_idx = 1:test_num;
%     end
%     sel_val_idx = sel_val_idx';
%     save(train_sel_idx_name, 'sel_idx', 'sel_val_idx');
% end
% fprintf('Total training image is %d\n', sum(sel_idx));
% fprintf('Total test image is %d\n', length(sel_val_idx));
% % randomly select flipped train
% sel_idx = logical(sel_idx);
% dataset.imdb_train.image_ids = dataset.imdb_train.image_ids(sel_idx,:);
% dataset.imdb_train.flip_from = dataset.imdb_train.flip_from(sel_idx,:);
% dataset.imdb_train.sizes = dataset.imdb_train.sizes(sel_idx,:);
% dataset.roidb_train.rois = dataset.roidb_train.rois(:, sel_idx);
% % randomly select test 
% dataset.imdb_test.image_ids = dataset.imdb_test.image_ids(sel_val_idx,:);
% %dataset.imdb_test.flip_from = dataset.imdb_test.flip_from(sel_val_idx,:);
% dataset.imdb_test.sizes = dataset.imdb_test.sizes(sel_val_idx,:);
% dataset.roidb_test.rois = dataset.roidb_test.rois(:, sel_val_idx);

%0805 added, make sure imdb_train and roidb_train are of cell type
if ~iscell(dataset.imdb_train)
    dataset.imdb_train = {dataset.imdb_train};
end
if ~iscell(dataset.roidb_train)
    dataset.roidb_train = {dataset.roidb_train};
end

% %% -------------------- TRAIN --------------------
% conf
conf_proposal               = proposal_config_widerface_twopath_happy('image_means', model.mean_image, ...
                                                    'feat_stride_res23', model.feat_stride_res23, ...
                                                    'feat_stride_res45', model.feat_stride_res45);
%conf_fast_rcnn              = fast_rcnn_config_widerface('image_means', model.mean_image);
% generate anchors and pre-calculate output size of rpn network 

conf_proposal.exp_name = exp_name;
%conf_fast_rcnn.exp_name = exp_name;
%[conf_proposal.anchors, conf_proposal.output_width_map, conf_proposal.output_height_map] ...
%                            = proposal_prepare_anchors(conf_proposal, model.stage1_rpn.cache_name, model.stage1_rpn.test_net_def_file);
% ###4/5### CHANGE EACH TIME*** : name of output map
output_map_name = 'output_map_twopath_happy';  % output_map_conv4, output_map_conv5
output_map_save_name = fullfile(cache_data_this_model_dir, output_map_name);
[conf_proposal.output_width_res23, conf_proposal.output_height_res23, ...
 conf_proposal.output_width_res45, conf_proposal.output_height_res45]...
                             = proposal_calc_output_size_twopath_happy(conf_proposal, model.stage1_rpn.test_net_def_file, output_map_save_name);
% 1209: no need to change: same with all multibox
[conf_proposal.anchors_res23,conf_proposal.anchors_res45] = proposal_generate_anchors_twopath_flip(cache_data_this_model_dir, ...
                                                            'ratios', [1.25 0.8], 'scales',  2.^[-1:4], 'add_size', [432]);  %[8 16 32 64 128 256 360 512 720 900]
%1009: from 7 to 12 anchors
%1012: from 12 to 24 anchors
%conf_proposal.anchors = proposal_generate_24anchors(cache_data_this_model_dir, 'scales', [10 16 24 32 48 64 90 128 180 256 360 512 720]);
        
%%  train
fprintf('\n***************\nstage one RPN \n***************\n');
model.stage1_rpn            = Faster_RCNN_Train.do_proposal_train_widerface_twopath_happy(conf_proposal, dataset, model.stage1_rpn, opts.do_val);

% 1020: currently do not consider test
% cache_name = 'widerface';
% method_name = 'RPN-ped';
% nms_option_test = 3;
% %0101: use all validation set instead of 500
% dataset                     = Dataset.widerface_all(dataset, 'test', false, event_num, cache_data_this_model_dir, model_name_base);
% %0106 use all test set for final evaluation: dataset.imdb_realtest
% %dataset                     = Dataset.widerface_all(dataset, 'realtest', false, event_num, cache_data_this_model_dir, model_name_base);
% Faster_RCNN_Train.do_proposal_test_widerface_multibox_ohem_happy_flip(conf_proposal, model.stage1_rpn, dataset.imdb_test, dataset.roidb_test, cache_name, method_name, nms_option_test);
% %Faster_RCNN_Train.do_proposal_test_widerface_multibox_realtest(conf_proposal, model.stage1_rpn, dataset.imdb_realtest, cache_name, method_name, nms_option_test);

end
