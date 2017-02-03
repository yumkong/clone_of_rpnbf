function script_rpn_bf_face_VGG16_widerface_multibox_nobf()
%function script_rpn_bf_face_VGG16_widerface_multibox_4x4_puck_3scale(start_num)

clc;
clear mex;
clear is_valid_handle; % to clear init_key
run(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'startup'));

%0929 added: switch from '$root/experiments' to '$root', to run on puck with matlab -nojvm
%cd('..');  %
%cd('/usr/local/data/yuguang/git_all/RPN_BF_pedestrain/RPN_BF-RPN-pedestrian');
%% -------------------- CONFIG --------------------
%0930 change caffe folder according to platform
if ispc
    opts.caffe_version          = 'caffe_faster_rcnn_win_cudnn_final'; % 'caffe_faster_rcnn_win_cudnn_dilate'; 
    cd('D:\\RPN_BF_master');
elseif isunix
    % caffe_faster_rcnn_rfcn is from caffe-rfcn-r-fcn_othersoft
    % caffe_faster_rcnn_rfcn_normlayer is also from
    % caffe-rfcn-r-fcn_othersoft with l2-normalization layer added
    opts.caffe_version          = 'caffe_faster_rcnn_dilate_ohem'; %'caffe_faster_rcnn_dilate'; 
    cd('/usr/local/data/yuguang/git_all/RPN_BF_pedestrain/RPN_BF-RPN-pedestrian');
end
opts.gpu_id                 = auto_select_gpu;
active_caffe_mex(opts.gpu_id, opts.caffe_version);

% 1009 path to the prototxt files which define the network
exp_name = 'VGG16_widerface';

% do validation, or not 
opts.do_val                 = true; 
% model
model                       = Model.VGG16_for_rpn_widerface_multibox_flip_3scale(exp_name);
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
model_name_base = 'vgg16_multibox';  % ZF, vgg16_conv5
%1009 change exp here for output
if ispc
    exp_name = 'VGG16_widerface_multibox_ohem_happy_flip';
else
    exp_name = 'VGG16_widerface_multibox_ohem_happy_flip';
end
% the dir holding intermediate data paticular
cache_data_this_model_dir = fullfile(cache_data_root, exp_name, 'rpn_cachedir');
mkdir_if_missing(cache_data_this_model_dir);
use_flipped                 = true;  %true --> false
event_num                   = -1; %11
event_num_test              = -1;  %1007 added: test all val images
%dataset                     = Dataset.widerface_all(dataset, 'train', use_flipped, event_num, cache_data_this_model_dir, model_name_base);
dataset                     = Dataset.widerface_all_flip(dataset, 'train', use_flipped, event_num, cache_data_this_model_dir, model_name_base);
dataset                     = Dataset.widerface_all(dataset, 'test', false, event_num_test, cache_data_this_model_dir, model_name_base);

train_sel_idx_name = fullfile(cache_data_this_model_dir, 'sel_idx.mat');
try
    %load('output\train_roidb_event123.mat');
    load(train_sel_idx_name);
catch
    example_num = length(dataset.imdb_train.image_ids);
    half_example_num = example_num/2; %12880
    % only select half of the flipped image for memory efficiency
    %tmp_idx = round(rand([half_example_num,1]));
    tmp_idx = (rand([half_example_num,1])>=0.8);  %1/3 are 1, 2/3 are 0
    sel_idx = ones(example_num, 1); % all original images are set as 1
    sel_idx(2:2:end) = tmp_idx;  % flipped images are randomly set
    
    test_num = length(dataset.imdb_test.image_ids);
    if test_num > 500
        sel_val_idx = randperm(test_num, 500);
    else
        sel_val_idx = 1:test_num;
    end
    sel_val_idx = sel_val_idx';
    save(train_sel_idx_name, 'sel_idx', 'sel_val_idx');
end
fprintf('Total training image is %d\n', sum(sel_idx));
fprintf('Total test image is %d\n', length(sel_val_idx));
% randomly select flipped train
sel_idx = logical(sel_idx);
dataset.imdb_train.image_ids = dataset.imdb_train.image_ids(sel_idx,:);
dataset.imdb_train.flip_from = dataset.imdb_train.flip_from(sel_idx,:);
dataset.imdb_train.sizes = dataset.imdb_train.sizes(sel_idx,:);
% 1227 fix a bug here: the inconsistency between image path and labels
dataset.imdb_train.image_at = @(i)sprintf('%s%c%s.%s',dataset.imdb_train.image_dir,filesep, dataset.imdb_train.image_ids{i},dataset.imdb_train.extension);
dataset.roidb_train.rois = dataset.roidb_train.rois(:, sel_idx);
% 1227: keep all test set

%0805 added, make sure imdb_train and roidb_train are of cell type
if ~iscell(dataset.imdb_train)
    dataset.imdb_train = {dataset.imdb_train};
end
if ~iscell(dataset.roidb_train)
    dataset.roidb_train = {dataset.roidb_train};
end

% %% -------------------- TRAIN --------------------
% conf
conf_proposal               = proposal_config_widerface_multibox_ohem_happy('image_means', model.mean_image, ...
                                                    'feat_stride_conv34', model.feat_stride_conv34, ...
                                                    'feat_stride_conv5', model.feat_stride_conv5, ...
                                                    'feat_stride_conv6', model.feat_stride_conv6 );
%conf_fast_rcnn              = fast_rcnn_config_widerface('image_means', model.mean_image);
% generate anchors and pre-calculate output size of rpn network 

conf_proposal.exp_name = exp_name;
% [conf_proposal.anchors, conf_proposal.output_width_map, conf_proposal.output_height_map] ...
%                             = proposal_prepare_anchors(conf_proposal, model.stage1_rpn.cache_name, model.stage1_rpn.test_net_def_file);
% ###4/5### CHANGE EACH TIME*** : name of output map
output_map_name = 'output_map_multibox_ohem_happy';  % output_map_conv4, output_map_conv5
output_map_save_name = fullfile(cache_data_this_model_dir, output_map_name);
[conf_proposal.output_width_conv34, conf_proposal.output_height_conv34, ...
 conf_proposal.output_width_conv5, conf_proposal.output_height_conv5, ...
 conf_proposal.output_width_conv6, conf_proposal.output_height_conv6]...
                                            = proposal_calc_output_size_multibox_happy(conf_proposal, model.stage1_rpn.test_net_def_file, output_map_save_name);
% 1209: no need to change: same with all multibox
[conf_proposal.anchors_conv34,conf_proposal.anchors_conv5, conf_proposal.anchors_conv6] = proposal_generate_anchors_multibox_ohem_flip(cache_data_this_model_dir, ...
                                                            'ratios', [1.25 0.8], 'scales',  2.^[-1:5], 'add_size', [360 720 900]);  %[8 16 32 64 128 256 360 512 720 900]

%% read the RPN model
imdbs_name = cell2mat(cellfun(@(x) x.name, dataset.imdb_train, 'UniformOutput', false));
log_dir = fullfile(pwd, 'output', exp_name, 'rpn_cachedir', model.stage1_rpn.cache_name, imdbs_name);
%log_dir = fullfile(pwd, 'output', model.stage1_rpn.cache_name, imdbs_name);

final_model_path = fullfile(log_dir, 'final');
if exist(final_model_path, 'file')
    model.stage1_rpn.output_model_file = final_model_path;
else
    error('RPN model does not exist.');
end
            
%% generate proposal for training the BF
model.stage1_rpn.nms.per_nms_topN = -1;
%model.stage1_rpn.nms.nms_overlap_thres = 1; %1004: 1-->0.5
model.stage1_rpn.nms.nms_overlap_thres_conv4   	= 0.7; %1
model.stage1_rpn.nms.nms_overlap_thres_conv5   	= 0.7; %1
model.stage1_rpn.nms.nms_overlap_thres_conv6   	= 0.7; %1
%1201: since only 3 anchors, 100 is enough(in RPN: only 50 for conv4)
%model.stage1_rpn.nms.after_nms_topN = 50;  %600 --> 100 
model.stage1_rpn.nms.after_nms_topN_conv4      	= 100;  %50
model.stage1_rpn.nms.after_nms_topN_conv5      	= 100;  %30
model.stage1_rpn.nms.after_nms_topN_conv6      	= 10;  %3
is_test = true;
%roidb_test_BF = Faster_RCNN_Train.do_generate_bf_proposal_multibox_ohem_happy_3scale(conf_proposal, model.stage1_rpn, dataset.imdb_test, dataset.roidb_test, is_test, start_num);
Faster_RCNN_Train.do_generate_bf_proposal_multibox_3scale_nobf(conf_proposal, model.stage1_rpn, dataset.imdb_test, dataset.roidb_test, is_test);

end
