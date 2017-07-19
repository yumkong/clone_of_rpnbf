function script_main_2d_centerloss()
%% start
% --------------------------------------------------------
% Yuguang Liu
% 3 layers of loss output
% --------------------------------------------------------

clc;
clear mex;
clear is_valid_handle; % to clear init_key
% call './set_path.m'
run(fullfile(fileparts(mfilename('fullpath')), 'set_path'));
%% -------------------- CONFIG --------------------
%0930 change caffe folder according to platform
if ispc
    %opts.caffe_version          = 'caffe_faster_rcnn_win_cudnn_bn'; 
    opts.caffe_version          = 'caffe_centerloss_exp'; %'caffe_centerloss'
    cd('D:\\RPN_BF_master');
elseif isunix
    % caffe_faster_rcnn_rfcn is from caffe-rfcn-r-fcn_othersoft
    % caffe_faster_rcnn_rfcn_normlayer is also from
    % caffe-rfcn-r-fcn_othersoft with l2-normalization layer added
    opts.caffe_version          = 'caffe_centerloss_exp'; %'caffe_faster_rcnn_bn'
    cd('/usr/local/data/yuguang/git_all/RPN_BF_pedestrain/RPN_BF-RPN-pedestrian');
end
opts.gpu_id                 = helper.auto_select_gpu;
helper.active_caffe_mex(opts.gpu_id, opts.caffe_version);

%0120 to use bbApply('draw',bbs_show,'m') in widerface_all_flip_512
addpath(genpath('external/toolbox'));  % piotr's image and video toolbox
addpath(genpath('external/export_fig'));  % save image to png

% 1009 use more anchors
exp_name = 'VGG16_widerface';

% do validation, or not 
opts.do_val                 = true; 
% 0714 changed from orginal to 2d
%model                       = helper.model_conv3_s4(exp_name);
model                       = rpnmodel.get_prototxt_conv3_2d_centerloss(exp_name);
% cache base
cache_base_proposal         = 'widerface_VGG16';
%cache_base_fast_rcnn        = '';
% set cache folder for each stage
%model                       = Faster_RCNN_Train.set_cache_folder_widerface(cache_base_proposal,cache_base_fast_rcnn, model);
%model                       = Faster_RCNN_Train.set_cache_folder_widerface(cache_base_proposal, model);
model.stage1_rpn.cache_name = [cache_base_proposal, '_stage1_rpn'];
% train/test data
dataset                     = [];
% the root directory to hold any useful intermediate data during training process
cache_data_root = 'output2';  %cache_data
helper.mkdir_if_missing(cache_data_root);
addpath(cache_data_root);  %0713 added to fix warning of "could not find path"
%1009 change exp here for output
exp_name = 'old_conv3_s4_2d';
% the dir holding intermediate data paticular
cache_data_this_model_dir = fullfile(cache_data_root, exp_name, 'rpn_cache');
helper.mkdir_if_missing(cache_data_this_model_dir);
use_flipped                 = false;  %true --> false
% 0127: in vn7 only use 11 event for demo
train_event_pool            = [1 61 3 5 6 9 11 12 14 33 37 38 45 51 56]; %-1
dataset                     = helper.create_imdb_ablation_512(dataset, 'train', use_flipped, train_event_pool, cache_data_this_model_dir);
%dataset                     = Dataset.widerface_all(dataset, 'test', false, event_num, cache_data_this_model_dir, model_name_base);
%0106 added all test images
test_event_pool             = 1:61;
dataset                     = helper.create_imdb_ablation_512(dataset, 'test', false, test_event_pool, cache_data_this_model_dir);
% 0206 added: adapt dataset created in puck to VN7
%verify if image and box matches
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
%0805 added, make sure imdb_train and roidb_train are of cell type
if ~iscell(dataset.imdb_train)
    dataset.imdb_train = {dataset.imdb_train};
end
if ~iscell(dataset.roidb_train)
    dataset.roidb_train = {dataset.roidb_train};
end

% %% -------------------- TRAIN --------------------
% configurating rpn
conf_rpn          = helper.rpn_config('image_means', model.mean_image, 'feat_stride', model.feat_stride);
%conf_fast_rcnn              = fast_rcnn_config_widerface('image_means', model.mean_image);
% generate anchors and pre-calculate output size of rpn network 
conf_rpn.exp_name = exp_name;
%conf_fast_rcnn.exp_name = exp_name;

% ###4/5### CHANGE EACH TIME*** : name of output map
output_map_name = ['output_map_' exp_name];  % output_map_conv4, output_map_conv5
output_map_path = fullfile(cache_data_this_model_dir, output_map_name);
[conf_rpn.output_width_map, conf_rpn.output_height_map] = helper.calc_rpn_output_size(conf_rpn, ...
                                                                    model.stage1_rpn.test_net_def_file, output_map_path);
% 1209: no need to change: same with all multibox
conf_rpn.anchors = helper.rpn_generate_anchor(cache_data_this_model_dir, ...
                         'ratios', [1], 'scales',  0.5, 'add_size', []);  %[8 16 32 64 128 256 480]
                        % 'ratios', [1], 'scales',  2.^[-1:4], 'add_size', [480]);  %[8 16 32 64 128 256 480]     
%%  train
fprintf('\n***************\nstage one RPN \n***************\n');
model.stage1_rpn            = helper.rpn_train_wrap(conf_rpn, dataset, model.stage1_rpn, opts.do_val);
%% test
% 0129: use full-size validation images instead of 512x512
%dataset                     = Dataset.widerface_all(dataset, 'test', false, -1, cache_data_this_model_dir, model_name_base);
%0715 changed
%helper.rpn_test_wrap(conf_rpn, model.stage1_rpn, dataset.imdb_test, dataset.roidb_test, nms_option_test);
%rpn.rpn_test_wrap_2d(conf_rpn, model.stage1_rpn, dataset.imdb_test, dataset.roidb_test);
rpn.rpn_test_wrap_2d(conf_rpn, model.stage1_rpn, dataset.imdb_test);

end
