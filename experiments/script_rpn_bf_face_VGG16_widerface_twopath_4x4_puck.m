function script_rpn_bf_face_VGG16_widerface_twopath_4x4_puck()

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

% 1009 use more anchors
exp_name = 'Res50_widerface';

% do validation, or not 
opts.do_val                 = true; 
% model
model                       = Model.Res101_for_rpn_widerface_twopath_happy_flip_batch2(exp_name);
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
event_num_test              = -1;  %1007 added: test all val images

dataset                     = Dataset.widerface_all_flip_512(dataset, 'train', use_flipped, event_num, cache_data_this_model_dir, model_name_base);
dataset                     = Dataset.widerface_all(dataset, 'test', false, event_num_test, cache_data_this_model_dir, model_name_base);
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
model.stage1_rpn.nms.nms_overlap_thres   	= 1; %0.7
%1201: since only 3 anchors, 100 is enough(in RPN: only 50 for conv4)
%model.stage1_rpn.nms.after_nms_topN = 50;  %600 --> 100 
model.stage1_rpn.nms.after_nms_topN_res23      	= 100;  %50
model.stage1_rpn.nms.after_nms_topN_res45      	= 100;  %30
roidb_test_BF = Faster_RCNN_Train.do_generate_bf_proposal_twopath(conf_proposal, model.stage1_rpn, dataset.imdb_test, dataset.roidb_test, 3); %3 is nms option
% %model.stage1_rpn.nms.nms_overlap_thres = 0.7; % not have so much overlap, since the upmost size is only 32x32, but still do it here
% model.stage1_rpn.nms.nms_overlap_thres_conv4   	= 0.7; % no nms for conv4
% model.stage1_rpn.nms.nms_overlap_thres_conv5   	= 0.7;
% model.stage1_rpn.nms.nms_overlap_thres_conv6   	= 0.7;
% %model.stage1_rpn.nms.after_nms_topN = 50; %1000--> 200. 200 is enough (double of test topN), only keep the hard negative one
% model.stage1_rpn.nms.after_nms_topN_conv4      	= 50;  %50
% model.stage1_rpn.nms.after_nms_topN_conv5      	= 30;  %30
% model.stage1_rpn.nms.after_nms_topN_conv6      	= 3;  %3
% roidb_train_BF = Faster_RCNN_Train.do_generate_bf_proposal_multibox_ohem_happy_vn7(conf_proposal, model.stage1_rpn, dataset.imdb_train{1}, dataset.roidb_train{1}, ~is_test);

%% train the BF
BF_cachedir = fullfile(pwd, 'output', exp_name, 'bf_cachedir_context_4x4_context_puck');  %puck
mkdir_if_missing(BF_cachedir);
dataDir = fullfile('datasets','caltech');                % Caltech ==> to be replaced?
%posGtDir = fullfile(dataDir, 'train', 'annotations');  % Caltech ==> to be replaced?
addpath(fullfile('external', 'code3.2.1'));              % Caltech ==> to be replaced?
addpath(genpath('external/toolbox'));  % piotr's image and video toolbox
%addpath(fullfile('..','external', 'toolbox'));
BF_prototxt_path = fullfile('models', 'VGG16_widerface', 'bf_prototxts', 'test_feat_conv34atrous_multibox_ohem.prototxt'); %test_feat_conv34atrous_multibox_ohem_5x5
conf.image_means = model.mean_image;
conf.test_scales = conf_proposal.test_scales;
conf.test_max_size = conf_proposal.max_size;
if ischar(conf.image_means)
    s = load(conf.image_means);
    s_fieldnames = fieldnames(s);
    assert(length(s_fieldnames) == 1);
    conf.image_means = s.(s_fieldnames{1});
end
log_dir = fullfile(BF_cachedir, 'log');
mkdir_if_missing(log_dir);
caffe_log_file_base = fullfile(log_dir, 'caffe_log');
caffe.init_log(caffe_log_file_base);
caffe_net = caffe.Net(BF_prototxt_path, 'test');  % error here
caffe_net.copy_from(final_model_path);
%1004 changed:
if conf_proposal.use_gpu
    caffe.set_mode_gpu();
else
    caffe.set_mode_cpu();
end

% set up opts for training detector (see acfTrain)
opts = DeepTrain_otf_trans_ratio(); 
opts.cache_dir = BF_cachedir;
opts.name = fullfile(opts.cache_dir, 'DeepCaltech_otf');
opts.nWeak = [64 128 256 512 1024 1536 2048];
opts.bg_hard_min_ratio = [1 1 1 1 1 1 1];
opts.pBoost.pTree.maxDepth = 5; 
opts.pBoost.discrete = 0;  %?
opts.pBoost.pTree.fracFtrs = 1/4;  %? 
opts.first_nNeg = 140000;  %1227: 150000 --> 120000
opts.nNeg = 30000;  % #neg needed by every stage 5000--> 300000
opts.nAccNeg = 190000;  % #1227: 200000 --> 180000 --> 160000
% 1203 added
opts.nPerNeg = 10;
pLoad={'lbls',{'person'},'ilbls',{'people'},'squarify',{3,.41}};  % delete?
opts.pLoad = [pLoad 'hRng',[50 inf], 'vRng',[1 1] ];   % delete?

% 1001: add 'ignores' field to roidb
% [roidb_train_BF.rois(:).ignores] = deal([]);
% for kk = 1:length(roidb_train_BF.rois)
%     tm_gt = roidb_train_BF.rois(kk).gt;
%     tm_gt = tm_gt(tm_gt > 0);
%     %all gts are not ignored in widerface
%     roidb_train_BF.rois(kk).ignores = zeros(size(tm_gt));
% end
% opts.roidb_train = roidb_train_BF;
% 1001: add 'ignores' field to roidb
[roidb_test_BF.rois(:).ignores] = deal([]);
for kk = 1:length(roidb_test_BF.rois)
    tm_gt = roidb_test_BF.rois(kk).gt;
    tm_gt = tm_gt(tm_gt > 0);
    %all gts are not ignored in widerface
    roidb_test_BF.rois(kk).ignores = zeros(size(tm_gt));
end
opts.roidb_test = roidb_test_BF;
opts.imdb_train = dataset.imdb_train{1};
opts.imdb_test = dataset.imdb_test;
opts.fg_thres_hi = 1;
opts.fg_thres_lo = 0.5; %[lo, hi) 1018: 0.8 --> 0.5 --> 0.6 (1203) --> 0.5 (1210)
opts.bg_thres_hi = 0.3; %1018: 0.5 --> 0.3
opts.bg_thres_lo = 0; %[lo hi)
opts.dataDir = dataDir;
opts.caffe_net = caffe_net;
opts.conf = conf;
opts.exp_name = exp_name;
opts.fg_nms_thres = 1;
opts.fg_use_gt = true;
opts.bg_nms_thres = 1;
opts.max_rois_num_in_gpu = 3000;
opts.init_detector = '';
opts.load_gt = false;
opts.ratio = 2.0;  %1018: 1.0 --> 2.0: left-0.5*width, right+0.5*width, top-0.2*height, bottom + 0.8height
opts.nms_thres = 0.5;

% forward an image to check error and get the feature length
img = imread(dataset.imdb_test.image_at(1));
tic;
tmp_box = roidb_test_BF.rois(1).boxes;
% only keep (bg_hard_min_ratio X #boxes) random bbxes
retain_num = round(size(tmp_box, 1) * opts.bg_hard_min_ratio(end));
retain_idx = randperm(size(tmp_box, 1), retain_num);
sel_idx = true(size(tmp_box, 1), 1);
sel_idx = sel_idx(retain_idx);
% doing nms to reduce boxes number ==> here bg_nums_thres is 1, so not do it
if opts.bg_nms_thres < 1
    sel_box = roidb_test_BF.rois(1).boxes(sel_idx, :);
    sel_scores = roidb_test_BF.rois(1).scores(sel_idx, :);
    nms_sel_idxes = nms([sel_box sel_scores], opts.bg_nms_thres);
    sel_idx = sel_idx(nms_sel_idxes);
end
tmp_box = roidb_test_BF.rois(1).boxes(sel_idx, :);
% liu@1001: extract deep features from tmp_box
% opts.max_rois_num_in_gpu = 3000, opts.ratio = 1
feat = rois_get_features_ratio_4x4_context(conf, caffe_net, img, tmp_box, opts.max_rois_num_in_gpu, opts.ratio);
toc;
opts.feat_len = size(feat,2); %1203 changed: length(feat)

% fs=bbGt('getFiles',{posGtDir});
% train_gts = cell(length(fs), 1);
% for i = 1:length(fs)
%     [~,train_gts{i}]=bbGt('bbLoad',fs{i},opts.pLoad);
% end

% get gt boxes
train_gts = cell(length(dataset.roidb_train{1}.rois), 1);
for i = 1:length(train_gts)
    % [x y x2 y2]
     tmp_gt = dataset.roidb_train{1}.rois(i).boxes;
     % [x y w h is_gt], always set is_gt as 1
    %train_gts{i} = cat(2, train_gts{i}, ones(size(train_gts{i},1),1));
     train_gts{i} = [tmp_gt(:,1) tmp_gt(:,2) tmp_gt(:,3)-tmp_gt(:,1)+1 tmp_gt(:,4)-tmp_gt(:,2)+1 ones(size(tmp_gt(:,1)))];
    
end
opts.train_gts = train_gts;

% train BF detector
detector = DeepTrain_otf_trans_ratio_4x4_context( opts );

show_image = false;
SUBMIT_cachedir = fullfile(pwd, 'output', exp_name, 'submit_bf_val');
mkdir_if_missing(SUBMIT_cachedir);
final_score_path = fullfile(pwd, 'output', exp_name, 'rpn_cachedir', model.stage1_rpn.cache_name, dataset.imdb_test.name);
mkdir_if_missing(final_score_path);
final_score_file = fullfile(final_score_path, 'val_box_score.mat');
try
    % try to load cache
    ld = load(final_score_file);
    bbs_repo = ld.bbs_repo;
    bf_score_min = ld.bf_score_min;
    bf_score_max = ld.bf_score_max;
    clear ld;
catch   
    rois = opts.roidb_test.rois;
    %bf_score_min = 0;
    %bf_score_max = 0;
    bbs_repo = cell(length(rois), 1);
    num_val = length(rois);
    for i = 1:num_val
        fprintf('Processing image %d / %d\n', i, num_val);
        if ~isempty(rois(i).boxes)
            img = imread(dataset.imdb_test.image_at(i));  
            feat = rois_get_features_ratio_4x4_context(conf, caffe_net, img, rois(i).boxes, opts.max_rois_num_in_gpu, opts.ratio);   
            bf_scores = adaBoostApply(feat, detector.clf);
            
            mprpn_scores = rois(i).scores;
            bbs_all = [rois(i).boxes bf_scores mprpn_scores];

            sel_idx = (1:size(bbs_all,1))'; %'
            sel_idx = intersect(sel_idx, find(~rois(i).gt)); % exclude gt

            bbs = bbs_all(sel_idx, :);
			%bf_score_min = min(bf_score_min, min(bbs(:,5)));
            %bf_score_max = max(bf_score_max, max(bbs(:,5)));
            bbs_repo{i} = bbs;
        end
    end
    %get min/max bf scores and save them
    bbs_tmp = cell2mat(bbs_repo);
    bf_score_min = min(bbs_tmp(:,5));
    bf_score_max = max(bbs_tmp(:,5));
    save(final_score_file, 'bbs_repo','bf_score_min','bf_score_max');
    clear bbs_tmp;
end

% optional: cubic root of bf scores
bf_score_min = nthroot(bf_score_min, 3);
bf_score_max = nthroot(bf_score_max, 3);

for i = 1:length(bbs_repo)
    sstr = strsplit(dataset.imdb_test.image_ids{i}, filesep);
    event_name = sstr{1};
    event_dir = fullfile(SUBMIT_cachedir, event_name);
    mkdir_if_missing(event_dir);
    fid = fopen(fullfile(event_dir, [sstr{2} '.txt']), 'a');
    fprintf(fid, '%s\n', [dataset.imdb_test.image_ids{i} '.jpg']);

    bbs = bbs_repo{i};
    % 0107 fixed a bug here: empty bbs
    if ~isempty(bbs)
        % optinal: cubic root of bf scores
        bbs(:,5) = nthroot(bbs(:,5), 3);
        bbs(:,5) = (bbs(:,5) - bf_score_min) / (bf_score_max - bf_score_min);
    end
    % 0107: add visualization here!!!
    if show_image
        img = imread(dataset.imdb_test.image_at(i));
        bbs_show = [bbs(:,1:4) (bbs(:,5)+bbs(:,6))/2];
        figure(1), clf;
        imshow(img);
        hold on
        if ~isempty(bbs_show)
            bbs_show(:, 3) = bbs_show(:, 3) - bbs_show(:, 1) + 1;
            bbs_show(:, 4) = bbs_show(:, 4) - bbs_show(:, 2) + 1;
            bbApply('draw',bbs_show,'m');
        end
        hold off
    end
    
    % print the bbox number
    fprintf(fid, '%d\n', size(bbs, 1));
    if ~isempty(bbs)
        for j = 1:size(bbs,1)
            %each row: [x1 y1 w h score]
            %fprintf(fid, '%d %d %d %d %f\n', round([bbs(j,1) bbs(j,2) bbs(j,3)-bbs(j,1)+1 bbs(j,4)-bbs(j,2)+1]), bbs(j, 5));
            %fprintf(fid, '%d %d %d %d %f\n', round([bbs(j,1) bbs(j,2) bbs(j,3)-bbs(j,1)+1 bbs(j,4)-bbs(j,2)+1]), max(bbs(j, 5), bbs(j, 6)));
            fprintf(fid, '%d %d %d %d %f\n', round([bbs(j,1) bbs(j,2) bbs(j,3)-bbs(j,1)+1 bbs(j,4)-bbs(j,2)+1]), (bbs(j, 5) + 2*bbs(j, 6))/3);
        end
    end

    fclose(fid);
    fprintf('Done with saving image %d bboxes.\n', i);
end

caffe.reset_all();
end
