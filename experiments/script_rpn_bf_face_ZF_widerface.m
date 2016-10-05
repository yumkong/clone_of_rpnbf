function script_rpn_bf_face_ZF_widerface()

clc;
clear mex;
clear is_valid_handle; % to clear init_key
run(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'startup'));

%0929 added: switch from '$root/experiments' to '$root', to run on puck
%with matlab -nojvm
%cd('..');  %
%% -------------------- CONFIG --------------------
%0930 change caffe folder according to platform
if ispc
    opts.caffe_version          = 'caffe_faster_rcnn_win';
    cd('D:\\RPN_BF_master');
elseif isunix
    opts.caffe_version          = 'caffe_faster_rcnn';
    cd('/usr/local/data/yuguang/git_all/RPN_BF_pedestrain/RPN_BF-RPN-pedestrian');
end
opts.gpu_id                 = auto_select_gpu;
active_caffe_mex(opts.gpu_id, opts.caffe_version);

exp_name = 'ZF_widerface';

% do validation, or not 
opts.do_val                 = true; 
% model
model                       = Model.ZF_for_rpn_widerface(exp_name);
% cache base
cache_base_proposal         = 'rpn_widerface_ZF';
% set cache folder for each stage
model                       = Faster_RCNN_Train.set_cache_folder_widerface(cache_base_proposal, model);
% train/test data
dataset                     = [];
% the root directory to hold any useful intermediate data during training process
cache_data_root = 'output';  %cache_data
mkdir_if_missing(cache_data_root);
% ###3/5### CHANGE EACH TIME*** use this to name intermediate data's mat files
model_name_base = 'ZF';  % ZF, vgg16_conv5
% the dir holding intermediate data paticular
cache_data_this_model_dir = fullfile(cache_data_root, exp_name, 'rpn_cachedir');
mkdir_if_missing(cache_data_this_model_dir);
use_flipped                 = false;  %true --> false
event_num                   = 3;
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
conf_proposal               = proposal_config_widerface('image_means', model.mean_image, 'feat_stride', model.feat_stride);

% generate anchors and pre-calculate output size of rpn network 

conf_proposal.exp_name = exp_name;
% [conf_proposal.anchors, conf_proposal.output_width_map, conf_proposal.output_height_map] ...
%                             = proposal_prepare_anchors(conf_proposal, model.stage1_rpn.cache_name, model.stage1_rpn.test_net_def_file);
% ###4/5### CHANGE EACH TIME*** : name of output map
output_map_name = 'output_map_ZF';  % output_map_conv4, output_map_conv5
output_map_save_name = fullfile(cache_data_this_model_dir, output_map_name);
[conf_proposal.output_width_map, conf_proposal.output_height_map] = proposal_calc_output_size(conf_proposal, ...
                                                                    model.stage1_rpn.test_net_def_file, output_map_save_name);
conf_proposal.anchors = proposal_generate_anchors(cache_data_this_model_dir, 'scales',  2.^[-1:5]);

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
model.stage1_rpn.nms.nms_overlap_thres = 1;
model.stage1_rpn.nms.after_nms_topN = 500;  %40
is_test = true;
roidb_test_BF = Faster_RCNN_Train.do_generate_bf_proposal_widerface(conf_proposal, model.stage1_rpn, dataset.imdb_test, dataset.roidb_test, is_test);
model.stage1_rpn.nms.nms_overlap_thres = 0.7;
model.stage1_rpn.nms.after_nms_topN = 1000;
roidb_train_BF = Faster_RCNN_Train.do_generate_bf_proposal_widerface(conf_proposal, model.stage1_rpn, dataset.imdb_train{1}, dataset.roidb_train{1}, ~is_test);

%% train the BF
BF_cachedir = fullfile(pwd, 'output', exp_name, 'bf_cachedir');
mkdir_if_missing(BF_cachedir);
dataDir = fullfile('datasets','caltech');                % Caltech ==> to be replaced?
posGtDir = fullfile(dataDir, 'train', 'annotations');  % Caltech ==> to be replaced?
addpath(fullfile('external', 'code3.2.1'));              % Caltech ==> to be replaced?
addpath(genpath('external/toolbox'));  % piotr's image and video toolbox
%addpath(fullfile('..','external', 'toolbox'));
BF_prototxt_path = fullfile('models', exp_name, 'bf_prototxts', 'test_feat_conv34atrous_v2.prototxt');
%BF_prototxt_path = fullfile('..','models', exp_name, 'bf_prototxts', 'test_feat_conv34atrous_v2.prototxt');
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
caffe.set_mode_gpu();

% set up opts for training detector (see acfTrain)
opts = DeepTrain_otf_trans_ratio(); 
opts.cache_dir = BF_cachedir;
opts.name=fullfile(opts.cache_dir, 'DeepCaltech_otf');
opts.nWeak=[64 128 256 512 1024 1536 2048];
opts.bg_hard_min_ratio = [1 1 1 1 1 1 1];
opts.pBoost.pTree.maxDepth=5; 
opts.pBoost.discrete=0;  %?
opts.pBoost.pTree.fracFtrs=1/4;  %? 
opts.first_nNeg = 30000;  %?  #neg of the 1st stage
opts.nNeg = 5000;  % #neg needed by every stage
opts.nAccNeg = 50000;  % #accumulated neg from stage2 -- 7
pLoad={'lbls',{'person'},'ilbls',{'people'},'squarify',{3,.41}};  % delete?
opts.pLoad = [pLoad 'hRng',[50 inf], 'vRng',[1 1] ];   % delete?

% 1001: add 'ignores' field to roidb
[roidb_train_BF.rois(:).ignores] = deal([]);
for kk = 1:length(roidb_train_BF.rois)
    tm_gt = roidb_train_BF.rois(kk).gt;
    tm_gt = tm_gt(tm_gt > 0);
    %all gts are not ignored in widerface
    roidb_train_BF.rois(kk).ignores = zeros(size(tm_gt));
end
opts.roidb_train = roidb_train_BF;
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
opts.fg_thres_lo = 0.8; %[lo, hi)
opts.bg_thres_hi = 0.5;
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
opts.ratio = 1.0;
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
feat = rois_get_features_ratio(conf, caffe_net, img, tmp_box, opts.max_rois_num_in_gpu, opts.ratio);
toc;
opts.feat_len = length(feat);

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
detector = DeepTrain_otf_trans_ratio( opts );

% visual
%if 1 % set to 1 for visual
    % ########## save the final result (after BF) here #############
    cache_dir1 = fullfile(pwd, 'output', exp_name, 'rpn_cachedir', model.stage1_rpn.cache_name, dataset.imdb_test.name);
    fid = fopen(fullfile(cache_dir1, 'ZF_e1-e3-RPN+BF.txt'), 'a');
  rois = opts.roidb_test.rois;
  %imgNms=bbGt('getFiles',{[dataDir 'test/images']});
  for i = 1:length(rois)
      if ~isempty(rois(i).boxes)
          img = imread(dataset.imdb_test.image_at(i));  
          feat = rois_get_features_ratio(conf, caffe_net, img, rois(i).boxes, opts.max_rois_num_in_gpu, opts.ratio);   
          scores = adaBoostApply(feat, detector.clf);
          bbs = [rois(i).boxes scores];
 
          % do nms
          % nms_thres  = 0.5
          %if i~=29
          %sel_idx = nms(bbs, opts.nms_thres);
          %end
          sel_idx = (1:size(bbs,1))'; %'
          sel_idx = intersect(sel_idx, find(~rois(i).gt)); % exclude gt
          
          % ########## save the final result (after BF) here #############
          %fid = fopen(fullfile(cache_dir, 'ZF_e1-e3-RPN+BF.txt'), 'a');
            %for i = 1:size(aboxes, 1)
            bbs = bbs(sel_idx, :);
                if ~isempty(bbs)
                    sstr = strsplit(dataset.imdb_test.image_ids{i}, '\');
                    % [x1 y1 x2 y2] pascal VOC style
                    for j = 1:size(bbs,1)
                        %each row: [image_name score x1 y1 x2 y2]
                        fprintf(fid, '%s %f %d %d %d %d\n', sstr{2}, bbs(j, 5), round(bbs(j, 1:4)));
                    end
                end
                %if i == 29
                disp(i);
                %end
            %end
            %fclose(fid);
            %fprintf('Done with saving RPN+BF detected boxes.\n');
    
          % liu@1001: use a higher thresh to get rid of false alarms:  opts.cascThr = -1,  new set 5
          %sel_idx = intersect(sel_idx, find(bbs(:, end) > opts.cascThr));
%           sel_idx = intersect(sel_idx, find(bbs(:, end) > 5));
%           bbs = bbs(sel_idx, :);
%           bbs(:, 3) = bbs(:, 3) - bbs(:, 1);
%           bbs(:, 4) = bbs(:, 4) - bbs(:, 2);
%           if ~isempty(bbs)
%               %I=imread(imgNms{i});
%               figure(1); 
%               im(img);  %im(I)
%               bbApply('draw',bbs); pause();
%           end
      end
  end
  %1004 added
  fclose(fid);
  fprintf('Done with saving RPN+BF+nms detected boxes.\n');
%end

% test detector and plot roc
% method_name = 'RPN+BF';
% folder1 = fullfile(pwd, 'output', exp_name, 'bf_cachedir', method_name);
% folder2 = fullfile('..', 'external', 'code3.2.1', 'data-USA', 'res', method_name);
% 
% %liu@0929: 'show' 2-->0
% if ~exist(folder1, 'dir')
%     [~,~,gt,dt]=DeepTest_otf_trans_ratio('name',opts.name,'roidb_test', opts.roidb_test, 'imdb_test', opts.imdb_test, ...
%         'gtDir',[dataDir 'test/annotations'],'pLoad',[pLoad, 'hRng',[50 inf],...
%         'vRng',[.65 1],'xRng',[5 635],'yRng',[5 475]],...
%         'reapply',1,'show',0, 'nms_thres', opts.nms_thres, ...
%         'conf', opts.conf, 'caffe_net', opts.caffe_net, 'silent', false, 'cache_dir', opts.cache_dir, 'ratio', opts.ratio);
% end
% 
% copyfile(folder1, folder2);
% tmp_dir = pwd;
% cd(fullfile(pwd, 'external', 'code3.2.1'));
% dbEval_RPNBF;
% cd(tmp_dir);

caffe.reset_all();
end

function [anchors, output_width_map, output_height_map] = proposal_prepare_anchors(conf, cache_name, test_net_def_file)
    [output_width_map, output_height_map] ...                           
                                = proposal_calc_output_size_caltech(conf, test_net_def_file);
    anchors                = proposal_generate_anchors_caltech(cache_name, ...
                                    'scales',  2.6*(1.3.^(0:8)), ...
                                    'ratios', [1 / 0.41], ...
                                    'exp_name', conf.exp_name);
end
