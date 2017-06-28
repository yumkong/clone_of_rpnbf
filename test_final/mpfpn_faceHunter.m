function mpfpn_faceHunter()
clc;
clear mex;
cd('/usr/local/data/yuguang/git_all/RPN_BF_pedestrain/RPN_BF-RPN-pedestrian/test_final');

clear is_valid_handle;
run(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'startup'));

if ispc
    opts.caffe_version          = 'caffe_faster_rcnn_win_cudnn_newL1'; %'caffe_faster_rcnn_win_cudnn_bn'
elseif isunix
    opts.caffe_version          = 'caffe_faster_rcnn_newL1'; %'caffe_faster_rcnn_bn'; 
    %cd('/usr/local/data/yuguang/git_all/RPN_BF_pedestrain/RPN_BF-RPN-pedestrian');
end
cd('..');
addpath('test_final');
addpath(fullfile('test_final','util_funcs'));
opts.gpu_id = auto_select_gpu;
active_caffe_mex(opts.gpu_id, opts.caffe_version);

% load config file
model_file = fullfile('test_final', 'model','configure.mat');
if exist(model_file, 'file')
    ldat = load(model_file);
    conf = ldat.conf;
else
    exp_name = 'VGG16_widerface';
    outfile = fullfile('test_final', 'model','outmap_size.mat');
    cache_data_this_model_dir = fullfile('test_final', 'model');
    model = Model.VGG16_for_mpfvn_ablation_total_conv2345_cxt_smart(exp_name);
    conf = proposal_config_me('image_means', model.mean_image, 'feat_stride_s4', model.feat_stride_s4,...
                              'feat_stride_s8', model.feat_stride_s8, 'feat_stride_s16', model.feat_stride_s16);
    [conf.output_width_s4, conf.output_height_s4, ...
    conf.output_width_s8, conf.output_height_s8, ...
    conf.output_width_s16, conf.output_height_s16] = proposal_calc_output_size_me(conf, ...
                                                         model.stage1_rpn.test_net_def_file, outfile);
    % 1209: no need to change: same with all multibox
    [conf.anchors_s4,conf.anchors_s8, conf.anchors_s16] = proposal_generate_anchors_ablation_total(cache_data_this_model_dir, ...
                         'ratios', [1.25 0.8], 'scales',  2.^[-1:4], 'add_size', [480]);  %[8 16 32 64 128 256 480]
    save(model_file, 'conf');
end

% load model 
opts.net_def_file = fullfile('test_final', 'model', 'test_final3_nodivanchor_flip.prototxt');
opts.net_file = fullfile('test_final', 'model', 'final');
caffe_net = caffe.Net(opts.net_def_file, 'test');
caffe_net.copy_from(opts.net_file);

% set gpu/cpu
if conf.use_gpu
    caffe.set_mode_gpu();
else
    caffe.set_mode_cpu();
end 

% read image
im = imread(fullfile('test_final','selfie.jpg'));
% detect faces
final_box = mpfpn_detect(conf, caffe_net, im);

%show result
figure(1),imshow(im);
hold on
bbApply('draw', final_box,'g');
hold off

caffe.reset_all(); 
end