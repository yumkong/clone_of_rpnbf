function model = VGG16_for_rpn_widerface_multibox_ohem_ablation_newL1(exp_name, model)


model.mean_image                                = fullfile(pwd, 'models', exp_name, 'pre_trained_models', 'vgg_16layers', 'mean_image');
model.pre_trained_net_file                      = fullfile(pwd, 'models', exp_name, 'pre_trained_models', 'vgg_16layers', 'vgg16.caffemodel');
% Stride in input image pixels at the last conv layer
model.feat_stride_s4                              = 4; %16
model.feat_stride_s8                              = 16;
model.feat_stride_s16                              = 32;

%% stage 1 rpn, inited from pre-trained network
model.stage1_rpn.solver_def_file                = fullfile(pwd, 'models', exp_name, 'rpn_prototxts', 'vgg_16layers_final', 'solver_60k80k_widerface_multibox_ohem_ablation_newL1.prototxt'); 
model.stage1_rpn.test_net_def_file              = fullfile(pwd, 'models', exp_name, 'rpn_prototxts', 'vgg_16layers_final', 'test_widerface_multibox_ohem_ablation_newL1.prototxt');
model.stage1_rpn.init_net_file                  = model.pre_trained_net_file;

% rpn test setting
% 0224: same settings for conv3 4 and 5
model.stage1_rpn.nms.per_nms_topN               = -1; %20000
model.stage1_rpn.nms.nms_overlap_thres       	= 0.7;%0111: should also try 0.7 when doing testing
model.stage1_rpn.nms.after_nms_topN         	= 300;%1204:800--100

end