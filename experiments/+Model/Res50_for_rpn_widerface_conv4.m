function model = Res50_for_rpn_widerface_conv4(exp_name, model)


model.mean_image                                = fullfile(pwd, 'models', exp_name, 'pre_trained_models', 'ResNet-50L', 'mean_image');
model.pre_trained_net_file                      = fullfile(pwd, 'models', exp_name, 'pre_trained_models', 'ResNet-50L', 'ResNet-50-model.caffemodel');
% Stride in input image pixels at the last conv layer
model.feat_stride                               = 8; %16

%% stage 1 rpn, inited from pre-trained network
model.stage1_rpn.solver_def_file                = fullfile(pwd, 'models', exp_name, 'rpn_prototxts', 'ResNet-50L_res3a', 'solver_80k110k_widerface_res50_conv4.prototxt'); 
model.stage1_rpn.test_net_def_file              = fullfile(pwd, 'models', exp_name, 'rpn_prototxts', 'ResNet-50L_res3a', 'test_widerface_res50_conv4.prototxt');
model.stage1_rpn.init_net_file                  = model.pre_trained_net_file;

% rpn test setting
model.stage1_rpn.nms.per_nms_topN               = -1; %20000
model.stage1_rpn.nms.nms_overlap_thres       	= 1;%0.7
model.stage1_rpn.nms.after_nms_topN         	= 800;  %1000


end