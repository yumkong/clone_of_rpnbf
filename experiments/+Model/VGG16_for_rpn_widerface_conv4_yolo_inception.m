function model = VGG16_for_rpn_widerface_conv4_yolo_inception(exp_name, cache_base_proposal, model)


model.mean_image                                = fullfile(pwd, 'models', exp_name, 'pre_trained_models', 'vgg_16layers', 'mean_image');
model.pre_trained_net_file                      = fullfile(pwd, 'models', exp_name, 'pre_trained_models', 'vgg_16layers', 'vgg16.caffemodel');
% Stride in input image pixels at the last conv layer
model.feat_stride                               = 8; %16

%% stage 1 rpn, inited from pre-trained network
model.stage1_rpn.solver_def_file                = fullfile(pwd, 'models', exp_name, 'rpn_prototxts', 'vgg_16layers_yolo', 'solver_60k80k_widerface_conv4_yolo_inception_lrn.prototxt'); 
model.stage1_rpn.test_net_def_file              = fullfile(pwd, 'models', exp_name, 'rpn_prototxts', 'vgg_16layers_yolo', 'test_widerface_conv4_yolo_inception_lrn.prototxt');
model.stage1_rpn.init_net_file                  = model.pre_trained_net_file;
model.stage1_rpn.cache_name = [cache_base_proposal, '_stage1_rpn'];
% rpn test setting
model.stage1_rpn.nms.per_nms_topN               = -1; %20000
model.stage1_rpn.nms.nms_overlap_thres       	= 1; %0.7
model.stage1_rpn.nms.after_nms_topN         	= 800;  %1019: 300 --> 500, since #anchor is doubled

end