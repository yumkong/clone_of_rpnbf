function model = VGG16_for_rpn_widerface_conv3plus4(exp_name, model)


model.mean_image                                = fullfile(pwd, 'models', exp_name, 'pre_trained_models', 'vgg_16layers', 'mean_image');
model.pre_trained_net_file                      = fullfile(pwd, 'models', exp_name, 'pre_trained_models', 'vgg_16layers', 'vgg16.caffemodel');
% Stride in input image pixels at the last conv layer
model.feat_stride                               = 4; %16

%% stage 1 rpn, inited from pre-trained network
model.stage1_rpn.solver_def_file                = fullfile(pwd, 'models', exp_name, 'rpn_prototxts', 'vgg_16layers_conv3_final', 'solver_60k80k_widerface_conv3plus4.prototxt'); 
model.stage1_rpn.test_net_def_file              = fullfile(pwd, 'models', exp_name, 'rpn_prototxts', 'vgg_16layers_conv3_final', 'test_widerface_conv3plus4.prototxt');
model.stage1_rpn.init_net_file                  = model.pre_trained_net_file;

% rpn test setting
model.stage1_rpn.nms.per_nms_topN               = -1; %20000
model.stage1_rpn.nms.nms_overlap_thres       	= 1;%0.7
model.stage1_rpn.nms.after_nms_topN         	= 300;  %1204:800--100

end