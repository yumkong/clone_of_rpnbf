function model = VGG16_for_rpn_widerface_multibox_ohem_singleconv4(exp_name, model)


model.mean_image                                = fullfile(pwd, 'models', exp_name, 'pre_trained_models', 'vgg_16layers', 'mean_image');
model.pre_trained_net_file                      = fullfile(pwd, 'models', exp_name, 'pre_trained_models', 'vgg_16layers', 'vgg16.caffemodel');
% Stride in input image pixels at the last conv layer
model.feat_stride_conv4                              = 8; %16
model.feat_stride_conv5                              = 16;
model.feat_stride_conv6                              = 32;

%% stage 1 rpn, inited from pre-trained network
% xxx_final and xxx_final2 share this model def file, because they are
% pretty much the same thing, only difference is add a conv4_atros layer
model.stage1_rpn.solver_def_file                = fullfile(pwd, 'models', exp_name, 'rpn_prototxts', 'vgg_16layers_multibox', 'solver_60k80k_widerface_multibox_ohem_singleconv4.prototxt'); 
model.stage1_rpn.test_net_def_file              = fullfile(pwd, 'models', exp_name, 'rpn_prototxts', 'vgg_16layers_multibox', 'test_widerface_multibox_ohem_singleconv4.prototxt');
model.stage1_rpn.init_net_file                  = model.pre_trained_net_file;

% rpn test setting
model.stage1_rpn.nms.per_nms_topN               = -1; %20000
model.stage1_rpn.nms.nms_overlap_thres       	= 1;%0.7
%model.stage1_rpn.nms.after_nms_topN         	= 300;  %1000
model.stage1_rpn.nms.after_nms_topN_conv4      	= 50;  %1000
model.stage1_rpn.nms.after_nms_topN_conv5      	= 100;  %1000
model.stage1_rpn.nms.after_nms_topN_conv6      	= 10;  %1000

end