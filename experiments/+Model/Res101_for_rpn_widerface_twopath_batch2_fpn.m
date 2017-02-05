function model = Res101_for_rpn_widerface_twopath_batch2_fpn(exp_name, model)


model.mean_image                                = fullfile(pwd, 'models', exp_name, 'pre_trained_models', 'ResNet-101L', 'mean_image');
model.pre_trained_net_file                      = fullfile(pwd, 'models', exp_name, 'pre_trained_models', 'ResNet-101L', 'ResNet-101-model.caffemodel');
% Stride in input image pixels at the last conv layer
model.feat_stride_res23                              = 4;
model.feat_stride_res45                              = 16;

%% stage 1 rpn, inited from pre-trained network
model.stage1_rpn.solver_def_file                = fullfile(pwd, 'models', exp_name, 'rpn_prototxts', 'res101', 'solver_300k400k_fpn.prototxt'); 
model.stage1_rpn.test_net_def_file              = fullfile(pwd, 'models', exp_name, 'rpn_prototxts', 'res101', 'test_fpn.prototxt');
model.stage1_rpn.init_net_file                  = model.pre_trained_net_file;

% rpn test setting
model.stage1_rpn.nms.per_nms_topN               = -1;   %20000
model.stage1_rpn.nms.nms_overlap_thres       	= 0.7;    %0.5
%model.stage1_rpn.nms.after_nms_topN         	= 300;  %1000
model.stage1_rpn.nms.after_nms_topN_res23       = 200;  %50
model.stage1_rpn.nms.after_nms_topN_res45      	= 200;  %100

end