function model = ZF_for_rpn_widerface(exp_name, model)


model.mean_image                                = fullfile(pwd, 'models', exp_name, 'pre_trained_models', 'ZF', 'mean_image');
model.pre_trained_net_file                      = fullfile(pwd, 'models', exp_name, 'pre_trained_models', 'ZF', 'ZF.caffemodel');
% Stride in input image pixels at the last conv layer
model.feat_stride                               = 16;

%% stage 1 rpn, inited from pre-trained network
model.stage1_rpn.solver_def_file                = fullfile(pwd, 'models', exp_name, 'rpn_prototxts', 'ZF', 'solver_60k80k.prototxt');
model.stage1_rpn.test_net_def_file              = fullfile(pwd, 'models', exp_name, 'rpn_prototxts', 'ZF', 'test.prototxt');
model.stage1_rpn.init_net_file                  = model.pre_trained_net_file;

% rpn test setting
model.stage1_rpn.nms.per_nms_topN               = -1;%10000
model.stage1_rpn.nms.nms_overlap_thres       	= 1;
model.stage1_rpn.nms.after_nms_topN         	= 500;  %40
end