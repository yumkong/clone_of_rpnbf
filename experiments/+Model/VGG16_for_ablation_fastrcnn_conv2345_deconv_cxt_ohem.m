function model = VGG16_for_ablation_fastrcnn_conv2345_deconv_cxt_ohem(exp_name, model)


model.mean_image                                = fullfile(pwd, 'models', exp_name, 'pre_trained_models', 'vgg_16layers', 'mean_image');
model.pre_trained_net_file                      = fullfile(pwd, 'models', exp_name, 'pre_trained_models', 'vgg_16layers', 'vgg16.caffemodel');
% Stride in input image pixels at the last conv layer
%model.feat_stride                               = 16;
model.feat_stride_s4                              = 4;
model.feat_stride_s8                              = 8;
model.feat_stride_s16                             = 16;

%% stage 1 rpn, inited from pre-trained network
%0304: final3 is currently the best model, use it
model.stage1_rpn.solver_def_file                = fullfile(pwd, 'models', exp_name, 'rpn_prototxts', 'vgg_16layers_ablation', 'final_models', 'solver_30k40k_final3.prototxt'); 
model.stage1_rpn.test_net_def_file              = fullfile(pwd, 'models', exp_name, 'rpn_prototxts', 'vgg_16layers_ablation', 'final_models', 'test_final3.prototxt');
model.stage1_rpn.init_net_file                  = model.pre_trained_net_file;

% rpn test setting
% 0224: same settings for conv3 4 and 5
model.stage1_rpn.nms.per_nms_topN               = -1; %20000
model.stage1_rpn.nms.nms_overlap_thres       	= 0.7;%0111: should also try 0.7 when doing testing
model.stage1_rpn.nms.after_nms_topN_s4        	= 100;%0304:300
model.stage1_rpn.nms.after_nms_topN_s8         	= 100;%0304:300
model.stage1_rpn.nms.after_nms_topN_s16         = 50;%0304:300

%stage 1 fast rcnn, inited from stage 1 rpn
% *** 0306: solver_60k80k_stage2_final_tmp is temperarily used here
model.stage1_fast_rcnn.solver_def_file          = fullfile(pwd, 'models', exp_name,'rpn_prototxts', 'vgg_16layers_ablation', 'final_model_stage2_cxt','solver_60k80k_stage2_conv2345_deconv_ohem.prototxt');
model.stage1_fast_rcnn.test_net_def_file        = fullfile(pwd, 'models', exp_name,'rpn_prototxts', 'vgg_16layers_ablation', 'final_model_stage2_cxt', 'test_stage2_conv2345_deconv_ohem.prototxt');
end