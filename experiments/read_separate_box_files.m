function read_separate_box_files()

cache_dir = fullfile(pwd, 'output', 'VGG16_widerface_multibox_ohem_happy_flip', 'rpn_cachedir', 'rpn_widerface_VGG16_stage1_rpn', 'WIDERFACE_test', 'val_img_bbox');
%load every results individually
val_num = 3226;
aboxes_conv4 = cell(val_num, 1);
aboxes_conv5 = cell(val_num, 1);
aboxes_conv6 = cell(val_num, 1);

for ii = 1:3226
    fprintf('Loading boxes of image %d/%d \n', ii, val_num);
    ld = load(fullfile(cache_dir, sprintf('proposal_boxes_%04d.mat',ii)));
    aboxes_conv4{ii} = ld.aboxes4_save;
    aboxes_conv5{ii} = ld.aboxes5_save;
    aboxes_conv6{ii} = ld.aboxes6_save;
end
save(fullfile(cache_dir, ['proposal_boxes_' 'WIDERFACE_test' '_thr_60_60_60']), 'aboxes_conv4', 'aboxes_conv5','aboxes_conv6', '-v7.3');

end