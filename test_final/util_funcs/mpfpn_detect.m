function [final_box, tm_cost] = mpfpn_detect(conf, caffe_net, im)
tic
[boxes_conv4, scores_conv4, boxes_conv5, scores_conv5, boxes_conv6, scores_conv6] = proposal_im_detect_me(conf, caffe_net, im);
tm_cost = toc;
scores_conv4 = scores_conv4(scores_conv4 >= 0.5, :);
boxes_conv4 = boxes_conv4(scores_conv4 >= 0.5, :);
scores_conv5 = scores_conv5(scores_conv5 >= 0.55, :);
boxes_conv5 = boxes_conv5(scores_conv5 >= 0.55, :);
scores_conv6 = scores_conv6(scores_conv6 >= 0.55, :);
boxes_conv6 = boxes_conv6(scores_conv6 >= 0.55, :);

aboxes4 = [boxes_conv4, scores_conv4];
aboxes5 = [boxes_conv5, scores_conv5];
aboxes6 = [boxes_conv6, scores_conv6];
aboxes_conv4              = boxes_filter({aboxes4}, -1, 0.7, -1, conf.use_gpu);
aboxes_conv5              = boxes_filter({aboxes5}, -1, 0.7, -1, conf.use_gpu);
aboxes_conv6              = boxes_filter({aboxes6}, -1, 0.7, -1, conf.use_gpu);

aboxes{1} = cat(1, aboxes_conv4{1}(aboxes_conv4{1}(:, end) > 0.65, :),...
                   aboxes_conv5{1}(aboxes_conv5{1}(:, end) > 0.7, :),...
                   aboxes_conv6{1}(aboxes_conv6{1}(:, end) > 0.7, :));
%aboxes_nms{1} = pseudoNMS_v8(aboxes{1}, 3);
aboxes = boxes_filter(aboxes, -1, 0.1, -1, conf.use_gpu); %0.33

final_box = aboxes{1};
if ~isempty(final_box)
    final_box(:,3) = final_box(:,3) - final_box(:,1) + 1;
    final_box(:,4) = final_box(:,4) - final_box(:,2) + 1;
else
    final_box = [];
end
