function [final_box, tm_cost] = mpfpn_detect(conf, caffe_net, im)
tic
[boxes_s4, scores_s4, boxes_s8, scores_s8, boxes_s16, scores_s16] = proposal_im_detect_me(conf, caffe_net, im);
tm_cost = toc;
aboxes4 = [];
aboxes5 = [];
aboxes6 = [];
for j = 1:numel(scores_s4)
    %1230 added
    boxes_s4{j} = boxes_s4{j}(scores_s4{j} >= 0.6,:);
    scores_s4{j} = scores_s4{j}(scores_s4{j} >= 0.6,:);  %0101:0.1-->0.55

    boxes_s8{j} = boxes_s8{j}(scores_s8{j} >= 0.6,:);
    scores_s8{j} = scores_s8{j}(scores_s8{j} >= 0.6,:);

    boxes_s16{j} = boxes_s16{j}(scores_s16{j} >= 0.6,:);
    scores_s16{j} = scores_s16{j}(scores_s16{j} >= 0.6,:);

    aboxes4_tmp = [boxes_s4{j}, scores_s4{j}];
    aboxes4 = cat(1, aboxes4, aboxes4_tmp);
    aboxes8_tmp = [boxes_s8{j}, scores_s8{j}];
    aboxes5 = cat(1, aboxes5, aboxes8_tmp);
    aboxes16_tmp = [boxes_s16{j}, scores_s16{j}];
    aboxes6 = cat(1, aboxes6, aboxes16_tmp);
end

%aboxes_conv4              = boxes_filter({aboxes4}, -1, 0.7, -1, false);
%aboxes_conv5              = boxes_filter({aboxes5}, -1, 0.7, -1, false);
%aboxes_conv6              = boxes_filter({aboxes6}, -1, 0.7, -1, false);

aboxes{1} = cat(1, aboxes4(aboxes4(:, end) > 0.65, :),...
                   aboxes5(aboxes5(:, end) > 0.7, :),...
                   aboxes6(aboxes6(:, end) > 0.7, :));
%aboxes_nms{1} = pseudoNMS_v8(aboxes{1}, 3);
% liu@0611: avoid 2 boxes with the same score, nms will keep the
% duplications if they are of the same score
aboxes{1}(:, 5) = aboxes{1}(:, 5) + 1e-2*randn([length(aboxes{1}(:, 5)) 1]);
%aboxes = boxes_filter(aboxes, -1, 0.1, -1, conf.use_gpu); %0.33
aboxes = boxes_filter(aboxes, -1, 0.1, -1, false); %0.33

final_box = aboxes{1};
final_box = final_box(final_box(:,5)>=0.999,:);
if ~isempty(final_box)
    final_box(:,3) = final_box(:,3) - final_box(:,1) + 1;
    final_box(:,4) = final_box(:,4) - final_box(:,2) + 1;
else
    final_box = [];
end

end

