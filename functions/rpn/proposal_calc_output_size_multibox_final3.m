function [output_width_map_conv4, output_height_map_conv4, ...
          output_width_map_conv5, output_height_map_conv5, ...
          output_width_map_conv6, output_height_map_conv6] = proposal_calc_output_size_multibox_final3(conf, test_net_def_file, output_map_save_name)
% [output_width_map, output_height_map] = proposal_calc_output_size(conf, test_net_def_file)
% --------------------------------------------------------
% Faster R-CNN
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------

%     caffe.init_log(fullfile(pwd, 'caffe_log'));
    %0805 added
    %cache_output_width_map = 'output_width_map_vgg16.mat';
    %cache_output_height_map = 'output_height_map_vgg16.mat';

    try
        %load(cache_output_width_map);
        %load(cache_output_height_map);
        load(output_map_save_name);  %output_width_map, output_height_map
    catch

        caffe_net = caffe.Net(test_net_def_file, 'test');

         % set gpu/cpu
        if conf.use_gpu
            caffe.set_mode_gpu();
        else
            caffe.set_mode_cpu();
        end

        input = 100:conf.max_size;
        output_w_conv4 = nan(size(input));
        output_h_conv4 = nan(size(input));
        output_w_conv5 = nan(size(input));
        output_h_conv5 = nan(size(input));
        output_w_conv6 = nan(size(input));
        output_h_conv6 = nan(size(input));
        for i = 1:length(input)
            fprintf('calulating input size %d / %d\n',i,conf.max_size);
            s = input(i);
            %liu@0926 changed,because when s>= 1000, s x s is too big to feed to a net
            %im_blob = single(zeros(s, s, 3, 1));
            im_blob = single(zeros(s, 500, 3, 1));
            net_inputs = {im_blob};

            % Reshape net's input blobs
            caffe_net.reshape_as_input(net_inputs);
            caffe_net.forward(net_inputs);

            cls_score_conv4 = caffe_net.blobs('proposal_cls_score_conv4').get_data();
            % w x h x ch x num
            output_w_conv4(i) = size(cls_score_conv4, 1);
            %liu@0926 changed, see above
            output_h_conv4(i) = size(cls_score_conv4, 1);
            %output_h(i) = size(cls_score, 2);
            
            cls_score_conv5 = caffe_net.blobs('proposal_cls_score_conv5').get_data();
            % w x h x ch x num
            output_w_conv5(i) = size(cls_score_conv5, 1);
            output_h_conv5(i) = size(cls_score_conv5, 1);
            
            cls_score_conv6 = caffe_net.blobs('proposal_cls_score_conv6').get_data();
            % w x h x ch x num
            output_w_conv6(i) = size(cls_score_conv6, 1);
            output_h_conv6(i) = size(cls_score_conv6, 1);
        end

        output_width_map_conv4 = containers.Map(input, output_w_conv4);
        output_height_map_conv4 = containers.Map(input, output_h_conv4);
        output_width_map_conv5 = containers.Map(input, output_w_conv5);
        output_height_map_conv5 = containers.Map(input, output_h_conv5);
        output_width_map_conv6 = containers.Map(input, output_w_conv6);
        output_height_map_conv6 = containers.Map(input, output_h_conv6);
        %0805
        %save(cache_output_width_map, 'output_width_map', '-v7.3');
        %save(cache_output_height_map, 'output_height_map', '-v7.3');
        %0925
        save(output_map_save_name, 'output_width_map_conv4', 'output_height_map_conv4',...
                'output_width_map_conv5', 'output_height_map_conv5', 'output_width_map_conv6', 'output_height_map_conv6');
        caffe.reset_all(); 
    end
end