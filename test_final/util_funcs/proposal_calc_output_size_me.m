function [w_s4, h_s4, w_s8, h_s8, w_s16, h_s16] = proposal_calc_output_size_me(conf, test_net_def_file, output_map_save_name)
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
        %input = 100:conf.max_size;
        %input = 100:conf.max_size+10;
        input = 800;
        %input = conf.min_test_length:conf.max_test_length;
        output_w_s4 = nan(size(input));
        output_h_s4 = nan(size(input));
        output_w_s8 = nan(size(input));
        output_h_s8 = nan(size(input));
        output_w_s16 = nan(size(input));
        output_h_s16 = nan(size(input));
        for i = 1:length(input)
            fprintf('calulating input size %d / %d\n',i,conf.max_size);
            s = input(i);
            %liu@0926 changed,because when s>= 1000, s x s is too big to feed to a net
            %im_blob = single(zeros(s, s, 3, 1));
            %im_blob = single(zeros(s, 500, 3, 1));
            %im_blob = single(zeros(s, conf.min_test_length, 3, 1));
            im_blob = single(zeros(s, 128, 3, 1));
            net_inputs = {im_blob};

            % Reshape net's input blobs
            caffe_net.reshape_as_input(net_inputs);
            caffe_net.forward(net_inputs);

            cls_score_s4 = caffe_net.blobs('proposal_cls_prob_s4').get_data();
            output_w_s4(i) = size(cls_score_s4, 1);
            output_h_s4(i) = size(cls_score_s4, 1);
            
            cls_score_s8 = caffe_net.blobs('proposal_cls_prob_s8').get_data();
            output_w_s8(i) = size(cls_score_s8, 1);
            output_h_s8(i) = size(cls_score_s8, 1);
            
            cls_score_s16 = caffe_net.blobs('proposal_cls_prob_s16').get_data();
            output_w_s16(i) = size(cls_score_s16, 1);
            output_h_s16(i) = size(cls_score_s16, 1);
        end

        w_s4 = containers.Map(input, output_w_s4);
        h_s4 = containers.Map(input, output_h_s4);
        w_s8 = containers.Map(input, output_w_s8);
        h_s8 = containers.Map(input, output_h_s8);
        w_s16 = containers.Map(input, output_w_s16);
        h_s16 = containers.Map(input, output_h_s16);
        %0805
        %save(cache_output_width_map, 'output_width_map', '-v7.3');
        %save(cache_output_height_map, 'output_height_map', '-v7.3');
        %0925
        save(output_map_save_name, 'w_s4', 'h_s4', 'w_s8', 'h_s8', 'w_s16', 'h_s16');
        caffe.reset_all(); 
    end
end