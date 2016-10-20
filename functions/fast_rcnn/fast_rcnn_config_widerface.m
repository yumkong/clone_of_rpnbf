function conf = fast_rcnn_config_widerface(varargin)
% conf = fast_rcnn_config(varargin)
% Fast R-CNN configuration
% --------------------------------------------------------
% Fast R-CNN
% Reimplementation based on Python Fast R-CNN (https://github.com/rbgirshick/fast-rcnn)
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------
%
    ip = inputParser;
    
    %% training
    % whether use gpu
    ip.addParamValue('use_gpu',         gpuDeviceCount > 0, ...            
                                                        @islogical);
    % Image scales -- the short edge of input image                                                
    ip.addParamValue('scales',          900,            @ismatrix); %600
    % Max pixel size of a scaled input image
    ip.addParamValue('max_size',        1500,           @isscalar);  %1000
    % Images per batch
    ip.addParamValue('ims_per_batch',   1,              @isscalar); %2-->1
    % Minibatch size
    ip.addParamValue('batch_size',      256,            @isscalar);  %32
    % Fraction of minibatch that is foreground labeled (class > 0)
    ip.addParamValue('fg_fraction',     0.25,           @isscalar);
    % Overlap threshold for a ROI to be considered foreground (if >= fg_thresh)
    ip.addParamValue('fg_thresh',       0.5,            @isscalar);
    % Overlap threshold for a ROI to be considered background (class = 0 if
    % overlap in [bg_thresh_lo, bg_thresh_hi))
    ip.addParamValue('bg_thresh_hi',    0.5,            @isscalar);
    ip.addParamValue('bg_thresh_lo',    0.1,            @isscalar);
    % mean image, in RGB order
    ip.addParamValue('image_means',     128,            @ismatrix);
    % Use horizontally-flipped images during training?
    ip.addParamValue('use_flipped',     false,           @islogical);
    % Vaild training sample (IoU > bbox_thresh) for bounding box regresion
    ip.addParamValue('bbox_thresh',     0.5,            @isscalar);

    % random seed
    ip.addParamValue('rng_seed',        6,              @isscalar);

    
    %% testing
    ip.addParamValue('test_scales',     900,            @isscalar); %600
    ip.addParamValue('test_max_size',   1500,           @isscalar); %1000
    ip.addParamValue('test_nms',        0.3,            @isscalar);
    ip.addParamValue('test_binary',     false,          @islogical);
    
    ip.parse(varargin{:});
    conf = ip.Results;
    
    % if image_means is a file, load it
    if ischar(conf.image_means)
        s = load(conf.image_means);
        s_fieldnames = fieldnames(s);
        assert(length(s_fieldnames) == 1);
        conf.image_means = s.(s_fieldnames{1});
    end
end