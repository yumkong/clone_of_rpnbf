function [anchors_s4, anchors_s8, anchors_s16] = proposal_generate_anchors_ablation_final_moreanchor(cache_name, varargin)
% anchors = proposal_generate_anchors(cache_name, varargin)
% --------------------------------------------------------
% Faster R-CNN
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------

%% inputs
    ip = inputParser;
    ip.addRequired('cache_name',                        @isstr);

    % the size of the base anchor 
    ip.addParamValue('base_size',       16,             @isscalar);
    % ratio list of anchors
    ip.addParamValue('ratios',          [1],            @ismatrix);%0807: [0.5, 1, 2] --> [1]
    % scale list of anchors
    ip.addParamValue('scales',          2.^[0:5],       @ismatrix); %0807: 2.^[3:5]-->2.^[0:5]
    ip.addParamValue('add_size',        [480],          @ismatrix); %1112 a new param: add anchor size that is != 2^x
    ip.addParamValue('ignore_cache',    false,          @islogical);
    ip.parse(cache_name, varargin{:});
    opts = ip.Results;

%%
    if ~opts.ignore_cache
        %anchor_cache_dir            = fullfile(pwd, 'output', 'rpn_cachedir', cache_name); 
%         anchor_cache_dir            = fullfile(pwd, 'cache_data', cache_name); 
%                                       mkdir_if_missing(anchor_cache_dir);
%         anchor_cache_file           = fullfile(anchor_cache_dir, 'anchors');
        anchor_cache_file           = fullfile(cache_name, 'anchors');
    end
    try
        ld                         = load(anchor_cache_file);
        anchors_s4                 = ld.anchors_s4;
        anchors_s8                 = ld.anchors_s8;
        anchors_s16                = ld.anchors_s16;
    catch
        base_anchor             = [1, 1, opts.base_size, opts.base_size];
        ratio_anchors           = ratio_jitter(base_anchor, opts.ratios);
        anchors                 = cellfun(@(x) scale_jitter(x, opts.scales, opts.add_size), num2cell(ratio_anchors, 2), 'UniformOutput', false);
        anchors                 = cat(1, anchors{:});
        % 1112 added
        anchors_s4 = anchors(1:2, :);  % [8]
        anchors_s8 = anchors(3:9, :);  % [16 32 64]-> [16 32 64 128]
        anchors_s16 = anchors(9:end, :);  % 0227: [128 256 480]->[64 128 256 480]--> [128 256 480]
        if ~opts.ignore_cache
            save(anchor_cache_file, 'anchors_s4', 'anchors_s8', 'anchors_s16');
        end
    end
    
end

function anchors = ratio_jitter(anchor, ratios)
    ratios = ratios(:);
    
    w = anchor(3) - anchor(1) + 1;
    h = anchor(4) - anchor(2) + 1;
    x_ctr = anchor(1) + (w - 1) / 2;
    y_ctr = anchor(2) + (h - 1) / 2;
    size = w * h;
    
    size_ratios = size ./ ratios;
    ws = round(sqrt(size_ratios));
    hs = round(ws .* ratios);
    
    anchors = [x_ctr - (ws - 1) / 2, y_ctr - (hs - 1) / 2, x_ctr + (ws - 1) / 2, y_ctr + (hs - 1) / 2];
end

function anchors = scale_jitter(anchor, scales, add_size)
    scales = scales(:);

    w = anchor(3) - anchor(1) + 1;
    h = anchor(4) - anchor(2) + 1;
    x_ctr = anchor(1) + (w - 1) / 2;
    y_ctr = anchor(2) + (h - 1) / 2;

    %ws = w * scales;
    %hs = h * scales;
    %ws = [w * scales; add_size];
    %hs = [h * scales; add_size];
    % 1121: in ascending order
    ws = sort([w * scales; add_size']);
    hs = sort([h * scales; add_size']);
    
    anchors = [x_ctr - (ws - 1) / 2, y_ctr - (hs - 1) / 2, x_ctr + (ws - 1) / 2, y_ctr + (hs - 1) / 2];
end

