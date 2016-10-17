function anchors = proposal_generate_24anchors(cache_name, varargin)
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
    ip.addParamValue('base_size',       10,             @isscalar); %1009: 16 -->12
    % ratio list of anchors
    ip.addParamValue('ratios',          [1],    @ismatrix);%0807: [0.5, 1, 2] --> [1]
    % scale list of anchors
    ip.addParamValue('scales',          2.^[0:5],       @ismatrix); %0807: 2.^[3:5]-->2.^[0:5]
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
        ld                      = load(anchor_cache_file);
        anchors                 = ld.anchors;
    catch
        base_anchor             = [1, 1, opts.base_size, opts.base_size];
        ratio_anchors           = ratio_jitter(base_anchor, opts.ratios);
        anchors                 = cellfun(@(x) scale_jitter(x, opts.scales), num2cell(ratio_anchors, 2), 'UniformOutput', false);
        
        % extend anchors
        tmp_anchors = anchors{1};
        % extend the length-10 anchor to 9 anchors
        extent_anchor10_delta = [-5 -5 -5 -5; 0 -5 0 -5; 5 -5 5 -5; ...
                                 -5  0 -5  0; 0 0  0  0; 5 0  5  0; ...
                                 -5  5 -5  5; 0  5 0  5; 5 5  5  5];
        anchor10_ext = bsxfun(@plus, tmp_anchors(1,:), extent_anchor10_delta);
        % extend the length-16 anchor to 4 anchors
        extent_anchor16_delta = [0 0 0 0; 0 8 0 8; ...
                                 8 0 8 0; 8 8 8 8];
        anchor16_ext = bsxfun(@plus, tmp_anchors(2,:), extent_anchor16_delta);
        % extend anchors
        tmp_anchors(1:2,:) = [];
        tmp_anchors = cat(1, anchor10_ext, anchor16_ext, tmp_anchors);
        anchors{1} = tmp_anchors;
        
        anchors                 = cat(1, anchors{:});
        if ~opts.ignore_cache
            save(anchor_cache_file, 'anchors');
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

function anchors = scale_jitter(anchor, scales)
    scales = scales(:);

    w = anchor(3) - anchor(1) + 1;
    h = anchor(4) - anchor(2) + 1;
    x_ctr = anchor(1) + (w - 1) / 2;
    y_ctr = anchor(2) + (h - 1) / 2;

    %liu@1009: scales is the real size rather than a ratio
    ws = scales; %w * scales;
    hs = scales; %h * scales;
    
    anchors = [x_ctr - (ws - 1) / 2, y_ctr - (hs - 1) / 2, x_ctr + (ws - 1) / 2, y_ctr + (hs - 1) / 2];
end

