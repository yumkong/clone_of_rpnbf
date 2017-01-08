function roidb = roidb_from_proposal_score_realtest(imdb, roidb, regions, varargin)
% roidb = roidb_from_proposal_score(imdb, roidb, regions, varargin)
% --------------------------------------------------------
% RPN_BF
% Copyright (c) 2016, Liliang Zhang
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------

ip = inputParser;
ip.addRequired('imdb', @isstruct);
ip.addRequired('roidb', @isstruct);
ip.addRequired('regions', @isstruct);
ip.parse(imdb, roidb, regions, varargin{:});
opts = ip.Results;

assert(strcmp(opts.roidb.name, opts.imdb.name));
rois = opts.roidb.rois;

% add new proposal boxes
for i = 1:length(rois)
    %1007 added, to judge whether detected boxes is zero
    if ~isempty(opts.regions.boxes{i})
        boxes = opts.regions.boxes{i}(:, 1:4);
        scores = opts.regions.boxes{i}(:, end);
    else
       boxes = [];
       scores = [];
    end

    rois(i).boxes = cat(1, rois(i).boxes, boxes);
    rois(i).scores = cat(1, rois(i).scores, scores);

end
%fprintf('gt_num: %d\n', gt_num);

roidb.rois = rois;

end