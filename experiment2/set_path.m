function set_path()
% startup()
% --------------------------------------------------------
% Faster R-CNN
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------
    %0711 changed
    %curdir = fileparts(mfilename('fullpath'));
    % this is the father path
    curdir = fileparts(fileparts(mfilename('fullpath')));
    %addpath(genpath(fullfile(curdir, 'utils')));
    %addpath(genpath(fullfile(curdir, 'functions')));
    %addpath(genpath(fullfile(curdir, 'bin')));
    %0711 changed
    %addpath(genpath(fullfile(curdir, 'experiments')));
    addpath(genpath(fullfile(curdir, 'experiment2'))); %has '+helper','bin', 'export_fig' folder
    %addpath(genpath(fullfile(curdir, 'imdb')));
    
    caffe_path = fullfile(curdir, 'external', 'caffe', 'matlab');
    if exist(caffe_path, 'dir') == 0
        error('matcaffe is missing from external/caffe/matlab; See README.md');
    end
    addpath(genpath(caffe_path));

    %0711 changed
    %mkdir_if_missing(fullfile(curdir, 'output'));
    helper.mkdir_if_missing(fullfile(curdir, 'output2'));

    helper.mkdir_if_missing(fullfile(curdir, 'models'));

    fprintf('rpn startup done\n');
end
