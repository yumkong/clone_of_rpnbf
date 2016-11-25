function dataset = widerface_all(dataset, usage, use_flip, event_num, cache_dir, model_name_base)
% Pascal voc 2007 trainval set
% set opts.imdb_train opts.roidb_train 
% or set opts.imdb_test opts.roidb_train

% change to point to widerface dataset (win and linux has different dir)
if ispc
    devkit = 'D:\\datasets\\WIDERFACE';
elseif isunix
    devkit = '/usr/local/data/yuguang/dataset/wider_face';
end

%cache_dir = 'cache_data';
%mkdir_if_missing(cache_dir);
    
switch usage
    case {'train'}
        %dataset.imdb_train    = {  imdb_from_widerface(devkit, 'trainval', use_flip) };
        %dataset.roidb_train   = cellfun(@(x) x.roidb_func(x), dataset.imdb_train, 'UniformOutput', false);
        [dataset.imdb_train, dataset.roidb_train] = imdb_from_widerface_final3_flip(devkit, 'trainval', use_flip, cache_dir, model_name_base, event_num);
        %[dataset.imdb_train, dataset.roidb_train] = imdb_from_widerface(devkit, 'trainval', use_flip, cache_dir, model_name_base, event_num);
    case {'test'}
        %dataset.imdb_test     = imdb_from_widerface(devkit, 'test', use_flip) ;
        %dataset.roidb_test    = dataset.imdb_test.roidb_func(dataset.imdb_test);
        [dataset.imdb_test, dataset.roidb_test] = imdb_from_widerface(devkit, 'test', false, cache_dir, model_name_base, event_num);
    otherwise
        error('usage = ''train'' or ''test''');
end

end