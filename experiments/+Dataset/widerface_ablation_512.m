function dataset = widerface_ablation_512(dataset, usage, use_flip, event_pool, cache_dir, model_name_base)
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
        %unlike final3_flip, here still use upside-down, left_right, rot90,
        %rot90+fliplr
        [dataset.imdb_train, dataset.roidb_train] = imdb_from_widerface_ablation_512(devkit, 'trainval', use_flip, cache_dir, model_name_base, event_pool);
    case {'test'}
        [dataset.imdb_test, dataset.roidb_test] = imdb_from_widerface_ablation_512(devkit, 'test', false, cache_dir, model_name_base, event_pool);
    otherwise
        error('usage = ''train'' or ''test''');
end

end