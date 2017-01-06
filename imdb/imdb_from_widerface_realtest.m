function [imdb] = imdb_from_widerface_realtest(root_dir, image_set, flip, cache_dir, model_name_base, event_num)
%function imdb_from_widerface(devkit, 'trainval', use_flip)

switch image_set
    case {'realtest'}
        if event_num == -1
            data_num_str = 'all';
        elseif event_num > 0
            data_num_str = sprintf('e1-e%d', event_num);
        end
        cache_imdb = fullfile(cache_dir, sprintf('realtest_imdb_%s_%s_raw.mat',model_name_base, data_num_str));  %imdb
        devpath = fullfile('WIDER_test','images');
        doc_dir = fullfile('wider_face_split','wider_face_test');
        name = 'WIDERFACE_realtest';
    otherwise
        error('usage = ''realtest''');
end

try
    load(cache_imdb); %'dataset' -- struct, containing imdb
catch
    imdb = [];
    %(1)
    imdb.name = name;
    %(2)
    rootdir = root_dir;
    imdb.image_dir = fullfile(rootdir, devpath);

    % have 3 cell elements
    % event_list
    % face_bbx_list
    % file_list
    annodoc = load(fullfile(rootdir, doc_dir));

    imgsum_ = 0;
    if event_num <= 0
        eventNum = numel(annodoc.file_list);  % use all events
    else
        eventNum = event_num;  % use the first N events
    end
    % ###########################################
    for eventIter = 1:eventNum
       imgsum_ = imgsum_ + numel(annodoc.file_list{eventIter});
    end

    %(3)
    imdb.image_ids = cell(imgsum_,1);
    
    %eventNum = numel(event_list);
    cnt = 0;
    % crop the faces of the first ten events, and save the results
    %0807 changed to smaller dataset for faster training and debug
    % ###########################################
    for eventIter = 1:eventNum
       event_ = annodoc.event_list{eventIter};
       images_ = annodoc.file_list{eventIter};

       imageNum = numel(images_);
       for imgIter = 1:imageNum
           cnt = cnt + 1;
           %0805: can make directory irrelevant to os: linux a/b, win a\b
           imdb.image_ids{cnt} = fullfile(event_, images_{imgIter});
       end
    end

    %(4)
    imdb.extension = 'jpg';
    %(5)
    imdb.flip = flip;
    %(6) -- no
    %dataset.flip_from
    %(7)
    imdb.classes = {'face'};
    %(8)
    imdb.num_classes = 1;
    %(9)
    imdb.class_to_id = containers.Map(imdb.classes, 1:imdb.num_classes);
    %(10)
    imdb.class_ids = 1:imdb.num_classes;
    %(11) --no
    %dataset.details
    %(12)
    imdb.eval_func = @imdb_eval_voc;
    %(13)
    imdb.roidb_func = @roidb_from_wider;
    %(14)
    imdb.image_at = @(i) sprintf('%s%c%s.%s', imdb.image_dir, filesep, imdb.image_ids{i}, imdb.extension);

    %(15) height, width
    sizes = zeros(imgsum_, 2);
    for iter = 1:imgsum_
    %    event_ = traindoc.event_list{eventIter};
    %    images_ = traindoc.file_list{eventIter};
       im = imread(imdb.image_at(iter));
       sizes(iter, :) = [size(im, 1) size(im, 2)];
    end
    imdb.sizes = sizes;
    %save it
    save(cache_imdb, 'imdb', '-v7.3');
end

end  %of function
