function save_bbox_to_txt(aboxes, image_ids, bbox_save_name)
% Save a bunch of bounding boxes to a text file

% aboxes: a N x 1 cell, each element contains bbox of one image
% image_ids: a N x 1 cell, each element is the absolute path name of one image
% bbox_save_name: name of text file to be saved

% open or create file for reading and writing; discard existing contents
fid = fopen(bbox_save_name, 'w+');
assert(length(image_ids) == size(aboxes, 1));
for i = 1:size(aboxes, 1)
    if ~isempty(aboxes{i})
        sstr = strsplit(image_ids{i}, filesep);
        % [x1 y1 x2 y2] pascal VOC style
        for j = 1:size(aboxes{i}, 1)
            %each row: [image_name score x1 y1 x2 y2]
            fprintf(fid, '%s %f %d %d %d %d\n', sstr{2}, aboxes{i}(j, 5), round(aboxes{i}(j, 1:4)));
        end
    end
end
fclose(fid);

fprintf('saving bboxes to file %s is completed\n', bbox_save_name);