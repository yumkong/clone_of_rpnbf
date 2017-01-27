function im_scale = prep_im_for_blob_size_keepsize(im_size, input_scale, min_size, max_size)
    %0124 changed
    %im_size_min = min(im_size(1:2));
    %im_size_max = max(im_size(1:2));
    im_size_min = round(min(im_size(1:2))*input_scale);
    im_size_max = round(max(im_size(1:2))*input_scale);
    %0123 added: if in the range of [min_size, max_size],keep ori size
    if (im_size_min >= min_size) && (im_size_max <= max_size)
        im_scale = input_scale;
        return;
    end
    
    %0123 added: if larger than max_size, shrink it
    if im_size_max > max_size
        im_scale = double(max_size) / double(im_size_max) * input_scale; % < 1
    else
        im_scale = input_scale;
    end
end