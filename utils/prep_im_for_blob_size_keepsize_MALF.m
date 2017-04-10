function im_scale = prep_im_for_blob_size_keepsize_MALF(im_size, input_scale, min_size, max_size)
    %0124 changed
    %im_size_min = min(im_size(1:2));
    %im_size_max = max(im_size(1:2));
    im_size_min = round(min(im_size(1:2))*input_scale);
    im_size_max = round(max(im_size(1:2))*input_scale);
    %0123 added: if in the range of [min_size, max_size],keep ori size
    if (im_size_min >= min_size) && (im_size_max <= max_size)
        %0410 added to shrink the square image larger than 1024 x 1024
        if im_size(1)*im_size(2) <= 9e5
            im_scale = input_scale;
        else
            im_scale = 0.9;
        end
        return;
    end
    
    %0123 added: if larger than max_size, shrink it
    if im_size_max > max_size
        im_scale = double(max_size) / double(im_size_max) * input_scale; % < 1
    else
        im_scale = input_scale;
    end
    %0410 added
    if im_size(1)*im_size(2)*im_scale*im_scale > 9e5
        %im_scale = im_scale * 0.9;
        im_scale = im_scale * sqrt(9e5/im_size(1)/im_size(2)/im_scale/im_scale);
    end
end