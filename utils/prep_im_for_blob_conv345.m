function [im, im_scale] = prep_im_for_blob_conv345(im, im_means, target_size, max_size)
    im = single(im);
    
    if ~isa(im, 'gpuArray')
        try
            im = bsxfun(@minus, im, im_means);
        catch
            im_means = imresize(im_means, [size(im, 1), size(im, 2)], 'bilinear', 'antialiasing', false);    
            im = bsxfun(@minus, im, im_means);
        end
        im_scale = prep_im_for_blob_size(size(im), target_size, max_size);

        target_size = round([size(im, 1), size(im, 2)] * im_scale);
        im = imresize(im, target_size, 'bilinear', 'antialiasing', false);
        %1206 added
        %0116 changed
        %new_target_size = ceil(target_size / 8) * 8;
        new_target_size = ceil(target_size / 16) * 16;
        botpad = new_target_size(1) - target_size(1);
        rightpad = new_target_size(2) - target_size(2);
        im = imPad(im , [0 botpad 0 rightpad], 0);
    else
        % for im as gpuArray
        try
            im = bsxfun(@minus, im, im_means);
        catch
            im_means_scale = max(double(size(im, 1)) / size(im_means, 1), double(size(im, 2)) / size(im_means, 2));
            im_means = imresize(im_means, im_means_scale);    
            y_start = floor((size(im_means, 1) - size(im, 1)) / 2) + 1;
            x_start = floor((size(im_means, 2) - size(im, 2)) / 2) + 1;
            im_means = im_means(y_start:(y_start+size(im, 1)-1), x_start:(x_start+size(im, 2)-1));
            im = bsxfun(@minus, im, im_means);
        end
        
        im_scale = prep_im_for_blob_size(size(im), target_size, max_size);
        im = imresize(im, im_scale);
    end
end