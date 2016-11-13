function test_flip()
close all
img = imread('001150.jpg');
[hei, wid, ~] = size(img);
bbox = [302 2 351 71];

figure(1), imshow(img);
drawbox(bbox);

% rot90 (counterclockwise 90)
bbox_rot90 = bbox(:, [2 1 4 3]);
bbox_rot90(:,[2 4]) = wid - bbox(:,[3 1]);
figure(2), imshow(rot90(img));
drawbox(bbox_rot90);

% fliplr
bbox_lr = bbox;
bbox_lr(:,[1 3]) = wid - bbox(:, [3 1]);
figure(3), imshow(fliplr(img));
drawbox(bbox_lr);

% flipud
bbox_ud = bbox;
bbox_ud(:,[2 4]) = hei - bbox(:, [4 2]);
figure(4), imshow(flipud(img));
drawbox(bbox_ud);

% rot90-fliplr (clockwise 90)
bbox_rot90_lr = [hei wid hei wid] - bbox(:, [4 3 2 1]); %bbox_rot90;
%bbox_rot90_lr = hei - bbox_rot90(:, [3 1]);
figure(5), imshow(fliplr(rot90(img)));
drawbox(bbox_rot90_lr);

end

function drawbox(bbox)
    dbox = [bbox(1) bbox(2) bbox(3)-bbox(1)+1 bbox(4)-bbox(2)+1];
    rectangle('Position',dbox, 'EdgeColor', [0 1 0])
end

