function rects = pseudoNMS_v8(candi_rects, nms_option, img)

% overlapping threshold for grouping nearby detections
overlappingThreshold = 0.9; %0.7:0.5233, 0.8:0.5218, 0.6:0.5233 %1012 0.7-->0.75
overlappingThreshold2 = 0.8; %0.5: 0.5233, 0.6: 0.5420   %1012: 0.6 --> 0.7
overlappingThreshold3 = 0.2;
embeddingThreshold = 0.55; %0.5: 0.5279  0.55: 0.5337 
large_small_ratio = 0.2; %0.1: 0.5271  %0.2: 0.5279
% candi_rects: N x 5 matrix, each row: [x1 y1 x2 y2 score]

%1019 added: first remove negative scoring rects
if ~isempty(candi_rects)
    candi_rects = candi_rects(candi_rects(:,5) > 0, :);
end

if isempty(candi_rects)
    rects = [];
    return;
end

%1016 added: when option 0, do nothing
if nms_option == 0
    rects = candi_rects;
    return;
end

nms_debug = false;

%============== first: do nms for nearby regions
area_all = (candi_rects(:,3) - candi_rects(:,1) + 1) .* (candi_rects(:,4) - candi_rects(:,2) + 1);
numCandidates = size(candi_rects, 1); % update number of regions
predicate = eye(numCandidates); % i and j belong to the same group if predicate(i,j) = 1

% mark nearby detections
for i = 1 : numCandidates
    for j = i + 1 : numCandidates
        h = min(candi_rects(i,4), candi_rects(j,4)) - max(candi_rects(i,2), candi_rects(j,2)) + 1;
        w = min(candi_rects(i,3), candi_rects(j,3)) - max(candi_rects(i,1), candi_rects(j,1)) + 1;
        if h <= 0 || w <= 0
           continue; 
        end
        s = max(h,0) * max(w,0);
        % 1006 changed to make it more strict
        if s / (area_all(i) + area_all(j) - s) >= overlappingThreshold  %iou >= 0.9
            predicate(i,j) = 1;
            predicate(j,i) = 1;
        end
    end
end

% merge nearby detections
[label, numCandidates] = Partition(predicate);
rects = zeros(numCandidates, 5); % output rects

for i = 1 : numCandidates
    index = find(label == i);
    %weight = Logistic([candi_rects(index).score]');
    % now a row vector
    %weight = candi_rects(index, 5)';
    weight = candi_rects(index, 5);
    rects(i,5) = sum( weight );  %1202: max --> sum
    % 1024 masked
    %weight = weight.^ 3; %make big score bigger and small score smaller
    
	%normalize weight
    weight = weight / sum(weight);
    
    ave_center_x = weight' * (candi_rects(index, 1) + candi_rects(index, 3))/2;
    ave_center_y = weight' * (candi_rects(index, 2) + candi_rects(index, 4))/2;
    ave_w = weight' * (candi_rects(index, 3) - candi_rects(index, 1) + 1);
    ave_h = weight' * (candi_rects(index, 4) - candi_rects(index, 2) + 1);
    %1007 no rounding, single value is ok
    rects(i,1) = ave_center_x - (ave_w-1)/2; %round(ave_center_x - (ave_w-1)/2);
    rects(i,2) = ave_center_y - (ave_h-1)/2; %round(ave_center_y - (ave_h-1)/2);
    rects(i,3) = ave_center_x + (ave_w-1)/2; %round(ave_center_x + (ave_w-1)/2);
    rects(i,4) = ave_center_y + (ave_h-1)/2; %round(ave_center_y + (ave_h-1)/2);
    %rects(i,1:4) = candi_rects(index(idx), 1:4);
end

if nms_debug
    figure(3),clf;
    imshow(img);  %im(img)
    hold on
    bbs = rects;
    bbs(:, 3) = bbs(:, 3) - bbs(:, 1) + 1;
    bbs(:, 4) = bbs(:, 4) - bbs(:, 2) + 1;
    bbApply('draw',bbs,'c');
    hold off
end
% ================ 2nd: begin the 2nd round of NMS
if nms_option >=2
    
    candi_rects = rects;
    area_all = (candi_rects(:,3) - candi_rects(:,1) + 1) .* (candi_rects(:,4) - candi_rects(:,2) + 1);
    numCandidates = size(candi_rects, 1); % update number of regions
    predicate = eye(numCandidates); % i and j belong to the same group if predicate(i,j) = 1

    % mark nearby detections
    for i = 1 : numCandidates
        for j = i + 1 : numCandidates
            h = min(candi_rects(i,4), candi_rects(j,4)) - max(candi_rects(i,2), candi_rects(j,2)) + 1;
            w = min(candi_rects(i,3), candi_rects(j,3)) - max(candi_rects(i,1), candi_rects(j,1)) + 1;
            if h <= 0 || w <= 0
               continue; 
            end
            s = max(h,0) * max(w,0);
            % 1006 changed to make it more strict
            % 1 >= intersection / single image area >= 0.8 (one boxes is almost nested in another, also the boxes area ratio should 1:5 ~ 1:1)
            if ((s / area_all(i) >= overlappingThreshold2 && s / area_all(i) < 1) || (s / area_all(j) >= overlappingThreshold2 && s / area_all(j) <1)) ...
                    &&(area_all(i)/ area_all(j) >= overlappingThreshold3 && area_all(j)/ area_all(i) >= overlappingThreshold3)
                predicate(i,j) = 1;
                predicate(j,i) = 1;
            end
        end
    end

    % merge nearby detections
    [label, numCandidates] = Partition(predicate);
    rects = []; %zeros(numCandidates, 5); % output rects

    for i = 1 : numCandidates
        % index of the i-th cluster
        index = find(label == i);
        % 1024 find a bug: '>' should change to '>='
        if numel(index) == 1
            % direct assign for singleton rect
            %rects(i, :) = candi_rects(index, :);
            rects = cat(1, rects, candi_rects(index, :));
        else
            % index -- cluster index
            weight = candi_rects(index, 5);           
            mean_weight = mean(weight);
            % get rid of low scoring rects
            new_idx = index(weight >= 0.5*mean_weight);
            new_rects = candi_rects(new_idx,:);
            % sort the rects by descending score
            [~, idx1] = sort(new_rects(:,5), 'descend');
            new_rects = new_rects(idx1,:);
            combined_flag = false(length(new_idx),1);
            %1009 after thresholding, only one elements are left
            if numel(new_idx) == 1
                rects = cat(1, rects, new_rects);
                continue; 
            end
            cnt = 0;
            local_area_all = (new_rects(:,3) - new_rects(:,1) + 1) .* (new_rects(:,4) - new_rects(:,2) + 1);
            for ii = 1:length(new_idx)
                if ~combined_flag(ii)
                    rects_one = new_rects(ii, :);
                    h = min(rects_one(4), new_rects(:,4)) - max(rects_one(2), new_rects(:,2)) + 1;
                    w = min(rects_one(3), new_rects(:,3)) - max(rects_one(1), new_rects(:,1)) + 1;
                    s = max(h,0) .* max(w,0);
                    % 1006 changed to make it more strict
                    nearby_idx = s ./ (local_area_all(ii) + local_area_all - s) >= 0.7;
                    combined_flag(nearby_idx) = true;
                    % use ii to represent all nearby boxes
                    rects = cat(1, rects, new_rects(ii, :));
                    cnt = cnt+1;
                    combined_flag(ii) = true;
                end
                if cnt>=2
                   break; %only keep two bboxes 
                end
            end
            % add the last un-combined rect
%             if ~combined_flag(end)
%                 rects = cat(1, rects, new_rects(end, :));
%                 combined_flag(end) = true;
%             end
            % assertion is true if none of the elements of combined_flag are zero
            %assert(all(combined_flag));
            
%             %rects(i,5) = max( weight );  %1006: sum --> max
%             [~, widx] = sort(weight, 'descend');
%             % index of the largest weight and 2nd largest weight 
%             idx_one = widx(1);
%             idx_two = widx(2);
%             % corresponding rects
%             rects_one = candi_rects(index(idx_one), :);
%             rects_two = candi_rects(index(idx_two), :);
%             
%             center_dist_x = 0.5*(rects_one(3) + rects_one(1)) - 0.5*(rects_two(3) + rects_two(1));
%             center_dist_y = 0.5*(rects_one(4) + rects_one(2)) - 0.5*(rects_two(4) + rects_two(2));
%             % the center distance should be less than or equal to 0.5 x
%             % size of the highest scoring bbox
%             if abs(center_dist_x) <= 0.5*(rects_one(3) - rects_one(1) + 1) && abs(center_dist_y) <= 0.5*(rects_one(4) - rects_one(2) + 1)
%                 %normalize weight
%                 weight = weight / sum(weight);
%                 ave_center_x = weight' * (candi_rects(index, 1) + candi_rects(index, 3))/2;
%                 ave_center_y = weight' * (candi_rects(index, 2) + candi_rects(index, 4))/2;
%                 ave_w = weight' * (candi_rects(index, 3) - candi_rects(index, 1) + 1);
%                 ave_h = weight' * (candi_rects(index, 4) - candi_rects(index, 2) + 1);
%                 %1007 no rounding, single value is ok
%                 rects(i,1) = ave_center_x - (ave_w-1)/2; %round(ave_center_x - (ave_w-1)/2);
%                 rects(i,2) = ave_center_y - (ave_h-1)/2; %round(ave_center_y - (ave_h-1)/2);
%                 rects(i,3) = ave_center_x + (ave_w-1)/2; %round(ave_center_x + (ave_w-1)/2);
%                 rects(i,4) = ave_center_y + (ave_h-1)/2; %round(ave_center_y + (ave_h-1)/2);
%             else
%                 % if the center distance is too large, save the largest two
%                 % bbox respectively
%                 rects(i, :) = rects_one;
%                 % append the 2nd high scoring boxes at the end of output rects
%                 rects = cat(1, rects, rects_two);
%             end
        
        end
    end

end

if nms_debug
    figure(4),clf;
    imshow(img);  %im(img)
    hold on
    bbs = rects;
    bbs(:, 3) = bbs(:, 3) - bbs(:, 1) + 1;
    bbs(:, 4) = bbs(:, 4) - bbs(:, 2) + 1;
    bbApply('draw',bbs,'c');
    hold off
end
%clear candi_rects;

% ============== 3rd: find embeded rectangles (main try)
if nms_option >=3
    
    candi_rects = rects;
    % 1024 solve a bug by adding this line:
    numCandidates = size(candi_rects, 1);
    predicate = eye(numCandidates);
    area_rect = (candi_rects(:,3) - candi_rects(:,1) + 1) .* (candi_rects(:,4) - candi_rects(:,2) + 1);
    for i = 1 : numCandidates
        for j = i + 1 : numCandidates
            h = min(candi_rects(i,4), candi_rects(j,4)) - max(candi_rects(i,2), candi_rects(j,2)) + 1;
            w = min(candi_rects(i,3), candi_rects(j,3)) - max(candi_rects(i,1), candi_rects(j,1)) + 1;
            if h <= 0 || w <= 0
               continue; 
            end
            s = max(h,0) * max(w,0);

            if s / area_rect(i) >= embeddingThreshold || s / area_rect(j) >= embeddingThreshold
                % the larger box should less or equal to 3 times of the size of
                % the smaller one
                if area_rect(i) / area_rect(j) >= large_small_ratio && area_rect(j) / area_rect(i) >= large_small_ratio
                    predicate(i,j) = true;
                    predicate(j,i) = true;
                end
            end
        end
    end
    
    % merge nearby detections
    [label, numCandidates] = Partition(predicate);
    rects = zeros(numCandidates, 5); % output rects

    for i = 1 : numCandidates
        index = find(label == i);
        %weight = Logistic([candi_rects(index).score]');
        % now a row vector
        %weight = candi_rects(index, 5)';
        weight = candi_rects(index, 5);
        rects(i,5) = max( weight );  %1006: sum --> max
        %weight = weight .^ 3; %make big score bigger and small score smaller

        %normalize weight
        weight = weight / sum(weight);

        ave_center_x = weight' * (candi_rects(index, 1) + candi_rects(index, 3))/2;
        ave_center_y = weight' * (candi_rects(index, 2) + candi_rects(index, 4))/2;
        ave_w = weight' * (candi_rects(index, 3) - candi_rects(index, 1) + 1);
        ave_h = weight' * (candi_rects(index, 4) - candi_rects(index, 2) + 1);
        %1007 no rounding, single value is ok
        rects(i,1) = ave_center_x - (ave_w-1)/2; %round(ave_center_x - (ave_w-1)/2);
        rects(i,2) = ave_center_y - (ave_h-1)/2; %round(ave_center_y - (ave_h-1)/2);
        rects(i,3) = ave_center_x + (ave_w-1)/2; %round(ave_center_x + (ave_w-1)/2);
        rects(i,4) = ave_center_y + (ave_h-1)/2; %round(ave_center_y + (ave_h-1)/2);
        %rects(i,1:4) = candi_rects(index(idx), 1:4);
    end

end

%     % merge nearby detections
%     [label, numCandidates] = Partition(predicate);
%     %rects = zeros(numCandidates, 5); % output rects
%     rects = []; % output rects
%     for i = 1 : numCandidates
%         index = find(label == i);
%         if numel(index) == 1
%             % direct assign for singleton rect
%             %rects(i, :) = candi_rects(index, :);
%             rects = cat(1, rects, candi_rects(index, :));
%         else
%             combined_flag = false(length(index),1);
%             new_rects = candi_rects(index,:);
%             % sort the rects by descending score
%             [~, idx1] = sort(new_rects(:,5), 'descend');
%             new_rects = new_rects(idx1,:);
%             % index -- cluster index
%             local_w_all = new_rects(:,3) - new_rects(:,1) + 1;
%             local_h_all = new_rects(:,4) - new_rects(:,2) + 1;
%             local_x_all = (new_rects(:,3) + new_rects(:,1))/2;
%             local_y_all = (new_rects(:,4) + new_rects(:,2))/2;
%             if std(local_x_all) >= 0.3 * mean(local_w_all) || std(local_y_all) >= 0.3 * mean(local_h_all)
%                 stop_num = 2;  % since std is large, guess there are 2 faces
%             else
%                 stop_num = 1;  % guess there is only one face
%             end
%  
%             cnt = 0;
%             local_area_all = (new_rects(:,3) - new_rects(:,1) + 1) .* (new_rects(:,4) - new_rects(:,2) + 1);
%             for ii = 1:length(index)
%                 if ~combined_flag(ii)
%                     rects_one = new_rects(ii, :);
%                     h = min(rects_one(4), new_rects(:,4)) - max(rects_one(2), new_rects(:,2)) + 1;
%                     w = min(rects_one(3), new_rects(:,3)) - max(rects_one(1), new_rects(:,1)) + 1;
%                     s = max(h,0) .* max(w,0);
%                     % 1006 changed to make it more strict
%                     nearby_idx = s ./ (local_area_all(ii) + local_area_all - s) >= 0.7;
%                     combined_flag(nearby_idx) = true;
%                     % use ii to represent all nearby boxes
%                     rects = cat(1, rects, new_rects(ii, :));
%                     cnt = cnt+1;
%                     combined_flag(ii) = true;
%                 end
%                 if cnt >= stop_num
%                    break; %only keep one or two bboxes 
%                 end
%             end
%         end
%     end
% 
% end
if nms_debug
    figure(5),clf;
    imshow(img);  %im(img)
    hold on
    bbs = rects;
    bbs(:, 3) = bbs(:, 3) - bbs(:, 1) + 1;
    bbs(:, 4) = bbs(:, 4) - bbs(:, 2) + 1;
    bbApply('draw',bbs,'c');
    hold off
end
end