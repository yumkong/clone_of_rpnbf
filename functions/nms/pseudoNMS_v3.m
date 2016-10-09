function rects = pseudoNMS_v3(candi_rects, nms_option)

% overlapping threshold for grouping nearby detections
overlappingThreshold = 0.7; %0.7:0.5233, 0.8:0.5218, 0.6:0.5233 
overlappingThreshold2 = 0.6; %0.5: 0.5233, 0.6: 0.5420 
overlappingThreshold3 = 0.2;
embeddingThreshold = 0.55; %0.5: 0.5279  0.55: 0.5337 
large_small_ratio = 0.2; %0.1: 0.5271  %0.2: 0.5279
% candi_rects: N x 5 matrix, each row: [x1 y1 x2 y2 score]
if isempty(candi_rects)
    rects = [];
    return;
end

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
        if s / (area_all(i) + area_all(j) - s) >= overlappingThreshold
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
    rects(i,5) = sum( weight );  %1006: sum --> max
    weight = weight .^ 3; %make big score bigger and small score smaller
    
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
            %if s / (area_all(i) + area_all(j) - s) >= overlappingThreshold2
            if (s / area_all(i) >= overlappingThreshold2 || s / area_all(j) >= overlappingThreshold2) && ...
                    (area_all(i)/ area_all(j) >= overlappingThreshold3 && area_all(j)/ area_all(i) >= overlappingThreshold3)
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
        
        if numel(index) > 1
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
            
            for ii = 1:length(new_idx)-1
                if ~combined_flag(ii)
                    
                    for jj = ii+1:length(new_idx)
                        rects_one = new_rects(ii, :);
                        w_one = rects_one(3) - rects_one(1) + 1;
                        h_one = rects_one(4) - rects_one(2) + 1;
                        if ~combined_flag(jj)
                            rects_two = new_rects(jj, :);
                            w_two = rects_two(3) - rects_two(1) + 1;
                            h_two = rects_two(4) - rects_two(2) + 1;
                            % absolute value
                            center_dist_x = abs(0.5*(rects_one(3) + rects_one(1) - rects_two(3) - rects_two(1)));
                            center_dist_y = abs(0.5*(rects_one(4) + rects_one(2) - rects_two(4) - rects_two(2)));
                            if center_dist_x <= 0.5*max(w_one, w_two) && center_dist_y <= 0.5*max(h_one, h_two)
                                combined_flag(jj) = true;
                                % combine two rects
                                weight = [rects_one(5) rects_two(5)] / sum([rects_one(5) rects_two(5)]);
                                ave_center_x = weight * [rects_one(1) + rects_one(3); rects_two(1) + rects_two(3)]/2;
                                ave_center_y = weight * [rects_one(2) + rects_one(4); rects_two(2) + rects_two(4)]/2;
                                ave_w = weight * [rects_one(3)-rects_one(1)+1; rects_two(3)-rects_two(1)+1];
                                ave_h = weight * [rects_one(4)-rects_one(2)+1; rects_two(4)-rects_two(2)+1];
                                %1007 no rounding, single value is ok
                                new_rects(ii, :) = [ave_center_x-(ave_w-1)/2  ave_center_y-(ave_h-1)/2  ave_center_x+(ave_w-1)/2  ave_center_y+(ave_h-1)/2  rects_one(5)+rects_two(5)];
                                %new_rects(ii, :) = [ave_center_x-(ave_w-1)/2  ave_center_y-(ave_h-1)/2  ave_center_x+(ave_w-1)/2  ave_center_y+(ave_h-1)/2  max([rects_one(5) rects_two(5)])];
                            end
                        end
                    end
                    rects = cat(1, rects, new_rects(ii, :));
                    combined_flag(ii) = true;
                end
            end
            % add the last un-combined rect
            if ~combined_flag(end)
                rects = cat(1, rects, new_rects(end, :));
                combined_flag(end) = true;
            end
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
        else
            % direct assign for singleton rect
            %rects(i, :) = candi_rects(index, :);
            rects = cat(1, rects, candi_rects(index, :));
        end
    end

end

%clear candi_rects;

% ============== 3rd: find embeded rectangles (main try)
if nms_option >=3
    
    candi_rects = rects;
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

end