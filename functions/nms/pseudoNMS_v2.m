function rects = pseudoNMS_v2(candi_rects, nms_option)

% overlapping threshold for grouping nearby detections
overlappingThreshold = 0.7; 
overlappingThreshold2 = 0.5; 
embeddingThreshold = 0.7;
large_small_ratio = 0.25;
% candi_rects: N x 5 matrix, each row: [x1 y1 x2 y2 score]
if isempty(candi_rects)
    rects = [];
    return;
end

%=========first: find embeded rectangles
% numCandidates = size(candi_rects, 1);
% predicate = false(numCandidates); % rect j contains rect i if predicate(i,j) = 1
% area_rect = (candi_rects(:,3) - candi_rects(:,1) + 1) .* (candi_rects(:,4) - candi_rects(:,2) + 1);
% for i = 1 : numCandidates
%     for j = i + 1 : numCandidates
%         h = min(candi_rects(i,4), candi_rects(j,4)) - max(candi_rects(i,2), candi_rects(j,2)) + 1;
%         w = min(candi_rects(i,3), candi_rects(j,3)) - max(candi_rects(i,1), candi_rects(j,1)) + 1;
%         if h <= 0 || w <= 0
%            % jump to the next inner for loop
%            continue; 
%         end
%         s = max(h,0) * max(w,0);
%         if s / area_rect(i) >= embeddingThreshold || s / area_rect(j) >= embeddingThreshold
%             predicate(i,j) = true;
%             predicate(j,i) = true;
%         end
%     end
% end
% 
% flag = true(numCandidates,1);
% % merge embeded rectangles
% for i = 1 : numCandidates
%     index = find(predicate(:,i));
%     if isempty(index)
%         continue;
%     end
%     s = max(candi_rects(index,5));
%     if s > candi_rects(i,5)
%         flag(i) = false;
%     end
% end
% % get rid of the low-scoring box in the embedding relationship
% candi_rects = candi_rects(flag, :);


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
        %if s / area_all(i) >= overlappingThreshold && s / area_all(j) >= overlappingThreshold
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
            if s / (area_all(i) + area_all(j) - s) >= overlappingThreshold2
            %if s / area_all(i) >= overlappingThreshold && s / area_all(j) >= overlappingThreshold
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

% % ============== 3rd: find embeded rectangles (spare)
% predicate = false(numCandidates); % rect j contains rect i if predicate(i,j) = 1
% area_rect = (rects(:,3) - rects(:,1) + 1) .* (rects(:,4) - rects(:,2) + 1);
% for i = 1 : numCandidates
%     for j = i + 1 : numCandidates
%         h = min(rects(i,4), rects(j,4)) - max(rects(i,2), rects(j,2)) + 1;
%         w = min(rects(i,3), rects(j,3)) - max(rects(i,1), rects(j,1)) + 1;
%         if h <= 0 || w <= 0
%            continue; 
%         end
%         s = max(h,0) * max(w,0);
% 
%         if s / area_rect(i) >= embeddingThreshold || s / area_rect(j) >= embeddingThreshold
%             % the larger box should less or equal to 3 times of the size of
%             % the smaller one
%             if area_rect(i) / area_rect(j) >= 0.33 && area_rect(j) / area_rect(i) >= 0.33
%                 predicate(i,j) = true;
%                 predicate(j,i) = true;
%             end
%         end
%     end
% end
% 
% flag = true(numCandidates,1);
% 
% % merge embeded rectangles
% for i = 1 : numCandidates
%     index = find(predicate(:,i));
% 
%     if isempty(index)
%         continue;
%     end
% 
%     %s = max([rects(index).score]);
%     s = max(rects(index,5));
%     %if s > rects(i).score
%     if s > rects(i,5)
%         flag(i) = false;
%     end
% end
% rects = rects(flag, :);

% check borders
% [height, width, ~] = size(I);
% numFaces = length(rects);
% 
% for i = 1 : numFaces
%     if rects(i).row < 1
%         rects(i).row = 1;
%     end
%     
%     if rects(i).col < 1
%         rects(i).col = 1;
%     end
%     
%     rects(i).height = rects(i).size;
%     rects(i).width = rects(i).size;
%     
%     if rects(i).row + rects(i).height - 1 > height
%         rects(i).height = height - rects(i).row + 1;
%     end
%     
%     if rects(i).col + rects(i).width - 1 > width
%         rects(i).width = width - rects(i).col + 1;
%     end    
% end
end