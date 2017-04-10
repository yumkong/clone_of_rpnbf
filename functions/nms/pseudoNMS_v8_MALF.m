function rects = pseudoNMS_v8_MALF(candi_rects)

embeddingThreshold = 0.5; %0.8
% candi_rects: N x 5 matrix, each row: [x1 y1 x2 y2 score]

%1019 added: first remove negative scoring rects
if ~isempty(candi_rects)
    candi_rects = candi_rects(candi_rects(:,5) > 0, :);
else
    rects = [];
    return;
end

nms_debug = false;

% ============== 3rd: find embeded rectangles (main try)
if 1
    
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
                predicate(i,j) = true;
                predicate(j,i) = true;
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
            rects = cat(1, rects, candi_rects(index, :));
        else
            % index -- cluster index
            weight = candi_rects(index, 5);           
            if any(weight >= 0.999)    
                idx2 = (weight >= 0.999);
                rects = cat(1, rects, candi_rects(index(idx2), :));
            else
                [~,idx2] = max(weight);
                rects = cat(1, rects, candi_rects(index(idx2), :));
            end
        end
    end

end

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