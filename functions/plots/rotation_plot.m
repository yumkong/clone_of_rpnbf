function rotation_plot(recall_vec, start_thresh, thresh_interval, thresh_end)
assert(length(recall_vec)==31, 'Recall vector must contain 31 elms');
% Generate some test data.  Assume that the X-axis represents months.
x = 1:31;
y = recall_vec;
% Plot the data.
h = plot(x,y, 'b^-', 'linewidth', 2);

% Reduce the size of the axis so that all the labels fit in the figure.
pos = get(gca,'Position');
set(gca,'Position',[pos(1), .2, pos(3) .65]);
% Add a title.
%title('This is a title')
% Set the X-Tick locations so that every face size is labeled.
Xt = 1:31;
Xl = [1 31];
set(gca, 'FontSize', 13);
set(gca, 'XTick', Xt, 'XLim', Xl);
% Add the face sizes as tick labels.
face_sizes = {};
for k = start_thresh:thresh_interval:thresh_end
    size_tmp = sprintf('%d-%d', k, k + thresh_interval);
    face_sizes = cat(1, face_sizes, {size_tmp});
end
% add the final range: '295-'
size_tmp = sprintf('%d-', k + thresh_interval);
face_sizes = cat(1, face_sizes, {size_tmp});
    
ax = axis;    % Current axis limits
axis(axis);    % Set the axis limit modes (e.g. XLimMode) to manual
Yl = ax(3:4);  % Y-axis limits
% Place the text labels
t = text(Xt,Yl(1)*ones(1,length(Xt)),face_sizes(1:1:31));
set(t,'HorizontalAlignment','right','VerticalAlignment','top', ...
      'Rotation',45, 'FontSize', 12, 'FontWeight', 'bold');
% Remove the default labels
set(gca,'XTickLabel','')
% Get the Extent of each text object.  This
% loop is unavoidable.
for i = 1:length(t)
  ext(i,:) = get(t(i),'Extent');
end
% Determine the lowest point.  The X-label will be
% placed so that the top is aligned with this point.
LowYPoint = min(ext(:,2));
% Place the axis label at this point
XMidPoint = Xl(1)+abs(diff(Xl))/2;
tl = text(XMidPoint,LowYPoint,'Face Sizes(pixels)', ...
          'VerticalAlignment','top', ...
          'HorizontalAlignment','center');
set(tl, 'FontSize', 15, 'FontWeight', 'bold');
ylabel('Recall Rate', 'FontSize', 15, 'FontWeight', 'bold');
end