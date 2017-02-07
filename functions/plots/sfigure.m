function h = sfigure(h,s1,s2)
% SFIGURE  Create figure window (minus annoying focus-theft).
%
% Usage is identical to figure.
%
% Daniel Eaton, 2005
%
% See also figure
%
% Modified by Peter Karasev, 2012, to optionally set scale
%

if nargin>=1 
    if ishandle(h)
        set(0, 'CurrentFigure', h);
    else
        h = figure(h);
    end
else
    h = figure;
end

if( nargin > 1 )
  scaleX = s1;
  scaleY = s1;
  if( nargin > 2 )
    scaleY = s2;
  end
  pos = get(h,'Position');
  pos(3:4) = [400 300].*[scaleX scaleY];
  set(gcf,'Position',pos);
end