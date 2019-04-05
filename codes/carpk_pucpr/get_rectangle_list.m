function [rect] = get_rectangle_list(file_path)
%%%%%%%%%%%%%%%%%%%%%%%
% rect = (N,4) matrix, each row contains (cmin,rmin,cmax,rmax) with 1-based
% indexing
%%%%%%%%%%%%%%%%%%%%%%%

%file_path = '/media/aich/DATA/carpk/datasets/PUCPR+_devkit/data/Annotations/0_Cloudy.txt';

fid = fopen(file_path, 'r');
assert(fid~=-1, 'Problem in opening file\n %s', file_path);
rect = fscanf(fid, '%u');
fclose(fid);

rect = reshape(rect, 5, [])'; % because reshape works column-wise
rect(:,5) = [];
%rect(:,3) = bsxfun(@minus, rect(:,3), rect(:,1)); % w = cmax - cmin
%rect(:,4) = bsxfun(@minus, rect(:,4), rect(:,2)); % h = rmax - rmin
rect = bsxfun(@plus, rect, 1); % 1-based indexing

end