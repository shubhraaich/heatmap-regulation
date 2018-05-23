function [im] = gen_gam_image(num_rows, num_cols, rect_set, hsize_gauss)

if mod(hsize_gauss,2) == 0
    hsize_gauss = hsize_gauss + 1;
end
sigma = hsize_gauss/(1.96*1.5);
h_gauss = fspecial('gauss', hsize_gauss, sigma);
h_gauss = bsxfun(@rdivide, h_gauss, max(h_gauss(:)));

% rect_set = (N,4) rect matrx (cmin, rmin, cmax, rmax)
center_set = zeros(size(rect_set,1), 2);
im = zeros(num_rows, num_cols);
dhsize = (hsize_gauss-1)/2;

% c_center, r_center
center_set(:,1) = bsxfun(@minus, rect_set(:,3), rect_set(:,1));
center_set(:,1) = round(bsxfun(@rdivide, center_set(:,1), 2));
center_set(:,2) = bsxfun(@minus, rect_set(:,4), rect_set(:,2)); 
center_set(:,2) = round(bsxfun(@rdivide, center_set(:,2), 2));
center_set(:,1) = bsxfun(@plus, center_set(:,1), rect_set(:,1));
center_set(:,2) = bsxfun(@plus, center_set(:,2), rect_set(:,2));

% border adjustment
tmp = center_set(:,1);
tmp(tmp<dhsize+1) = dhsize+1;
tmp(tmp>num_cols-dhsize) = num_cols - dhsize;
center_set(:,1) = tmp;
tmp = center_set(:,2);
tmp(tmp<dhsize+1) = dhsize+1;
tmp(tmp>num_rows-dhsize) = num_rows - dhsize;
center_set(:,2) = tmp;

for i=1:size(center_set,1)
    cmin = center_set(i,1)-dhsize;
    rmin = center_set(i,2)-dhsize;
    cmax = center_set(i,1)+dhsize;
    rmax = center_set(i,2)+dhsize;
%    fprintf('%d, %d, %d, %d\n', cmin, rmin, cmax, rmax);
    im(rmin:rmax, cmin:cmax) = bsxfun(@plus, im(rmin:rmax, cmin:cmax), h_gauss);
end

% clip joint activations in between objects
im(im>1) = 1;
%im = bsxfun(@rdivide, im, max(im(:))); % noramlize GAM

end

