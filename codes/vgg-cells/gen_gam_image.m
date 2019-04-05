function [im] = gen_gam_image(num_rows, num_cols, center_set, hsize_gauss)

if mod(hsize_gauss,2) == 0
    hsize_gauss = hsize_gauss + 1;
end

sigma = hsize_gauss/(1.96*1.5);
h_gauss = fspecial('gauss', hsize_gauss, sigma);
h_gauss = bsxfun(@rdivide, h_gauss, max(h_gauss(:)));

% center_set = (N,2) center matrix (c_center, r_center)
im = zeros(num_rows, num_cols);
dhsize = (hsize_gauss-1)/2;

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

im = bsxfun(@rdivide, im, max(im(:))); % noramlize GAM

end

