clear all; close all; clc;

% =========================================================== %
FACT_REDUC = 8; % 8-times reduction
img_ext_gam = '.png';

base_path = '/media/aich/DATA/cc_sjtu';
in_gam_path = 'train_gam';

out_gam_path = [in_gam_path, '_', num2str(FACT_REDUC)];
% =========================================================== %

in_gam_path = fullfile(base_path, in_gam_path);
out_gam_path = fullfile(base_path, out_gam_path);

rm_old_mk_new_dir(out_gam_path);

% start processing train set
img_list = dir(fullfile(in_gam_path, ['*', img_ext_gam]));
assert(~isempty(img_list));

for i=1:length(img_list)
    fprintf('gam=%d\n', i);
    im_gam = im2double(imread(fullfile(in_gam_path, img_list(i).name)));
    [num_rows, num_cols] = size(im_gam);
    im_gam = imresize(im_gam, [num_rows/FACT_REDUC, num_cols/FACT_REDUC]);
    imwrite(im_gam, fullfile(out_gam_path, img_list(i).name));
end