clear all; close all; clc;

% =========================================================== %
hsize_gauss = 1; % unit ratio for gaussian window size 

base_path = '/media/aich/DATA/vgg_cells';
in_path = 'train_split';
in_rgb_postfix = 'cell.png';
in_dot_postfix = 'dots.png';

out_rgb_path = 'train';
out_gam_path = 'train_gam';
out_gam_path_8 = 'train_gam_8';
out_super_path = 'train_gam_super';
out_collage = 'train_collage'; % needed for annotation error check
% =========================================================== %

in_path = fullfile(base_path, in_path);
out_rgb_path = fullfile(base_path, out_rgb_path);
out_gam_path = fullfile(base_path, out_gam_path);
out_gam_path_8 = fullfile(base_path, out_gam_path_8);
out_super_path = fullfile(base_path, out_super_path);
out_collage = fullfile(base_path, out_collage);
rm_old_mk_new_dir(out_rgb_path);
rm_old_mk_new_dir(out_gam_path);
rm_old_mk_new_dir(out_gam_path_8);
rm_old_mk_new_dir(out_super_path);
rm_old_mk_new_dir(out_collage);

count_total = 0;
count_gt = [];
img_list = dir(fullfile(in_path, ['*', in_rgb_postfix]));
for i=1:length(img_list) 
    count_total = count_total + 1;
    img_dot_name = [img_list(i).name(1:end-8), in_dot_postfix];
    im = im2double(imread(fullfile(in_path, img_list(i).name)));
    im_dot = imread(fullfile(in_path, img_dot_name));
    im_dot = single(im_dot(:,:,1)>=200 & im_dot(:,:,2)<=50 & im_dot(:,:,3)<=50);
    [num_rows, num_cols] = size(im_dot);
    center_set = get_center_coords(im_dot);
    gam = gen_gam_image(num_rows, num_cols, center_set, hsize_gauss);
    gam_img = gray2ind(gam, 256);
    gam_img(isnan(gam_img)) = 0;
    gam_img = ind2rgb(gam_img, jet(256));
    gam_img = im * 0.7 + gam_img * 0.3;

    count_gt = [count_gt, size(center_set,1)];

    fprintf('%d, %s\n', count_total, img_list(i).name);
    out_file_name = [num2str(count_total), '.png'];
    imwrite(im, fullfile(out_rgb_path, out_file_name));
    imwrite(gam, fullfile(out_gam_path, out_file_name));
    % no normalization
    gam_8 = imresize(gam, [num_rows/8, num_cols/8]);
    gam_8 = bsxfun(@rdivide, gam_8, max(gam_8(:)));
    imwrite(gam_8, fullfile(out_gam_path_8, out_file_name));
    imwrite(gam_img, fullfile(out_super_path, out_file_name));
    imwrite([im, gam_img], fullfile(out_collage, out_file_name));
end

save(fullfile(base_path, 'count_train.mat'), 'count_gt');