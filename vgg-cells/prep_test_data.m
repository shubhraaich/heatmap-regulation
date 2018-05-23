clear all; close all; clc;

% =========================================================== %
hsize_gauss = 5; % unit ratio for gaussian window size 

base_path = '/media/aich/DATA/vgg_cells';
in_path = 'test_split';
in_rgb_postfix = 'cell.png';
in_dot_postfix = 'dots.png';

out_rgb_path = 'test';
% =========================================================== %

in_path = fullfile(base_path, in_path);
out_rgb_path = fullfile(base_path, out_rgb_path);
rm_old_mk_new_dir(out_rgb_path);

count_gt = [];
img_list = dir(fullfile(in_path, ['*', in_rgb_postfix]));
for i=1:length(img_list) 
    img_dot_name = [img_list(i).name(1:end-8), in_dot_postfix];
    fprintf('%d, %s\n', i, img_list(i).name);

    im_dot = imread(fullfile(in_path, img_dot_name));
    im_dot = single(im_dot(:,:,1)>=200 & im_dot(:,:,2)<=50 & im_dot(:,:,3)<=50);
    center_set = get_center_coords(im_dot);
    count_gt = [count_gt, size(center_set,1)];

    out_file_name = [num2str(i), '.png'];
    assert(copyfile(fullfile(in_path, img_list(i).name), ...
                    fullfile(out_rgb_path, out_file_name), 'f'));
end

save(fullfile(base_path, 'count_test.mat'), 'count_gt');