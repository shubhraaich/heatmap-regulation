clear all; close all; clc;

rng(72);

hsize_gauss = 15; % 
val_perc = 0.03; % validation percentage

base_path = '../../data/cc_sjtu';
in_train_rgb_path = 'train_frame';
in_train_gt_path = 'train_label';
out_train_rgb_path = 'train';
out_train_gt_path = 'train_gam';
out_train_vis_path = 'train_gam_vis';
out_val_rgb_path = 'val';
img_ext = '.jpg';

in_train_rgb_path = fullfile(base_path, in_train_rgb_path);
in_train_gt_path = fullfile(base_path, in_train_gt_path);
out_train_rgb_path = fullfile(base_path, out_train_rgb_path);
out_train_gt_path = fullfile(base_path, out_train_gt_path);
out_train_vis_path = fullfile(base_path, out_train_vis_path);
out_val_rgb_path = fullfile(base_path, out_val_rgb_path);

rm_old_mk_new_dir(out_train_rgb_path);
rm_old_mk_new_dir(out_train_gt_path);
rm_old_mk_new_dir(out_train_vis_path);
rm_old_mk_new_dir(out_val_rgb_path);

img_list = dir(fullfile(in_train_rgb_path, ['*', img_ext]));
% generate piecewise random id list for validation images
num_val_img = round(val_perc * length(img_list));
step_size = floor(length(img_list)/num_val_img);
val_id_list = zeros(1, num_val_img);
for i=1:num_val_img
    val_id_list(i) = (i-1)*step_size + randi(step_size);
end
assert(numel(unique(val_id_list))==length(val_id_list));

count_train = 1;
count_val = 1;
count_gt_train = [];
count_gt_val = [];
for i=1:length(img_list)
    tmp_str = strsplit(img_list(i).name, '_');
    if length(tmp_str{1}) ~= 6
        tmp_str = strsplit(img_list(i).name, '-');
    end
    dir_scene = tmp_str{1}; 
    load(fullfile(in_train_gt_path, dir_scene, [img_list(i).name(1:end-4), '.mat']));
    
    if (count_val<=length(val_id_list)) && (val_id_list(count_val)==i) % move to val set
        fprintf('val=%d\n', count_val);
        count_gt_val = [count_gt_val, size(point_position, 1)];
        assert(copyfile(fullfile(in_train_rgb_path, img_list(i).name), ...
                        fullfile(out_val_rgb_path, [num2str(count_val), img_ext]), ...
                        'f'));
        count_val = count_val + 1;
    else % move to train set
        fprintf('train=%d\n', count_train);
        count_gt_train = [count_gt_train, size(point_position, 1)];
        im = im2double(imread(fullfile(in_train_rgb_path, img_list(i).name)));
        [num_rows, num_cols, ~] = size(im);
        gam = gen_gam_image(num_rows, num_cols, point_position, hsize_gauss);
        gam_img = gray2ind(gam, 256);
        gam_img(isnan(gam_img)) = 0;
        gam_img = ind2rgb(gam_img, jet(256));
        gam_img = im * 0.7 + gam_img * 0.3;
        
        imwrite(im, fullfile(out_train_rgb_path, [num2str(count_train), '.png']));
        imwrite(gam, fullfile(out_train_gt_path, [num2str(count_train), '.png']));
        imwrite(gam_img, fullfile(out_train_vis_path, [num2str(count_train), '.png']));
        count_train = count_train + 1;
    end
    
end

count_gt = count_gt_train;
save(fullfile(base_path, 'count_train.mat'), 'count_gt');
count_gt = count_gt_val;
save(fullfile(base_path, 'count_val.mat'), 'count_gt');
