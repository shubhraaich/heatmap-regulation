clear all; close all; clc;

base_path = '/media/aich/DATA/carpk/datasets';
data_paths = {'CARPK_devkit/data', 'PUCPR+_devkit/data'};
height_gauss = [40, 30]; % based on the mean values got from vis_height_weight.m
img_extensions = {'.png', '.jpg'};
annot_path_in = 'Annotations';
img_paths = {'train'};
annot_file_ext = '.txt';
out_path_postfix = '_gam';
super_path_postfix = '_gam_super';

count_img = 0;
for d=1:length(data_paths)
    for i=1:length(img_paths)
        in_path = fullfile(base_path, data_paths{d}, img_paths{i});
        out_path = fullfile(base_path, data_paths{d}, [img_paths{i}, out_path_postfix]);
        super_path = fullfile(base_path, data_paths{d}, [img_paths{i}, super_path_postfix]);
        rm_old_mk_new_dir(out_path);
        rm_old_mk_new_dir(super_path);
        
        img_list = dir(fullfile(in_path, ['*', img_extensions{d}]));
        assert(~isempty(img_list));
        for j=1:length(img_list)
            % retrieve rectangle sets from annotation file
            annot_file_path = [img_list(j).name(1:end-4), annot_file_ext];
            annot_file_path = fullfile(base_path, data_paths{d}, ...
                                        annot_path_in, annot_file_path);
            rect_set = get_rectangle_list(annot_file_path);
            % add rectangles in image and write
            im = im2double(imread(fullfile(in_path, img_list(j).name)));
            [num_rows, num_cols, ~] = size(im);
            [gam] = gen_gam_image(num_rows, num_cols, rect_set, height_gauss(d));
            gam_img = gray2ind(gam, 256);
            gam_img(isnan(gam_img)) = 0;
            gam_img = ind2rgb(gam_img, jet(256));
            gam_img = im * 0.5 + gam_img * 0.5;
            imwrite(gam, fullfile(out_path, img_list(j).name));
            imwrite(gam_img, fullfile(super_path, img_list(j).name));
            count_img = count_img + 1;
            fprintf('%d\n', count_img);
        end
        
    end
    
end