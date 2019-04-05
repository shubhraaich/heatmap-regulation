% save 1/2, 1/4, 1/8, 1/16th images
clear all; close all; clc;

base_path = '../../data/carpk_pucpr';
data_paths = {'CARPK_devkit/data', 'PUCPR+_devkit/data'};
img_extensions = {'.png', '.jpg'};
gam_paths = {'train_gam'};
super_paths = {'train_gam_super'};


count_img = 0;
for d=1:length(data_paths)
    for i=1:length(gam_paths)
        in_gam_path = fullfile(base_path, data_paths{d}, gam_paths{i});
        in_super_path = fullfile(base_path, data_paths{d}, super_paths{i});
        out_path_2 = fullfile(base_path, data_paths{d}, [gam_paths{i}, '_2']);
        super_path_2 = fullfile(base_path, data_paths{d}, [super_paths{i}, '_2']);
        out_path_16 = fullfile(base_path, data_paths{d}, [gam_paths{i}, '_16']);
        super_path_16 = fullfile(base_path, data_paths{d}, [super_paths{i}, '_16']);        
        
        rm_old_mk_new_dir(out_path_2);
        rm_old_mk_new_dir(out_path_16);
        rm_old_mk_new_dir(super_path_2);
        rm_old_mk_new_dir(super_path_16);
        
        img_list = dir(fullfile(in_gam_path, ['*', img_extensions{d}]));
        assert(~isempty(img_list));
        for j=1:length(img_list)
            im_gam = im2double(imread(fullfile(in_gam_path, img_list(j).name)));
            im_gam_super = im2double(imread(fullfile(in_super_path, img_list(j).name)));

            im_gam_2 = imresize(im_gam, [360,640]);
            im_gam_super_2 = imresize(im_gam, [360,640]);
            % normalization
%            im_gam_2 = bsxfun(@rdivide, im_gam_2, max(im_gam_2(:)));
%            im_gam_super_2 = bsxfun(@rdivide, im_gam_super_2, max(im_gam_super_2(:)));
            im_gam_16 = imresize(im_gam, [45,80]);
            im_gam_super_16 = imresize(im_gam, [45,80]);
            % normalization
%            im_gam_16 = bsxfun(@rdivide, im_gam_16, max(im_gam_16(:)));
%            im_gam_super_16 = bsxfun(@rdivide, im_gam_super_16, max(im_gam_super_16(:)));            
            
            imwrite(im_gam_2, fullfile(out_path_2, img_list(j).name));
            imwrite(im_gam_super_2, fullfile(super_path_2, img_list(j).name));
            imwrite(im_gam_16, fullfile(out_path_16, img_list(j).name));
            imwrite(im_gam_super_16, fullfile(super_path_16, img_list(j).name));            
            
            count_img = count_img + 1;
            fprintf('%d\n', count_img);
        end
        
    end
    
end
