clear all; close all; clc;

base_path = '../../data/carpk_pucpr';
data_paths = {'CARPK_devkit/data', 'PUCPR+_devkit/data'};
img_extensions = {'.png', '.jpg'};
img_paths = {'train', 'test', 'val'};
out_path_postfix = {'_half'};

count_img = 0;
for d=1:length(data_paths)
    for i=1:length(img_paths)
        in_path = fullfile(base_path, data_paths{d}, img_paths{i});
        out_path_half = fullfile(base_path, data_paths{d}, [img_paths{i}, out_path_postfix{1}]);
        rm_old_mk_new_dir(out_path_half);
        
        img_list = dir(fullfile(in_path, ['*', img_extensions{d}]));
        assert(~isempty(img_list));
        for j=1:length(img_list)
            im = imread(fullfile(in_path, img_list(j).name));
            imwrite(imresize(im, 0.5), fullfile(out_path_half, img_list(j).name));
            count_img = count_img + 1;
            fprintf('%d\n', count_img);
        end
    end
end
