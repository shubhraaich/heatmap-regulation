clear all; close all; clc;

% =========================================================== %
base_path = '../../data/vgg_cells';
zip_file_name = 'cells.zip';
in_path = 'cells';
train_path = 'train_split';
test_path = 'test_split';
% =========================================================== %

in_path = fullfile(base_path, in_path);
train_path = fullfile(base_path, train_path);
test_path = fullfile(base_path, test_path);


% extract zip file first
rm_old_mk_new_dir(in_path);
cmd_ = ['unzip ', fullfile(base_path, zip_file_name), ' -d ', in_path];
assert(~system(cmd_)); % status=0 for successful

rm_old_mk_new_dir(train_path);
rm_old_mk_new_dir(test_path);

img_list = dir(fullfile(in_path, '*.png'));

for i=1:length(img_list)/2 
    fprintf('%d, %s\n', i, img_list(i).name);
    assert(movefile(fullfile(in_path, img_list(i).name), ...
                    fullfile(train_path, img_list(i).name), 'f'));
end

for i=length(img_list)/2+1:length(img_list) 
    fprintf('%d, %s\n', i, img_list(i).name);
    assert(movefile(fullfile(in_path, img_list(i).name), ...
                    fullfile(test_path, img_list(i).name), 'f'));
end

assert(rmdir(in_path, 's'));
