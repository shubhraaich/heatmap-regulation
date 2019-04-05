clear all; close all; clc;

% =========================================================== %
base_path = '../../data/vgg_cells';
in_rgb_path = 'train';
in_gam_path = 'train_gam_8';

N_list = [32, 50];
% =========================================================== %

in_rgb_path = fullfile(base_path, in_rgb_path);
in_gam_path = fullfile(base_path, in_gam_path);

num_files = length(dir(fullfile(in_rgb_path, '*.png'))); % size of train set
load(fullfile(base_path, 'count_train.mat'));
count_all = count_gt;

for i=1:length(N_list)
    N = N_list(i);
    fprintf('N = %d\n', N);
    out_train_rgb_path = [in_rgb_path, '_N', num2str(N)];
    out_train_gam_path = [in_gam_path, '_N', num2str(N)];
    out_val_path = fullfile(base_path, ['val_N', num2str(N)]);
    train_gt_fname = ['count_train_N', num2str(N), '.mat'];
    val_gt_fname = ['count_val_N', num2str(N), '.mat'];
    rm_old_mk_new_dir(out_train_rgb_path);
    rm_old_mk_new_dir(out_train_gam_path);
    rm_old_mk_new_dir(out_val_path);
    
    for j=1:N
        assert(copyfile(fullfile(in_rgb_path, [num2str(j), '.png']), ...
                fullfile(out_train_rgb_path, [num2str(j), '.png']), 'f'));
        assert(copyfile(fullfile(in_gam_path, [num2str(j), '.png']), ...
                fullfile(out_train_gam_path, [num2str(j), '.png']), 'f'));            
    end
    count_val = 0;
    for j=N+1:num_files
        count_val = count_val + 1;
        assert(copyfile(fullfile(in_rgb_path, [num2str(j), '.png']), ...
                fullfile(out_val_path, [num2str(count_val), '.png']), 'f'));
    end    
    
    % save train_N* and val_N* ground truths
    count_gt = count_all(1:N);
    save(fullfile(base_path, train_gt_fname), 'count_gt');
    count_gt = count_all(N+1:num_files);
    save(fullfile(base_path, val_gt_fname), 'count_gt');
end
