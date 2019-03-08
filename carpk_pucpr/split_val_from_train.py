import os
import shutil
import random

base_path = '/media/aich/DATA/carpk/datasets';
data_paths = ['CARPK_devkit/data', 'PUCPR+_devkit/data'];
in_path = 'train';
out_path = 'val';
img_extensions = ['.png', '.jpg'];
move_perc_list = [0.1, 0.1]; # 10%, 10% for validation image


def move_val_images(in_path, out_path, img_ext, move_perc) :
    random.seed(72);
    if os.path.isdir(out_path) :
        shutil.rmtree(out_path);
    os.mkdir(out_path);

    img_list = sorted(os.listdir(in_path));
    num_val_files = int(round(len(img_list) * move_perc));
    val_id_list = random.sample(range(len(img_list)), num_val_files);

    for i, val_id in enumerate(val_id_list) :
        img_name = img_list[val_id];
        assert os.path.isfile(os.path.join(in_path, img_name));
        print "({}), {}".format(i+1, img_name);
        shutil.move(os.path.join(in_path, img_name),
                                os.path.join(out_path, img_name) );


def main() :
    for i, dpath in enumerate(data_paths) :
        tmp_in_path = os.path.join(base_path, dpath, in_path);
        tmp_out_path = os.path.join(base_path, dpath, out_path);
        if not os.path.isdir(tmp_in_path) :
            continue;
        print "=========== {} =============".format(tmp_in_path);
        img_ext = img_extensions[i];
        move_perc = move_perc_list[i];
        move_val_images(in_path=tmp_in_path,
                        out_path=tmp_out_path,
                        img_ext = img_ext,
                        move_perc=move_perc,
                        );


if __name__ == "__main__" :
    main();
