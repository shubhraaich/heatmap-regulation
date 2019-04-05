import os
import shutil

from open_files import *

base_path = '../../data/carpk_pucpr';
data_paths = ['CARPK_devkit/data', 'PUCPR+_devkit/data'];
img_path_in = 'Images';
file_path_in = 'ImageSets';
file_names = ['train.txt', 'test.txt'];
img_extensions = ['.png', '.jpg'];

def copy_images(file_name, in_path, out_path, img_ext) :
    if os.path.isdir(out_path) :
        shutil.rmtree(out_path);
    os.mkdir(out_path);

    fhand = get_file_handle(file_name, 'rb');
    count = 0;
    for img_name in fhand :
        img_name = img_name.rstrip();
        img_name = img_name + img_ext;
        assert os.path.isfile(os.path.join(in_path, img_name));
        count += 1;
        print "({}), {}".format(count, img_name);
        shutil.copyfile(os.path.join(in_path, img_name),
                                os.path.join(out_path, img_name) );
    fhand.close();


def main() :
    for i, dpath in enumerate(data_paths) :
        if not os.path.isdir(os.path.join(base_path, dpath)) :
            continue;
        for fname in file_names :
            print "=========== {}, {} =============".format(dpath, fname);
            img_ext = img_extensions[i];
            copy_images(file_name=os.path.join(base_path, dpath, file_path_in, fname),
                    in_path=os.path.join(base_path, dpath, img_path_in),
                    out_path=os.path.join(base_path, dpath, os.path.splitext(fname)[0] ),
                    img_ext=img_ext,
                    );


if __name__ == "__main__" :
    main();
