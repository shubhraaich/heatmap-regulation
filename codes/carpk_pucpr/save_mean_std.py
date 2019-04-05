""" saves mean and std of R,G,B channels of the dataset """
import os
import numpy as np
import cv2

from open_files import *
from datasets_carpk import is_image_file


base_path = '../../data/carpk_pucpr';
data_paths = ['CARPK_devkit/data', 'PUCPR+_devkit/data'];
img_path = 'train';
mean_file = 'mean';
std_file = 'std';
val_max_img = 255.0;

def get_mean_std(img_path) :
    count = 0;
    mean = np.array([0.] * 3);
    std = np.array([0.] * 3);
    for fname in sorted(os.listdir(img_path)) :
        if not is_image_file(fname) :
            continue;
        count += 1;
        print "image = {}".format(count);
        im = cv2.imread(os.path.join(img_path, fname), cv2.IMREAD_UNCHANGED);
        mean += im.mean(axis=(0,1))/val_max_img;
        std += im.std(axis=(0,1))/val_max_img;

    mean /= count;
    std /= count;

    # swap R and B channel values
    tmp = mean[0];
    mean[0] = mean[2];
    mean[2] = tmp;
    tmp = std[0];
    std[0] = std[2];
    std[2] = tmp;

    return mean, std;


def main() :
    for dpath in data_paths :
        tmp_img_path = os.path.join(base_path, dpath, img_path);
        if not os.path.isdir(tmp_img_path) :
            continue;
        mean, std = get_mean_std(tmp_img_path);
        print "Mean({}) = {}".format(dpath, mean);
        print "Std({}) = {}".format(dpath, std);
        save_pickle(os.path.join(base_path, dpath, mean_file), mean.tolist());
        save_pickle(os.path.join(base_path, dpath, std_file), std.tolist());


if __name__ == "__main__" :
    main();
