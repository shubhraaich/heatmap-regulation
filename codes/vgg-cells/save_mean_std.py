""" saves mean and std of R,G,B channels of the dataset """
import os
import numpy as np
import cv2

from open_files import *
from datasets_cells import is_image_file


img_path = '../../data/vgg_cells/train';
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
    mean, std = get_mean_std(img_path);
    print "Mean, Std = {}, {}".format(mean, std);
    save_pickle(mean_file, mean.tolist());
    save_pickle(std_file, std.tolist());


if __name__ == "__main__" :
    main();
