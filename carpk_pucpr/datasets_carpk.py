# modified from source :
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py

import torch.utils.data as data

from PIL import Image, ImageOps
import os
import os.path
from copy import deepcopy

from open_files import *

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    """Checks if a file is an image.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def get_num_lines(path_) :
    fhand = get_file_handle(path_, 'rb');
    count = 0;
    for line in fhand :
        count += 1;
    fhand.close();
    return count;


def make_dataset(img_dir, gam_dir, ann_dir):
    num_files = 1e4; # keep small for validation
    images = []
    img_dir = os.path.expanduser(img_dir);
    for img_name in sorted(os.listdir(img_dir)):
        if not is_image_file(img_name):
            continue;
        img_path = os.path.join(img_dir, img_name);
        gam_path = os.path.join(gam_dir, img_name) if gam_dir is not None else None;
        count = get_num_lines(os.path.join(ann_dir, os.path.splitext(img_name)[0]) + '.txt');
        images.append((img_path, gam_path, count));

    tmp_images = deepcopy(images); # used in list concat
    add_length = int(num_files//len(images)) - 1;
    # copy list for number of additional length
    for i in xrange(add_length) :
        images += tmp_images;
    del tmp_images;

    return images;


def make_dataset_test(img_dir, gam_dir, ann_dir):
    images = []
    img_dir = os.path.expanduser(img_dir);
    for img_name in sorted(os.listdir(img_dir)):
        if not is_image_file(img_name):
            continue;
        img_path = os.path.join(img_dir, img_name);
        gam_path = os.path.join(gam_dir, img_name) if gam_dir is not None else None;
        count = get_num_lines(os.path.join(ann_dir, os.path.splitext(img_name)[0]) + '.txt');
        images.append((img_path, gam_path, count));

    return images;


def pil_loader(path, is_gam):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        if is_gam :
            return img.convert('L')
        return img.convert('RGB');


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path, is_gam=False):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path, is_gam)
    else:
        return pil_loader(path, is_gam)


class ImageFolder(data.Dataset):
    """Read RGB image, GAM, and Count from the following directory strcuture :
        img_dir/filename.png
        gam_dir/filename.png
        ann_dir/filename.txt
    Args:
        img_dir (string): RGB image directory path.
        gam_dir (string): GAM imgae directory path.
        ann_dir (string): Annotation directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image, gam, count) tuples
    """

    def __init__(self, img_dir, ann_dir, gam_dir=None, transform_joint=None, transform=None, gam_transform=None,
                 loader=default_loader):
        imgs = make_dataset(img_dir, gam_dir, ann_dir);
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + img_dir + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.img_dir = img_dir
        self.gam_dir = gam_dir
        self.ann_dir = ann_dir

        self.imgs = imgs
        self.transform_joint = transform_joint
        self.transform = transform
        self.gam_transform = gam_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path_img, path_gam, count = self.imgs[index]
        img = self.loader(path_img, is_gam=False)
        gam = self.loader(path_gam, is_gam=True);
        if self.transform_joint is not None:
            img, gam = self.transform_joint(img, gam)
        if self.transform is not None:
            img = self.transform(img)
        if self.gam_transform is not None:
            gam = self.gam_transform(gam)

        return img, count, gam;

    def __len__(self):
        return len(self.imgs)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Data Location: {}\n'.format(self.data_dir)
        fmt_str += '    GT Location: {}\n'.format(self.gt_dir)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class ImageFolder_Simple(data.Dataset):
    """Read RGB image, GAM, and Count from the following directory strcuture :
        img_dir/filename.png
        ann_dir/filename.txt
    Args:
        img_dir (string): RGB image directory path.
        ann_dir (string): Annotation directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        loader (callable, optional): A function to load an image given its path.
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image, gam, count) tuples
    """

    def __init__(self, img_dir, ann_dir, gam_dir=None, transform_joint=None, transform=None, gam_transform=None,
                 loader=default_loader):
        imgs = make_dataset(img_dir, gam_dir, ann_dir);
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + img_dir + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.img_dir = img_dir
        self.ann_dir = ann_dir

        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path_img, _, count = self.imgs[index]
        img = self.loader(path_img, is_gam=False)
        if self.transform is not None:
            img = self.transform(img)

        return img, count;

    def __len__(self):
        return len(self.imgs)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Data Location: {}\n'.format(self.data_dir)
        fmt_str += '    GT Location: {}\n'.format(self.gt_dir)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
