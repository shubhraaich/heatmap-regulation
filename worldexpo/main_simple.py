# modified from sources:
# (1) https://github.com/pytorch/examples/blob/
# 42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py
import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np

#import torchvision.transforms as transforms
import transforms as transforms
import datasets_ccsjtu

from open_files import *

cudnn.benchmark = False;

parser = argparse.ArgumentParser(description='PyTorch lcc Training')
parser.add_argument('--train', default=1, type=int, metavar='N',
                    help='train(1) or test(0)');
parser.add_argument('--img-dir', type=str, metavar='DIR',
                    help='path to RGB images');
parser.add_argument('--gam-dir', type=str, default='', metavar='DIR',
                    help='path to GAM images');
parser.add_argument('--ann-dir', type=str, metavar='DIR',
                    help='path to annotation files');
parser.add_argument('--arch', '-a', type=str, metavar='ARCH', default='',
                    help='model architecture to be used');
parser.add_argument('--optim', type=str, metavar='OPTIMIZER', default='adam',
                    help='optimization algorithm for training');
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)');
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)');
parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                    help='starting epoch number (useful on restarts)');
parser.add_argument('--end-epoch', default=100, type=int, metavar='N',
                    help='end epoch number');
parser.add_argument('--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)');
parser.add_argument('--save-interval', default=1, type=int,
                    metavar='N', help='epoch interval to save the model');
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--load-epoch', default=50, type=int,
                    help='epoch to be loaded for test');


def main():
    torch.manual_seed(72)
    torch.cuda.manual_seed_all(72)
    global args
    args = parser.parse_args()

    # create model
    if args.arch == 'vgg16' :
        if args.train==1 :
            model = models.vgg16(pretrained=True);
        else :
            model = models.vgg16();
        tmp = list(model.features.children());
        for i in xrange(8) :
            tmp.pop();
        # ==== replace relu with prelu ===== #
#        id_relu = [1,3,6,8,11,13,15,18,20,22];
#        for i in id_relu :
#            tmp[i] = nn.PReLU(tmp[i-1].out_channels);
        # =========================================== #
        tmp.append(nn.AvgPool2d(kernel_size=(72,90), stride=(72,90))); # 45,80
        model.features = nn.Sequential(*tmp);
        model.classifier = nn.Linear(in_features=512, out_features=1);
        model = nn.DataParallel(model);

    if args.train == 1 :
        # open log fileg
        log_dir = 'logs_simple';
        log_name = args.arch + '_new.csv';
        if not os.path.isdir(log_dir) :
            os.mkdir(log_dir);
        log_handle = get_file_handle(os.path.join(log_dir, log_name), 'wb+');
        log_handle.write('Epoch, LearningRate, Momentum, WeightDecay,' + \
                        'Loss, TotalCount, Difference, Overestimate, Underestimate, RelativeDifference\n');
        log_handle.close();

    # check model directory
    model_dir = 'models_simple';
    if not os.path.isdir(model_dir) :
        os.mkdir(model_dir);

    # resume learning based on cmdline arguments
    if ((args.start_epoch > 1) and (args.train==1)) :
        load_epoch = args.start_epoch - 1;
    elif (args.train==0) and (args.load_epoch>0) :
        load_epoch = args.load_epoch;
    else :
        load_epoch = 0;

    if load_epoch > 0 :
        print("=> loading checkpoint for epoch = '{}'"
                        .format(load_epoch));
        checkpoint_name = args.arch + '_ep_' + str(load_epoch) + '.pth.tar';
        checkpoint = torch.load(os.path.join(model_dir, checkpoint_name));
        model.load_state_dict(checkpoint['state_dict']);

    model.cuda(); # transfer to cuda

    # get dataset mean and std
    mean = load_pickle('mean');
    std = load_pickle('std');
    # ImageNet mean and std
#    mean=[0.485, 0.456, 0.406];
#    std=[0.229, 0.224, 0.225];

    if args.train == 1 :
        criterion = nn.SmoothL1Loss().cuda();

        if args.optim == 'adam' :
            optimizer = torch.optim.Adam(model.parameters(),
                                    lr=args.learning_rate,
                                    weight_decay=args.weight_decay);
        elif args.optim == 'sgd' :
            optimizer = torch.optim.SGD(model.parameters(),
                        lr=args.learning_rate,
                        momentum=args.momentum,
                        weight_decay=args.weight_decay);

        img_dir, gam_dir, ann_dir = args.img_dir, args.gam_dir, args.ann_dir;
        if gam_dir == '' :
            gam_dir = None;

        # Data loading code
        train_loader = torch.utils.data.DataLoader(
            datasets_ccsjtu.ImageFolder_Simple(img_dir=img_dir, ann_dir=ann_dir,
                transform=transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                ]),
            ),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True);

        val_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                ]);

        for epoch in range(args.start_epoch, args.end_epoch+1):
            # train for one epoch
            stats_epoch = train(train_loader, model, criterion, optimizer, epoch);
            validate(img_dir, ann_dir, model, val_transform, epoch);

            model_name = args.arch + '_ep_' + str(epoch) + '.pth.tar';
            # get current parameters of optimizer
            for param_group in optimizer.param_groups :
                cur_lr = param_group['lr'];
                cur_wd = param_group['weight_decay'];
                if param_group.has_key('momentum') :
                    cur_momentum = param_group['momentum'];
                else :
                    cur_momentum = 'n/a';
                break; # constant parameters throughout the network

            if epoch % args.save_interval == 0 :
                state = {
                    'epoch': epoch,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'learning_rate': cur_lr,
                    'moemntum': cur_momentum,
                    'weight_decay': cur_wd
                };

                torch.save(state, os.path.join(model_dir, model_name));

            diff_new = stats_epoch['totalest'];
            if epoch > 1:
                diff_old = load_pickle('diff_old');
                diff_change = ((diff_old - diff_new)*100.0)/diff_old;
            else :
                diff_change = 0.0;
            save_pickle('diff_old', diff_new);

            # write logs using logHandle
            log_handle = get_file_handle(os.path.join(log_dir, log_name), 'ab');
            log_handle.write(str(epoch) + ',' +
                            str(cur_lr) + ',' +
                            str(cur_momentum) + ',' +
                            str(cur_wd) + ',' +
                            str(stats_epoch['loss']) + ',' +
                            str(stats_epoch['total_count']) + ',' +
                            str(stats_epoch['totalest']) + ',' +
                            str(stats_epoch['overest']) + ',' +
                            str(stats_epoch['underest']) + ',' +
                            str(diff_change) + '\n');

            log_handle.close();

            # ============== manually downgrade learning rate ================ #
            if epoch == 10 : # from 11
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.1;

#            adjust_learning_rate(optimizer, epoch, 30); # adjust learning rate

    elif args.train == 0 : # test
        img_dir, gam_dir, ann_dir = args.img_dir, args.gam_dir, args.ann_dir;
        test_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                ]);
        test(test_transform, model, args.arch, load_epoch, img_dir, ann_dir);

# ----------------------------------------------------------------------- #

def train(train_loader, model, criterion, optimizer, epoch):
    global args
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    underest_epoch, overest_epoch, totalest_epoch, total_count = 0.0, 0.0, 0.0, 0.0;

    # switch to train mode
    model.train()

    time_start = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - time_start);

        target = target.float().cuda(async=True);
        input_var = torch.autograd.Variable(input).cuda();
        target_var = torch.autograd.Variable(target);

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        overest, underest, totalest, batch_count = accuracy(output.data, target)
        losses.update(loss.data[0], input.size(0))

        underest_epoch += underest;
        overest_epoch += overest;
        totalest_epoch += totalest;
        total_count += batch_count;

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - time_start);
        time_start = time.time();

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Overestimate {overest:.3f} \t'
                  'Underestimate {underest:.3f} \t'
                  'TotalDifference {totalest:.3f} \t'
                  'Count {batch_count:.3f}'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, overest=overest,
                   underest=underest, totalest=totalest, batch_count=batch_count));

    return {'loss':losses.avg, 'overest':overest_epoch, 'underest':underest_epoch,
                    'totalest':totalest_epoch, 'total_count':total_count};


def validate(img_dir, ann_dir, model, val_transform, epoch) :
    img_dir = os.path.join(os.path.dirname(img_dir), "val");
    ann_dir = os.path.join(os.path.dirname(ann_dir), "count_val" + ".mat");
    val_dif_fname = 'val_diff_old_simple';

    images = datasets_ccsjtu.make_dataset(img_dir=img_dir,
                                         gam_dir=None,
                                         ann_dir=ann_dir, is_copy=False);

    model.eval();

    diff_abs = 0.;
    for i, (img_path, gam_path, target) in enumerate(images):
        # measure data loading time
        input = val_transform(datasets_ccsjtu.default_loader(img_path)).unsqueeze(0);
        input_var = torch.autograd.Variable(input, volatile=True).cuda();

        # compute output
        output = model(input_var);
        pred = int(round(output.data.cpu()[0,0]));
        diff_abs += abs(pred-target);

    if epoch == 1:
        diff_old = [];
    else :
        diff_old = load_pickle(val_dif_fname);
    diff_old.append(float(diff_abs));
    save_pickle(val_dif_fname, diff_old);
    print(diff_old[-1]);


def test(test_transform, model, arch_type, load_epoch, img_dir, ann_dir) :
    global args
    batch_time = AverageMeter()
    data_time = AverageMeter()

    # switch to train mode
    model.eval();
    model.cuda();

    out_cam_dir = os.path.join(os.path.dirname(img_dir), 'test_cam_simple');
    out_res_dir = os.path.join(os.path.dirname(img_dir), 'results_simple');
    rm_old_mk_new_dir(out_cam_dir);
    rm_old_mk_new_dir(out_res_dir);

    images_all = datasets_ccsjtu.make_dataset_test(img_dir, ann_dir);
    time_start = time.time()

    cam_weights = model.module.classifier.weight.data.cpu().clone();
    cam_weights.unsqueeze_(2).unsqueeze_(3);
    from copy import deepcopy;
    import scipy.io;
    import cv2;
    model_cam_part = deepcopy(model);
    tmp = list(model_cam_part.module.features.children());
    tmp.pop();
    model_cam_part.module = nn.Sequential(*tmp);

    for sub_dir in images_all :
        images = images_all[sub_dir];
        out_fhand = get_file_handle(os.path.join(out_res_dir, sub_dir + ".csv"), 'wb+');
        for i, (img_path, target) in enumerate(images):
            # measure data loading time
            data_time.update(time.time() - time_start);

            input = test_transform(datasets_ccsjtu.default_loader(img_path)).unsqueeze(0);
            input_var = torch.autograd.Variable(input, volatile=True).cuda();

            # compute output
            output = model(input_var);
            output_cam = model_cam_part(input_var).data.cpu();
            output_cam.mul_(cam_weights.expand_as(output_cam));
            output_cam = output_cam.sum(1).squeeze_();
            pred = int(round(output.data.cpu()[0,0]));

            out_fhand.write(os.path.basename(img_path) + ',' + str(target) + ',' + str(pred) + '\n');
            # convert to cam image using colormap conversion
            output_cam = output_cam.numpy();
            output_cam = cv2.resize(output_cam, None, fx=8, fy=8, interpolation=cv2.INTER_CUBIC);
            output_cam = (output_cam-output_cam.min())/(output_cam.max()-output_cam.min());
            scipy.io.savemat(os.path.join(out_cam_dir,
                                os.path.splitext(os.path.basename(img_path))[0] + '.mat'),
                                {'data':output_cam}
                            );
            # measure elapsed time
            batch_time.update(time.time() - time_start);
            time_start = time.time();

            print('Batch: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                   i+1, len(images), batch_time=batch_time, data_time=data_time));

        out_fhand.close();



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, epoch_interval):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

#    lr = args.learning_rate * (0.1 ** (epoch // epoch_interval))
    if epoch % epoch_interval == 0 :
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1;


def accuracy(output, target):
    """ returns overestimate, underestimate, and total estimated difference """
    diff_ = output - target.unsqueeze(1);
    overest = diff_[diff_.ge(0)].sum();
    underest = abs(diff_[diff_.lt(0)].sum());
    return overest, underest, (overest + underest), target.sum();


if __name__ == '__main__':
    main();
