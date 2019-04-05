#### Initial Preprocessing

* split\_train\_test\_data.m
* prep\_train\_data.m
* prep\_test\_data.m
* split\_train\_val\_data.m _(split train set for N32 and N50 experiments)_

#### Save Parameters for Training

* save_mean_std.py (save mean and std for train and val sets)
* copy both mean and std files into the directory containing main\_*.py files

#### Run Simple Baseline

* _(train vgg16 - N32)_ python main\_simple.py --train 1 --img-dir '../../data/train_N32' --ann-dir '../../data/count_train_N32' --arch 'vgg16' --optim 'adam' --workers 4 --batch-size 32 --start-epoch 1 --end-epoch 100 --learning-rate 0.00001 --momentum 0.9 --weight-decay 0.00001 --save-interval 1 --print-freq 20  
* get\_optimal\_epoch.py _(get the optimal epoch on the validation set)_
* _(test vgg16)_ python main\_simple.py --train 0 --img-dir '../../data/test_cells' --ann-dir '../../data/count_test' --arch 'vgg16' --load-epoch XX

* _(train vgg16 - N50)_ python main\_simple.py --train 1 --img-dir '../../data/train_N50' --ann-dir '../../data/count_train_N50' --arch 'vgg16' --optim 'adam' --workers 4 --batch-size 32 --start-epoch 1 --end-epoch 100 --learning-rate 0.00001 --momentum 0.9 --weight-decay 0.00001 --save-interval 1 --print-freq 20  
* get\_optimal\_epoch.py _(change file name appropriately inside)_
* _(test vgg16)_ python main\_simple.py --train 0 --img-dir '../../data/test_cells' --ann-dir '../../data/count_test' --arch 'vgg16' --load-epoch XX


#### Run HR Models

* _(train vgg16 - N32)_ python main\_gap\_gas.py --train 1 --img-dir '../../data/train_N32' --gam-dir '../../data/train_gam_8_N32' --ann-dir '../../data/count_train_N32' --arch 'vgg16' --optim 'adam' --workers 8 --batch-size 32 --start-epoch 1 --end-epoch 100 --learning-rate 0.00001 --momentum 0.9 --weight-decay 0.00001 --save-interval 1 --print-freq 20
* get\_optimal\_epoch.py _(change file name appropriately inside)_
* _(test vgg16)_ python main\_gap\_gas.py --train 0 --img-dir '../../data/test_cells' --ann-dir '../../data/count_test' --arch 'vgg16' --load-epoch XX
* _(train vgg16 - N50)_ python main\_gap\_gas.py --train 1 --img-dir '../../data/train_N50' --gam-dir '../../data/train_gam_8_N50' --ann-dir '../../data/count_train_N50' --arch 'vgg16' --optim 'adam' --workers 8 --batch-size 32 --start-epoch 1 --end-epoch 100 --learning-rate 0.00001 --momentum 0.9 --weight-decay 0.00001 --save-interval 1 --print-freq 20
* get\_optimal\_epoch.py _(change file name appropriately inside)_
* _(test vgg16)_ python main\_gap\_gas.py --train 0 --img-dir '../../data/test_cells' --ann-dir '../../data/count_test' --arch 'vgg16' --load-epoch XX


#### Generate Test Statistics

* print\_test\_stats.py _(generate test statistics for any experiment above)_
