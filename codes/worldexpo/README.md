#### Initial Preprocessing

* prep\_train\_data.m
* save\_resize\_gam\_image.m

#### Save Parameters for Training

* save_mean_std.py (save mean and std for train and val sets)
* copy both mean and std files into the directory containing main\_*.py files

#### Run Simple Baseline

* _(train vgg16)_ python main_simple.py --train 1 --img-dir '../../data/cc_sjtu/train' --ann-dir '../../../data/cc_sjtu/count_train' --arch 'vgg16' --optim 'adam' --workers 16 --batch-size 16 --start-epoch 1 --end-epoch 100 --learning-rate 0.0001 --momentum 0.9 --weight-decay 0.0001 --save-interval 1 --print-freq 50
* get\_optimal\_epoch.py _(get the optimal epoch on the validation set)_
* _(test vgg16)_ python main\_simple.py --train 0 --img-dir '../../data/cc_sjtu/test_frame' --ann-dir '../../data/cc_sjtu/test_label' --arch 'vgg16' --load-epoch XX


#### Run HR Models

* _(train vgg16)_ python main\_gap\_gas.py --train 1 --img-dir '../../data/cc_sjtu/train' --gam-dir '../../data/cc_sjtu/train_gam_8' --ann-dir '../../data/cc_sjtu/count_train' --arch 'vgg16' --optim 'adam' --workers 16 --batch-size 16 --start-epoch 1 --end-epoch 100 --learning-rate 0.0001 --momentum 0.9 --weight-decay 0.0001 --save-interval 1 --print-freq 50
* get\_optimal\_epoch.py _(get the optimal epoch on the validation set)_
* _(test vgg16)_ python main\_gap\_gas.py --train 0 --img-dir '../../data/cc_sjtu/test_frame' --ann-dir '../../data/cc_sjtu/test_label' --arch 'vgg16' --load-epoch XX

#### Generate Test Statistics

* print\_test\_stats.py _(generate test statistics)_
