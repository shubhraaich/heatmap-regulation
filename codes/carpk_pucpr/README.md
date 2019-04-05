#### Initial Preprocessing

* split\_train\_test.py _(split train test data)_
* split\_val\_from\_train.py _(split train val data)_

#### Save Parameters for Training

* save\_resized\_db.m _(get resized database into half and quarter)_
* save\_mean\_std.py _(save mean and std for both datasets)_
* copy both mean and std files into the directory containing main\_*.py files

#### Run Simple Baseline

* _(train vgg16)_ python main\_simple.py --train 1 --img-dir '../../data/carpk_pucpr/CARPK_devkit/train_half' --ann-dir '../../data/carpk_pucpr/CARPK_devkit/Annotations' --arch 'vgg16' --optim 'adam' --workers 8 --batch-size 32 --start-epoch 1 --end-epoch 100 --learning-rate 0.0001 --momentum 0.9 --weight-decay 0.0001 --save-interval 1 --print-freq 50 
* get\_optimal\_epoch.py _(get the optimal epoch on the validation set)_
* _(test vgg16)_ python main\_simple.py --train 0 --img-dir '../../data/carpk_pucpr/CARPK_devkit/test_half' --ann-dir '../../data/carpk_pucpr/CARPK_devkit/Annotations' --arch 'vgg16' --load-epoch XX

#### GAM Generation

* save\_gam\_image.m
* save\_resize\_gam\_image.m

#### Run HR Models

* _(train vgg16)_ python main\_gap\_gas.py --train 1 --img-dir '../../data/carpk_pucpr/CARPK_devkit/train_half' --gam-dir '../../data/carpk_pucpr/CARPK_devkit/train_gam_16' --ann-dir '../../data/carpk_pucpr/CARPK_devkit/Annotations' --arch 'vgg16' --optim 'adam' --workers 8 --batch-size 32 --start-epoch 1 --end-epoch 100 --learning-rate 0.0001 --momentum 0.9 --weight-decay 0.0001 --save-interval 1 --print-freq 50 
* get\_optimal\_epoch.py _(get the optimal epoch on the validation set)_
* _(test vgg16)_ python main\_gap\_gas.py --train 0 --img-dir '../../data/carpk_pucpr/CARPK_devkit/test_half' --ann-dir '../../data/carpk_pucpr/CARPK_devkit/Annotations' --arch 'vgg16' --load-epoch XX

#### Generate Test Statistics

* print\_test\_stats.py _(generate test statistics)_

#### PUCPR+ Experiments

* Just replace '\*carpk\*' with '\*pucpr\*' in the commands above. Also, do the same inside print\_test\_stats.py file.
