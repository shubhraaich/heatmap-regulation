-----
Initial Preprocessing
-----
* split\_train\_test\_data.m
* prep\_train\_data.m
* prep\_test\_data.m
* split\_train\_val\_data.m _(split train set for N32 and N50 experiments)_

-----
Run Simple Baseline
-----
* _(train vgg16)_ python main\_simple.py --img-dir 'TRAIN\_IMG\_DIR' --ann-dir 'ANN\_DIR' --arch 'vgg16' --optim 'adam' --workers 4 --batch-size 32 --start-epoch 1 --end-epoch 100 --learning-rate 0.00001 --momentum 0.9 --weight-decay 0.00001 --save-interval 1 --print-freq 20  
* get\_optimal\_epoch.py
* _(test vgg16)_ python main\_simple.py --train 0 --img-dir 'TEST\_IMG\_DIR' --ann-dir 'ANN\_DIR' --arch 'vgg16' --load-epoch XX

-----
Run GAP-GAS
-----
* _(train vgg16)_ python main\_gap\_gas.py --train 1 --img-dir 'TRAIN\_IMG\_DIR' --gam-dir 'GAM\_IMG\_DIR' --ann-dir 'ANN\_DIR' --arch 'vgg16' --optim 'adam' --workers 8 --batch-size 32 --start-epoch 1 --end-epoch 100 --learning-rate 0.00001 --momentum 0.9 --weight-decay 0.00001 --save-interval 1 --print-freq 20
* get\_optimal\_epoch.py
* _(test vgg16)_ python main\_gap\_gas.py --train 0 --img-dir 'TEST\_IMG\_DIR' --ann-dir 'ANN\_DIR' --arch 'vgg16' --load-epoch XX

* print\_test\_stats.py _(generate test statistics)_

