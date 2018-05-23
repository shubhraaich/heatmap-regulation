-----
Run Simple Baseline
-----
* _(train vgg16)_ python main\_simple.py --train 1 --img-dir 'TRAIN\_IMG\_DIR' --ann-dir 'ANN\_DIR' --arch 'vgg16' --optim 'adam' --workers 8 --batch-size 32 --start-epoch 1 --end-epoch 100 --learning-rate 0.0001 --momentum 0.9 --weight-decay 0.0001 --save-interval 1 --print-freq 50 
* get\_optimal\_epoch.py 
* _(test vgg16)_ python main\_simple.py --train 0 --img-dir 'TEST\_IMG\_DIR' --ann-dir 'ANN\_DIR' --arch 'vgg16' --load-epoch XX

-----
GAM Generation
-----
* save\_gam\_image.m
* save\_resize\_gam\_image.m

-----
Run GAP-GAS
-----
* _(train vgg16)_ python main\_gap\_gas.py --train 1 --img-dir 'TRAIN\_IMG\_DIR' --gam-dir 'GAM\_IMG\_DIR' --ann-dir 'ANN\_DIR' --arch 'vgg16' --optim 'adam' --workers 8 --batch-size 32 --start-epoch 1 --end-epoch 100 --learning-rate 0.0001 --momentum 0.9 --weight-decay 0.0001 --save-interval 1 --print-freq 50 
* get\_optimal\_epoch.py
* _(test vgg16)_ python main\_gap\_gas.py --train 0 --img-dir 'TEST\_IMG\_DIR' --ann-dir 'ANN\_DIR' --arch 'vgg16' --load-epoch XX

* print\_test\_stats.py _(generate test statistics)_
