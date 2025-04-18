mkdir runs/detect/tests/custom_model/test_custom_1

train_custom datasets/unitem_all/ --lambda-coord 5.0 --lambda-noobj 0.5 --lambda-conf 1.0 --lambda-iou 3.0 --lambda-metrics 3.0
eval_custom datasets/unitem_all/ --model-path runs/detect/custom_model/train
mv runs/detect/custom_model/train runs/detect/custom_model/test_custom_1/test_1

train_custom datasets/unitem_all/ --lambda-coord 3.0 --lambda-noobj 0.3 --lambda-conf 1.5 --lambda-iou 2.0 --lambda-metrics 4.0
eval_custom datasets/unitem_all/ --model-path runs/detect/custom_model/train
mv runs/detect/custom_model/train runs/detect/custom_model/test_custom_1/test_2

train_custom datasets/unitem_all/ --lambda-coord 7.0 --lambda-noobj 0.7 --lambda-conf 0.8 --lambda-iou 4.0 --lambda-metrics 2.0
eval_custom datasets/unitem_all/ --model-path runs/detect/custom_model/train
mv runs/detect/custom_model/train runs/detect/custom_model/test_custom_1/test_3

train_custom datasets/unitem_all/ --lambda-coord 4.0 --lambda-noobj 0.4 --lambda-conf 1.2 --lambda-iou 3.5 --lambda-metrics 3.5
eval_custom datasets/unitem_all/ --model-path runs/detect/custom_model/train
mv runs/detect/custom_model/train runs/detect/custom_model/test_custom_1/test_4

train_custom datasets/unitem_all/ --lambda-coord 6.0 --lambda-noobj 0.6 --lambda-conf 0.9 --lambda-iou 2.5 --lambda-metrics 4.5
eval_custom datasets/unitem_all/ --model-path runs/detect/custom_model/train
mv runs/detect/custom_model/train runs/detect/custom_model/test_custom_1/test_5

train_custom datasets/unitem_all/ --lambda-coord 8.0 --lambda-noobj 0.8 --lambda-conf 0.7 --lambda-iou 3.8 --lambda-metrics 2.5
eval_custom datasets/unitem_all/ --model-path runs/detect/custom_model/train
mv runs/detect/custom_model/train runs/detect/custom_model/test_custom_1/test_6

train_custom datasets/unitem_all/ --lambda-coord 4.5 --lambda-noobj 0.45 --lambda-conf 1.3 --lambda-iou 2.8 --lambda-metrics 3.8
eval_custom datasets/unitem_all/ --model-path runs/detect/custom_model/train
mv runs/detect/custom_model/train runs/detect/custom_model/test_custom_1/test_7

train_custom datasets/unitem_all/ --lambda-coord 6.5 --lambda-noobj 0.65 --lambda-conf 0.85 --lambda-iou 3.2 --lambda-metrics 3.2
eval_custom datasets/unitem_all/ --model-path runs/detect/custom_model/train
mv runs/detect/custom_model/train runs/detect/custom_model/test_custom_1/test_8

train_custom datasets/unitem_all/ --lambda-coord 5.5 --lambda-noobj 0.55 --lambda-conf 1.1 --lambda-iou 2.2 --lambda-metrics 4.2
eval_custom datasets/unitem_all/ --model-path runs/detect/custom_model/train
mv runs/detect/custom_model/train runs/detect/custom_model/test_custom_1/test_9

train_custom datasets/unitem_all/ --lambda-coord 7.5 --lambda-noobj 0.75 --lambda-conf 0.75 --lambda-iou 3.6 --lambda-metrics 2.8
eval_custom datasets/unitem_all/ --model-path runs/detect/custom_model/train
mv runs/detect/custom_model/train runs/detect/custom_model/test_custom_1/test_10


mv runs/detect/custom_model/test_custom_1 runs/detect/tests/
