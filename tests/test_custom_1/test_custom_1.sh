train_custom datasets/unitem_all/ --epochs 10
eval_custom datasets/unitem_all/ --model-path runs/detect/custom_model/train
mv runs/detect/custom_model/train runs/detect/custom_model/test_custom_1_10_epochs

train_custom datasets/unitem_all/ --epochs 50
eval_custom datasets/unitem_all/ --model-path runs/detect/custom_model/train
mv runs/detect/custom_model/train runs/detect/custom_model/test_custom_1_50_epochs

train_custom datasets/unitem_all/ --epochs 100
eval_custom datasets/unitem_all/ --model-path runs/detect/custom_model/train
mv runs/detect/custom_model/train runs/detect/custom_model/test_custom_1_100_epochs

train_custom datasets/unitem_all/ --epochs 200
eval_custom datasets/unitem_all/ --model-path runs/detect/custom_model/train
mv runs/detect/custom_model/train runs/detect/custom_model/test_custom_1_200_epochs

train_custom datasets/unitem_all/ --epochs 300
eval_custom datasets/unitem_all/ --model-path runs/detect/custom_model/train
mv runs/detect/custom_model/train runs/detect/custom_model/test_custom_1_300_epochs

train_custom datasets/unitem_all/ --batch-size 16
eval_custom datasets/unitem_all/ --model-path runs/detect/custom_model/train
mv runs/detect/custom_model/train runs/detect/custom_model/test_custom_1_16_batch_size

train_custom datasets/unitem_all/ --batch-size 32
eval_custom datasets/unitem_all/ --model-path runs/detect/custom_model/train
mv runs/detect/custom_model/train runs/detect/custom_model/test_custom_1_32_batch_size

train_custom datasets/unitem_all/ --batch-size 64
eval_custom datasets/unitem_all/ --model-path runs/detect/custom_model/train
mv runs/detect/custom_model/train runs/detect/custom_model/test_custom_1_64_batch_size

train_custom datasets/unitem_all/ --batch-size 128
eval_custom datasets/unitem_all/ --model-path runs/detect/custom_model/train
mv runs/detect/custom_model/train runs/detect/custom_model/test_custom_1_128_batch_size

train_custom datasets/unitem_all/ --batch-size 256
eval_custom datasets/unitem_all/ --model-path runs/detect/custom_model/train
mv runs/detect/custom_model/train runs/detect/custom_model/test_custom_1_256_batch_size

train_custom datasets/unitem_all/ --image-size 300
eval_custom datasets/unitem_all/ --model-path runs/detect/custom_model/train --image-size 300
mv runs/detect/custom_model/train runs/detect/custom_model/test_custom_1_300_image_size

train_custom datasets/unitem_all/ --image-size 400
eval_custom datasets/unitem_all/ --model-path runs/detect/custom_model/train --image-size 400
mv runs/detect/custom_model/train runs/detect/custom_model/test_custom_1_400_image_size

train_custom datasets/unitem_all/ --image-size 500
eval_custom datasets/unitem_all/ --model-path runs/detect/custom_model/train --image-size 500
mv runs/detect/custom_model/train runs/detect/custom_model/test_custom_1_500_image_size

train_custom datasets/unitem_all/ --image-size 600
eval_custom datasets/unitem_all/ --model-path runs/detect/custom_model/train --image-size 600
mv runs/detect/custom_model/train runs/detect/custom_model/test_custom_1_600_image_size

mkdir runs/detect/tests/test_custom_1
mv runs/detect/custom_model/test_custom_1* runs/detect/tests/test_custom_1
