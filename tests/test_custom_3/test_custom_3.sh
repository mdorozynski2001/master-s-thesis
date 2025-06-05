mkdir -p runs/detect/tests/custom_model/test_custom_3/

train_custom datasets/unitem_all --batch-size 16 --bacbkone-trainable-layers 4 --image-size 480
eval_custom datasets/unitem_all --model-path runs/detect/custom_model/train/
mv runs/detect/custom_model/train runs/detect/tests/custom_model/test_custom_3/train_image_size_480

train_custom datasets/unitem_all --batch-size 16 --bacbkone-trainable-layers 4 --image-size 576
eval_custom datasets/unitem_all --model-path runs/detect/custom_model/train/
mv runs/detect/custom_model/train runs/detect/tests/custom_model/test_custom_3/train_image_size_576

train_custom datasets/unitem_all --batch-size 16 --bacbkone-trainable-layers 4 --image-size 640
eval_custom datasets/unitem_all --model-path runs/detect/custom_model/train/
mv runs/detect/custom_model/train runs/detect/tests/custom_model/test_custom_3/train_image_size_640

train_custom datasets/unitem_all --batch-size 16 --bacbkone-trainable-layers 4 --image-size 800
eval_custom datasets/unitem_all --model-path runs/detect/custom_model/train/
mv runs/detect/custom_model/train runs/detect/tests/custom_model/test_custom_3/train_image_size_800

train_custom datasets/unitem_all --batch-size 16 --bacbkone-trainable-layers 4 --image-size 960
eval_custom datasets/unitem_all --model-path runs/detect/custom_model/train/
mv runs/detect/custom_model/train runs/detect/tests/custom_model/test_custom_3/train_image_size_960

compare_custom runs/detect/tests/custom_model/test_custom_3/ --output-dir tests/test_custom_3/