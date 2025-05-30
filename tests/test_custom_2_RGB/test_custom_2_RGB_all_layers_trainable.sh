mkdir -p runs/detect/tests/custom_model/test_custom_2_RGB/

train_custom datasets/unitem_all --batch-size 16 --bacbkone-trainable-layers 30
eval_custom datasets/unitem_all --model-path runs/detect/custom_model/train/
mv runs/detect/custom_model/train runs/detect/tests/custom_model/test_custom_2_RGB/train_16_batch_size_full

train_custom datasets/unitem_all --batch-size 32 --bacbkone-trainable-layers 30
eval_custom datasets/unitem_all --model-path runs/detect/custom_model/train/
mv runs/detect/custom_model/train runs/detect/tests/custom_model/test_custom_2_RGB/train_32_batch_size_full

train_custom datasets/unitem_all --batch-size 64 --bacbkone-trainable-layers 30
eval_custom datasets/unitem_all --model-path runs/detect/custom_model/train/
mv runs/detect/custom_model/train runs/detect/tests/custom_model/test_custom_2_RGB/train_64_batch_size_full

train_custom datasets/unitem_all --batch-size 128 --bacbkone-trainable-layers 30
eval_custom datasets/unitem_all --model-path runs/detect/custom_model/train/
mv runs/detect/custom_model/train runs/detect/tests/custom_model/test_custom_2_RGB/train_128_batch_size_full

compare_custom runs/detect/tests/custom_model/test_custom_2_RGB/ --output-dir tests/test_custom_2_RGB/
