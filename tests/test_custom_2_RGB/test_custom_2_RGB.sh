mkdir -p runs/detect/tests/custom_model/test_custom_2_RGB/

train_custom datasets/unitem_all --batch-size 16
eval_custom datasets/unitem_all --model-path runs/detect/custom_model/train/
mv runs/detect/custom_model/train runs/detect/tests/custom_model/test_custom_2_RGB/train_16_batch_size

train_custom datasets/unitem_all --batch-size 32
eval_custom datasets/unitem_all --model-path runs/detect/custom_model/train/
mv runs/detect/custom_model/train runs/detect/tests/custom_model/test_custom_2_RGB/train_32_batch_size

train_custom datasets/unitem_all --batch-size 64
eval_custom datasets/unitem_all --model-path runs/detect/custom_model/train/
mv runs/detect/custom_model/train runs/detect/tests/custom_model/test_custom_2_RGB/train_64_batch_size

train_custom datasets/unitem_all --batch-size 128
eval_custom datasets/unitem_all --model-path runs/detect/custom_model/train/
mv runs/detect/custom_model/train runs/detect/tests/custom_model/test_custom_2_RGB/train_128_batch_size

compare_custom runs/detect/tests/custom_model/test_custom_2_RGB/ --output-dir tests/test_custom_2_RGB/
