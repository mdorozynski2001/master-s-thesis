mkdir -p runs/detect/tests/custom_model/test_custom_2_RGB/

train_custom datasets/unitem_all --batch-size 16 --bacbkone-trainable-layers 8
eval_custom datasets/unitem_all --model-path runs/detect/custom_model/train/
mv runs/detect/custom_model/train runs/detect/tests/custom_model/test_custom_2_RGB/train_016_batch_size_08_layers_trainable

train_custom datasets/unitem_all --batch-size 32 --bacbkone-trainable-layers 8
eval_custom datasets/unitem_all --model-path runs/detect/custom_model/train/
mv runs/detect/custom_model/train runs/detect/tests/custom_model/test_custom_2_RGB/train_032_batch_size_08_layers_trainable

train_custom datasets/unitem_all --batch-size 64 --bacbkone-trainable-layers 8
eval_custom datasets/unitem_all --model-path runs/detect/custom_model/train/
mv runs/detect/custom_model/train runs/detect/tests/custom_model/test_custom_2_RGB/train_064_batch_size_08_layers_trainable

train_custom datasets/unitem_all --batch-size 128 --bacbkone-trainable-layers 8
eval_custom datasets/unitem_all --model-path runs/detect/custom_model/train/
mv runs/detect/custom_model/train runs/detect/tests/custom_model/test_custom_2_RGB/train_128_batch_size_08_layers_trainable

train_custom datasets/unitem_all --batch-size 8 --bacbkone-trainable-layers 8
eval_custom datasets/unitem_all --model-path runs/detect/custom_model/train/
mv runs/detect/custom_model/train runs/detect/tests/custom_model/test_custom_2_RGB/train_008_batch_size_08_layers_trainable

train_custom datasets/unitem_all --batch-size 8 --bacbkone-trainable-layers 4
eval_custom datasets/unitem_all --model-path runs/detect/custom_model/train/
mv runs/detect/custom_model/train runs/detect/tests/custom_model/test_custom_2_RGB/train_008_batch_size_04_layers_trainable

train_custom datasets/unitem_all --batch-size 16 --bacbkone-trainable-layers 4
eval_custom datasets/unitem_all --model-path runs/detect/custom_model/train/
mv runs/detect/custom_model/train runs/detect/tests/custom_model/test_custom_2_RGB/train_016_batch_size_04_layers_trainable

train_custom datasets/unitem_all --batch-size 32 --bacbkone-trainable-layers 4
eval_custom datasets/unitem_all --model-path runs/detect/custom_model/train/
mv runs/detect/custom_model/train runs/detect/tests/custom_model/test_custom_2_RGB/train_032_batch_size_04_layers_trainable

train_custom datasets/unitem_all --batch-size 64 --bacbkone-trainable-layers 4
eval_custom datasets/unitem_all --model-path runs/detect/custom_model/train/
mv runs/detect/custom_model/train runs/detect/tests/custom_model/test_custom_2_RGB/train_064_batch_size_04_layers_trainable

train_custom datasets/unitem_all --batch-size 128 --bacbkone-trainable-layers 4
eval_custom datasets/unitem_all --model-path runs/detect/custom_model/train/
mv runs/detect/custom_model/train runs/detect/tests/custom_model/test_custom_2_RGB/train_128_batch_size_04_layers_trainable

train_custom datasets/unitem_all --batch-size 8 --bacbkone-trainable-layers 30
eval_custom datasets/unitem_all --model-path runs/detect/custom_model/train/
mv runs/detect/custom_model/train runs/detect/tests/custom_model/test_custom_2_RGB/train_008_batch_size_20_layers_trainable

compare_custom runs/detect/tests/custom_model/test_custom_2_RGB/ --output-dir tests/test_custom_2_RGB/
