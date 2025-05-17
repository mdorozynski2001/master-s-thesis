mkdir -p runs/detect/tests/custom_model/test_custom_2/

train_custom dataset/unitem_all --epochs 100
evaluate_custom dataset/unitem_all --model-path runs/detect/custom_model/train/
mv runs/detect/custom_model/train runs/detect/tests/custom_model/test_custom_2/train_100_epochs

train_custom dataset/unitem_all --epochs 200
evaluate_custom dataset/unitem_all --model-path runs/detect/custom_model/train/
mv runs/detect/custom_model/train runs/detect/tests/custom_model/test_custom_2/train_200_epochs

train_custom dataset/unitem_all --epochs 300
evaluate_custom dataset/unitem_all --model-path runs/detect/custom_model/train/
mv runs/detect/custom_model/train runs/detect/tests/custom_model/test_custom_2/train_300_epochs

train_custom dataset/unitem_all --epochs 400
evaluate_custom dataset/unitem_all --model-path runs/detect/custom_model/train/
mv runs/detect/custom_model/train runs/detect/tests/custom_model/test_custom_2/train_400_epochs

train_custom dataset/unitem_all --batch-size 16
evaluate_custom dataset/unitem_all --model-path runs/detect/custom_model/train/
mv runs/detect/custom_model/train runs/detect/tests/custom_model/test_custom_2/train_16_batch_size

train_custom dataset/unitem_all --batch-size 32
evaluate_custom dataset/unitem_all --model-path runs/detect/custom_model/train/
mv runs/detect/custom_model/train runs/detect/tests/custom_model/test_custom_2/train_32_batch_size

train_custom dataset/unitem_all --batch-size 64
evaluate_custom dataset/unitem_all --model-path runs/detect/custom_model/train/
mv runs/detect/custom_model/train runs/detect/tests/custom_model/test_custom_2/train_64_batch_size

train_custom dataset/unitem_all --batch-size 128
evaluate_custom dataset/unitem_all --model-path runs/detect/custom_model/train/
mv runs/detect/custom_model/train runs/detect/tests/custom_model/test_custom_2/train_128_batch_size

compare_custom runs/detect/tests/custom_model/test_custom_2/ --output-dir tests/test_custom_2/
