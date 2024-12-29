train datasets/unitem_all/train.yaml --batch-size 2 2>&1 | tee runs/detect/train/training.log
mv runs/detect/train runs/detect/train_002_batch_size
train datasets/unitem_all/train.yaml --batch-size 4 2>&1 | tee runs/detect/train/training.log
mv runs/detect/train runs/detect/train_004_batch_size
train datasets/unitem_all/train.yaml --batch-size 8 2>&1 | tee runs/detect/train/training.log
mv runs/detect/train runs/detect/train_008_batch_size
train datasets/unitem_all/train.yaml --batch-size 16 2>&1 | tee runs/detect/train/training.log
mv runs/detect/train runs/detect/train_016_batch_size
train datasets/unitem_all/train.yaml --batch-size 32 2>&1 | tee runs/detect/train/training.log
mv runs/detect/train runs/detect/train_032_batch_size
train datasets/unitem_all/train.yaml --batch-size 64 2>&1 | tee runs/detect/train/training.log
mv runs/detect/train runs/detect/train_064_batch_size
train datasets/unitem_all/train.yaml --batch-size 128 2>&1 | tee runs/detect/train/training.log
mv runs/detect/train runs/detect/train_128_batch_size
mkdir runs/detect/tests/test_2
mv runs/detect/train_* runs/detect/tests/test_2
bash compare_models.sh -d runs/detect/tests/test_2/ -o tests/test_2
