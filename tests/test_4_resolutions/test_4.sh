mkdir runs/detect/train
train datasets/unitem_all/train.yaml --image-width 480
mv runs/detect/train runs/detect/train_480_width
mkdir runs/detect/train
train datasets/unitem_all/train.yaml --image-width 576
mv runs/detect/train runs/detect/train_576_width
mkdir runs/detect/train
train datasets/unitem_all/train.yaml --image-width 640
mv runs/detect/train runs/detect/train_640_width
mkdir runs/detect/train
train datasets/unitem_all/train.yaml --image-width 768
mv runs/detect/train runs/detect/train_768_width
mkdir runs/detect/train
train datasets/unitem_all/train.yaml --image-width 960
mv runs/detect/train runs/detect/train_960_width
mkdir runs/detect/tests/test_4
mv runs/detect/train_* runs/detect/tests/test_4
bash compare_models.sh -d runs/detect/tests/test_4/ -o tests/test_4_resolutions -r true -e true
