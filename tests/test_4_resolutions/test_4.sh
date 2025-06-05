train datasets/unitem_all/train.yaml --image_width 480
mv runs/detect/train runs/detect/train_480_width
train datasets/unitem_all/train.yaml --image_width 576
mv runs/detect/train runs/detect/train_576_width
train datasets/unitem_all/train.yaml --image_width 640
mv runs/detect/train runs/detect/train_640_width
train datasets/unitem_all/train.yaml --image_width 768
mv runs/detect/train runs/detect/train_768_width
train datasets/unitem_all/train.yaml --image_width 960
mv runs/detect/train runs/detect/train_960_width
mkdir runs/detect/tests/test_4
mv runs/detect/train_* runs/detect/tests/test_4
bash compare_models.sh -d runs/detect/tests/test_4/ -o tests/test_4_resolutions -r true -e true
