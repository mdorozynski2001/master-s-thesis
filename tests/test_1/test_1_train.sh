train datasets/unitem_all/train.yaml --epochs 10
mv runs/detect/train runs/detect/train_010_epochs
train datasets/unitem_all/train.yaml --epochs 25
mv runs/detect/train runs/detect/train_025_epochs
train datasets/unitem_all/train.yaml --epochs 50
mv runs/detect/train runs/detect/train_050_epochs
train datasets/unitem_all/train.yaml --epochs 100
mv runs/detect/train runs/detect/train_100_epochs
train datasets/unitem_all/train.yaml --epochs 150
mv runs/detect/train runs/detect/train_150_epochs
train datasets/unitem_all/train.yaml --epochs 200
mv runs/detect/train runs/detect/train_200_epochs
train datasets/unitem_all/train.yaml --epochs 300
mv runs/detect/train runs/detect/train_300_epochs
train datasets/unitem_all/train.yaml --epochs 400
mv runs/detect/train runs/detect/train_400_epochs
train datasets/unitem_all/train.yaml --epochs 500
mv runs/detect/train runs/detect/train_500_epochs
mkdir runs/detect/tests/test_1
mv runs/detect/train_* runs/detect/tests/test_1
bash compare_models.sh -d runs/detect/tests/test_1/ -o tests/test_1
