train datasets/test_3/unitem_only/train.yaml --epochs 100
mv runs/detect/train runs/detect/train_unitem_only_100_epochs
train datasets/test_3/synthetic/train.yaml --epochs 100
mv runs/detect/train runs/detect/train_synthetic_100_epochs
train datasets/test_3/from_internet/train.yaml --epochs 100
mv runs/detect/train runs/detect/train_from_internet_100_epochs
train datasets/test_3/unitem_only/train.yaml --epochs 200
mv runs/detect/train runs/detect/train_unitem_only_200_epochs
train datasets/test_3/synthetic/train.yaml --epochs 200
mv runs/detect/train runs/detect/train_synthetic_200_epochs
train datasets/test_3/from_internet/train.yaml --epochs 200
mv runs/detect/train runs/detect/train_from_internet_200_epochs
train datasets/test_3/unitem_only/train.yaml --epochs 300
mv runs/detect/train runs/detect/train_unitem_only_300_epochs
train datasets/test_3/synthetic/train.yaml --epochs 300
mv runs/detect/train runs/detect/train_synthetic_300_epochs
train datasets/test_3/from_internet/train.yaml --epochs 300
mv runs/detect/train runs/detect/train_from_internet_300_epochs
train datasets/test_3/unitem_only/train.yaml --epochs 400
mv runs/detect/train runs/detect/train_unitem_only_400_epochs
train datasets/test_3/synthetic/train.yaml --epochs 400
mv runs/detect/train runs/detect/train_synthetic_400_epochs
train datasets/test_3/from_internet/train.yaml --epochs 400
mv runs/detect/train runs/detect/train_from_internet_400_epochs
mkdir runs/detect/tests/test_3
mv runs/detect/train_* runs/detect/tests/test_3
bash compare_models.sh -d runs/detect/tests/test_3/ -o tests/test_3 -r true
