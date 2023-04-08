## ViT

python train.py dataset/cifar100 --dataset torch/CIFAR100 --dataset-download --model vit_base_patch32_224_in21k --pretrained --weight-decay 0 --lr 0.003 -b 512 --amp --pin-mem --epochs

python train_vit.py --name vit_base_32 --data 'inat21_mini' --data_dir './datasets/iNat2021' --model_file 'resnet' --model_name 'resnet50' --pretrained --batch_size 128 --start_lr 0.04 --image_onl