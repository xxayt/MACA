## ViT

- vit_base_32:
    python train_vit.py --name vit_base_32_2 --data 'cifar100' --model_file 'model_ViT' \
--model_name 'vit_base_patch32_224_ImageNet21k' --pretrain 'vit_base_patch32_224_in21k.pth' \
--batch_size 128 --lr 0.004

    python train_vit.py --name vit_base_32_2 --data 'cifar100' --model_file 'model_rawvit' \
--model_name 'vit_base_patch32_224_in21k' --pretrain 'vit_base_patch32_224_in21k.pth' \
--batch_size 128 --lr 0.004

- vit_large_16:
    python train_vit.py --name vit_large_16 --data 'cifar100' --model_file 'model_ViT' \
--model_name 'vit_large_patch16_224_ImageNet21k' --pretrain 'vit_large_patch16_224_in21k.pth' \
--batch_size 32 --lr 0.004

- 查看tensorboard
    tensorboard --logdir ./logs/cifar100/vit_base_32 