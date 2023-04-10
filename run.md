## ViT

- vit_base_32:
    python train_vit.py --name vit_base_32_2 --data 'cifar100' --model_file 'model_ViT' \
--model_name 'vit_base_patch32_224_ImageNet21k' --pretrain 'vit_base_patch32_224_in21k.pth' \
--batch_size 128 --lr 0.004
- vit_base_32: use model_rawvit
    python train_vit.py --name vit_base_32_2 --data 'cifar100' --model_file 'model_rawvit' \
--model_name 'vit_base_patch32_224_in21k' --pretrain 'vit_base_patch32_224_in21k.pth' \
--batch_size 128 --lr 0.004
- vit_base_32: use model_rawvit lr = 0.002
    python train_vit.py --name vit_base_32_lr2_e40 --data 'cifar100' --model_file 'model_rawvit' \
--model_name 'vit_base_patch32_224_in21k' --pretrain 'vit_base_patch32_224_in21k.pth' \
--batch_size 128 --epochs 40 --lr 0.002

- vit_large_16:
    python train_vit.py --name vit_large_16 --data 'cifar100' --model_file 'model_ViT' \
--model_name 'vit_large_patch16_224_ImageNet21k' --pretrain 'vit_large_patch16_224_in21k.pth' \
--batch_size 32 --lr 0.004

- 查看tensorboard
    tensorboard --logdir ./logs/cifar100/vit_base_32


## ViLBERT

- RefCOCO+ for Grounding Referring Expressions
  - train
    CUDA_VISIBLE_DEVICES=0 python vilbert_train_tasks.py --bert_model bert-base-uncased \
    --from_pretrained vilbert/pretrain/bert_base_6_layer_6_connect_freeze_0/pytorch_model_8.bin \
    --config_file vilbert/config/bert_base_6layer_6conect.json  --learning_rate 4e-5 --num_workers 0 \
    --tasks 4 --save_name pretrained
  - eval
    CUDA_VISIBLE_DEVICES=0 python vilbert_eval_tasks.py --bert_model bert-base-uncased --from_pretrained vilbert/pretrain/refcoco+_bert_base_6layer_6conect-pretrained/pytorch_model_19.bin --config_file vilbert/config/bert_base_6layer_6conect.json --task 4

- Flickr30k+ for Caption-Based Image Retrieval
  - train
    CUDA_VISIBLE_DEVICES=0 python vilbert_train_tasks.py --bert_model bert-base-uncased \
    --from_pretrained vilbert/pretrain/bert_base_6_layer_6_connect_freeze_0/pytorch_model_8.bin  \
    --config_file vilbert/config/bert_base_6layer_6conect.json  --learning_rate 4e-5 --num_workers 0 \
    --tasks 3 --save_name pretrained
  - eval
    CUDA_VISIBLE_DEVICES=0 python vilbert_eval_retrieval.py --bert_model bert-base-uncased \
    --from_pretrained vilbert/pretrain/RetrievalFlickr30k_bert_base_6layer_6conect-pretrained/pytorch_model_19.bin \
    --config_file vilbert/config/bert_base_6layer_6conect.json --task 3 --split test --batch_size 1
    