

## Repository Setup
1. Create a fresh conda environment, and install all dependencies.
    ```
    conda create -n env_MACA python=3.7
    conda activate env_MACA
    git clone https://github.com/xxayt/MACA.git
    cd MACA
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
    ```

2. Install pytorch
    ```
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch==1.7.1 torchvision==0.8.2
    ```

2. Install others
    ```
    apt update
    apt-get install tmux
    apt-get install aria2c
    cd tools/refer
    python setup.py install
    make
    ```


## Experiment for fine-tune

所有实验均采用pre-train-and-fine-tune方案，本repo主要尝试将x类模型迁移学习到某些下游任务中

- download pre-training weights:
  ```
  aria2c [pre-training weights url]
  ```
  
- see acc/loss curve by tensorboard

  ```
  tensorboard --logdir ./logs/[path of tensorboard file]
  ```

  

### 1 ViT $^{[3]}$

tree only for ViT

```
~/MACA# tree
.
├── data
│   └── cifar100
│       └── cifar-100-python
│           ├── file.txt~
│           ├── meta
│           ├── test
│           └── train
├── logs
│   └── cifar100
│       └── vit_base_32..
├── train_vit.py
├── vit
│   ├── cifar100_dataset.py
│   ├── model_rawvit.py
│   ├── model_ViT.py
│   ├── pretrain
│   │   ├── README.md
│   │   ├── vit_base_patch32_224_in21k.pth
│   │   └── vit_large_patch16_224_in21k.pth
│   ├── __pycache__
│   │   ├── cifar100_dataset.cpython-38.pyc
│   │   ├── model_rawvit.cpython-38.pyc
│   │   ├── model_ViT.cpython-38.pyc
│   │   └── utils.cpython-38.pyc
│   └── utils.py
```

- **Introduction**：完全使用Transformer处理CV任务

- **Model**：将输入图片拆分为patch通过linear embedding后，将flattened patch放入Transformer

  <img src=".\image\vit.png" alt="vit" style="zoom:40%;" />

- **Pre-Training Weights**：
  
  |       model       |                   Pre-train on ImageNet1k                    |                   Pre-train on ImageNet21k                   |
  | :---------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
  | base-patch16-224  | [vit_base_patch16_224_in1k](https://pan.baidu.com/s/1zqb08naP0RPqqfSXfkB2EA), key: eu9f | [vit_base_patch16_224_in21k](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth) |
  | base-patch32-224  | [vit_base_patch32_224_in1k](https://pan.baidu.com/s/1hCv0U8pQomwAtHBYc4hmZg), key: s5hl | [vit_base_patch32_224_in21k](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch32_224_in21k-8db57226.pth) |
  | large-patch16-224 | [vit_large_patch16_224_in1k](https://pan.baidu.com/s/1cxBgZJJ6qUWPSBNcE4TdRQ), key: qqt8 | [vit_large_patch16_224_in21k](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch16_224_in21k-606da67d.pth) |
  | large-patch32-224 |                              /                               | [vit_large_patch32_224_in21k](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_224_in21k-9046d2e7.pth) |
  | huge-patch14-224  |                              /                               |                   not currently availabel                    |
  
- **Transfer Tasks**：
  - cifar100
    - dataset：[cifar100](http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz)
    - train for vit_base_patch32_224_ImageNet21k
      ```
      python train_vit.py --name vit_base_32 --data 'cifar100' --model_file 'model_ViT' \
      --model_name 'vit_base_patch32_224_ImageNet21k' --pretrain 'vit_base_patch32_224_in21k.pth' \
      --batch_size 128 --lr 0.002
      ```
      train for vit_large_patch16_224_ImageNet21k
      ```
      python train_vit.py --name vit_large_16 --data 'cifar100' --model_file 'model_ViT' \
      --model_name 'vit_large_patch16_224_ImageNet21k' --pretrain 'vit_large_patch16_224_in21k.pth' \
      --batch_size 32 --lr 0.004
      ```
    - test

### 2 ViLBERT $^{[7]}$

- **Introduction**：
  - 第一个提出Co-Attention（即交换Attention中不同模态的query）的模态融合方法。
  - 在Conceptual Captions数据集上进行VLP；再迁移到下游的四个vision-language任务中（VQA、VCR、Grounding Referring Expressions、Caption-Based Image Retrieval）。
- **Model**：
  - Textual Embedder：BERT
  - Visual Embedder：Faster-CNN ?
  - Modality Interaction：Co-Attention

  <img src=".\image\vilbert.png" alt="vilbert" style="zoom:50%;" />

- **Pre-Training Tasks**：
  1. masked multi-modal modelling
  2. multi-modal alignment prediction

- **Pre-Training Weights**：

  |       model       |                   Pre-train on ImageNet1k                    |                   Pre-train on ImageNet21k                   |
  | :---------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
  | ..  | .. | .. |

- **Transfer Tasks**：
  - VQA
    - dataset：VQA 2.0（1.1 million questions about COCO images each with 10 answers）
  - VCR

  - Grounding Referring Expressions
    - dataset：RefCOCO+
    - train
      ```
      CUDA_VISIBLE_DEVICES=0 python vilbert_train_tasks.py --bert_model bert-base-uncased --from_pretrained vilbert/pretrain/bert_base_6_layer_6_connect_freeze_0/pytorch_model_8.bin --config_file vilbert/config/bert_base_6layer_6conect.json  --learning_rate 4e-5 --num_workers 0 --tasks 4 --save_name pretrained
      ```
    - evaluation
      ```
      CUDA_VISIBLE_DEVICES=0 python vilbert_eval_tasks.py --bert_model bert-base-uncased --from_pretrained save/refcoco+_bert_base_6layer_6conect-pretrained/pytorch_model_19.bin --config_file config/bert_base_6layer_6conect.json --task 4
      ```

  - Caption-Based Image Retrieval 基于标题的图像检索
    - dataset：Flickr30k（31,000 images from Flickr with five captions each）
    - train
      ```
      CUDA_VISIBLE_DEVICES=0 python vilbert_train_tasks.py --bert_model bert-base-uncased --from_pretrained vilbert/pretrain/bert_base_6_layer_6_connect_freeze_0/pytorch_model_8.bin  --config_file vilbert/config/bert_base_6layer_6conect.json  --learning_rate 4e-5 --num_workers 0 --tasks 3 --save_name pretrained
      ```
    - evaluation
      ```
      CUDA_VISIBLE_DEVICES=0 python vilbert_eval_retrieval.py --bert_model bert-base-uncased --from_pretrained save/RetrievalFlickr30k_bert_base_6layer_6conect-pretrained/pytorch_model_19.bin --config_file config/bert_base_6layer_6conect.json --task 3 --split test --batch_size 1
      ```

### 3 ViLT

- 介绍
- 模型
- 下游任务测试

### 4 CrossViT

- 介绍
- 模型
- 下游任务测试



## Reference

- [1] [Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762)
- [2] [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (2021)](https://arxiv.org/abs/2010.11929)
- [3] [https://github.com/google-research/vision_transformer](https://github.com/google-research/vision_transformer)
- [4] [Multimodal Learning with Transformers: A Survey (2022)](https://arxiv.org/abs/2206.06488)
- [5] [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (2018)](https://arxiv.org/abs/1810.04805)
- [6] [ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks (2019)](https://arxiv.org/abs/1908.02265)
- [7] [https://github.com/google-research/vision_transformer](https://github.com/google-research/vision_transformer)
- [8] [ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision (2021)](https://arxiv.org/abs/2102.03334)
- [9] [CrossViT: Cross-Attention Multi-Scale Vision Transformer for Image Classification (2021)](https://arxiv.org/abs/2103.14899)

