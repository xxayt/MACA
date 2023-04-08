

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
    ```


## Experiment

所有实验均采用pre-train-and-fine-tune方案，本repo主要尝试将x类模型迁移学习到某些下游任务中

- download pre-training weights:
  ```
  aria2c [pre-training weights url]
  ```
  

### 1 ViT

- **Introduction**：完全使用Transformer处理CV任务
- **Model**：将输入图片拆分为patch通过linear embedding后，将flattened patch放入Transformer
- **Pre-Training Weights**：
  - [vit_base_patch32_224_ImageNet21k](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch32_224_in21k-8db57226.pth)
  - [vit_large_patch16_224_ImageNet21k](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch16_224_in21k-606da67d.pth)
- **Transfer Tasks**：
  - cifar100
    - dataset：[cifar100](http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz)
    - train
      for vit_base_patch32_224_ImageNet21k
      ```
      python train_vit.py --name vit_base_32 --data 'cifar100' --model_file 'model_ViT' \
      --model_name 'vit_base_patch32_224_ImageNet21k' --pretrain 'vit_base_patch32_224_in21k.pth' \
      --batch_size 128 --lr 0.004
      ```
      for vit_large_patch16_224_ImageNet21k
      ```
      python train_vit.py --name vit_large_16 --data 'cifar100' --model_file 'model_ViT' \
      --model_name 'vit_large_patch16_224_ImageNet21k' --pretrain 'vit_large_patch16_224_in21k.pth' \
      --batch_size 32 --lr 0.004
      ```
    - test

### 2 ViLBERT

- **Introduction**：
  - 第一个提出Co-Attention（即交换Attention中不同模态的query）的模态融合方法。
  - 在Conceptual Captions数据集上进行VLP；再迁移到下游的四个vision-language任务中（VQA、VCR、Grounding Referring Expressions、Caption-Based Image Retrieval）。
- **Model**：
  - Textual Embedder：BERT
  - Visual Embedder：Faster-CNN
  - Modality Interaction：Co-Attention
- **Pre-Training Tasks**：
  1. masked multi-modal modelling
  2. multi-modal alignment prediction
- **Transfer Tasks**：
  - VQA
    - dataset：VQA 2.0（1.1 million questions about COCO images each with 10 answers）
  - VCR
  - Grounding Referring Expressions
    - dataset：RefCOCO+
  - Caption-Based Image Retrieval 基于标题的图像检索
    - dataset：Flickr30k（31,000 images from Flickr with five captions each）
  - python ./tools/generate_tsv.py --cfg experiments/cfgs/faster_rcnn_end2end_resnet.yml --def models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt --out feature/VCR/VCR_resnet101_faster_rcnn_genome.tsv --net data/faster_rcnn_models/resnet101_faster_rcnn_final.caffemodel --total_group 1 --group_id 0 --split VCR
  - python train_tasks.py --bert_model bert-base-uncased --from_pretrained <multi_task_model_path> --config_file config/bert_base_6layer_6conect.json --tasks 1 --lr_scheduler 'warmup_linear' --train_iter_gap 4 --task_specific_tokens --save_name finetune_from_multi_task_model

### 3 ViLT

- 介绍
- 模型
- 下游任务测试

### 4 CrossViT

- 介绍
- 模型
- 下游任务测试



## Reference

1. [Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762)
2. [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (2021)](https://arxiv.org/abs/2010.11929)
3. [Multimodal Learning with Transformers: A Survey (2022)](https://arxiv.org/abs/2206.06488)
4. [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (2018)](https://arxiv.org/abs/1810.04805)
5. [ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks (2019)](https://arxiv.org/abs/1908.02265)
6. [ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision (2021)](https://arxiv.org/abs/2102.03334)
7. [CrossViT: Cross-Attention Multi-Scale Vision Transformer for Image Classification (2021)](https://arxiv.org/abs/2103.14899)

