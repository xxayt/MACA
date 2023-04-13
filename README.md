- [Theory](#Theory)
- [Experiment](#Experiment)
  - [Repository Setup](#Repository-Setup)
  - [Experiment for fine-tune](#Experiment-for-fine-tune)
    - [1 ViT](#1-ViT)
      - [Introduction](#1-ViT)
      - [Model](#1-ViT)
      - [Pre-Training Weights](#1-ViT)
      - [Transfer Tasks](#1-ViT)
    - [2 ViLBERT](#2-ViLBERT)
      - [Introduction](#2-ViLBERT)
      - [Model](#2-ViLBERT)
      - [Pre-Training Tasks](#2-ViLBERT)
      - [Pre-Training Weights](#2-ViLBERT)
      - [Transfer Tasks](#2-ViLBERT)
- [Reference](#Reference)

I mainly discuss **the influence of Multi-Head Attention in Cross-Modal Transformers** from both **theoretical and experimental** perspectives.

---



# Theory

Check [Q&A](./Q&A.pdf) for more details, pay attention to **3.2 Transformer in MultiModel task**

# Experiment

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
  tensorboard --logdir ./logs/[path of tensorboard file] --port=[eg:6008]
  ```

  

### 1 ViT

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
│       └── ..
├── vit_train.py
└── vit
    ├── cifar100_dataset.py
    ├── model_rawvit.py
    ├── model_ViT.py
    ├── pretrain
    │   ├── README.md
    │   ├── vit_base_patch32_224_in21k.pth
    │   └── vit_large_patch16_224_in21k.pth
    └── utils.py
```

- **Introduction**：完全使用Transformer处理CV任务

- **Model**：将输入图片拆分为patch通过linear embedding后，将flattened patch放入Transformer

  <img src=".\image\vit.png" alt="vit" style="zoom:40%;" />

  ```python
  def forward(self, x):
          B, N, C = x.shape
          qkv = self.qkv(x)
          # 拆分多头
          qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
          q, k, v = qkv[0], qkv[1], qkv[2]
          # QK点积
          attn = (q @ k.transpose(-2, -1)) * self.scale
          attn = self.attn_drop(attn.softmax(dim=-1))
          # attn-score和V点积
          x = (attn @ v).transpose(1, 2).reshape(B, N, C)
          # 拼接后，MLP映射
          x = self.proj(x)
          x = self.proj_drop(x)
          return x
  ```

  

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
      python vit_train.py --name vit_base_32 --data 'cifar100' --model_file 'model_ViT' \
      --model_name 'vit_base_patch32_224_ImageNet21k' --pretrain 'vit_base_patch32_224_in21k.pth' \
      --batch_size 128 --lr 0.002
      ```
      train for vit_large_patch16_224_ImageNet21k
      ```
      python vit_train.py --name vit_large_16 --data 'cifar100' --model_file 'model_ViT' \
      --model_name 'vit_large_patch16_224_ImageNet21k' --pretrain 'vit_large_patch16_224_in21k.pth' \
      --batch_size 32 --lr 0.004
      ```
      
    - test
    
    |         model          | src paper |  me   |
    | :--------------------: | :-------: | :---: |
    | ViT-B/32 (ImageNet21k) |   91.97   | 91.82 |
    | ViT-L/16 (ImageNet21k) |   93.25   | 93.27 |

### 2 ViLBERT

tree only for ViLBERT

```

```

- **Introduction**：
  
  - 第一个提出Co-Attention（即交换Attention中不同模态的query）的模态融合方法。
  - 在Conceptual Captions数据集上进行VLP；再迁移到下游的四个vision-language任务中（VQA、VCR、Grounding Referring Expressions、Caption-Based Image Retrieval）。
  
- **Model**：
  - Textual Embedder：BERT
  - Visual Embedder：Faster-CNN ?
  - Modality Interaction：Co-Attention

  <img src=".\image\vilbert.png" alt="vilbert" style="zoom:50%;" />

  ```python
  '''Co-Attention
  '''
  # Take the dot product between "query2" and "key1" to get the raw attention scores for value 1.
  attention_scores1 = torch.matmul(query_layer2, key_layer1.transpose(-1, -2))
  attention_scores1 = attention_scores1 / math.sqrt(self.attention_head_size)
  
  if use_co_attention_mask:
      attention_scores1 = attention_scores1 + co_attention_mask.permute(0,1,3,2)
  
  # Normalize the attention scores to probabilities.
  attention_probs1 = nn.Softmax(dim=-1)(attention_scores1)
  
  # This is actually dropping out entire tokens to attend to, which might
  # seem a bit unusual, but is taken from the original Transformer paper.
  attention_probs1 = self.dropout1(attention_probs1)
  
  context_layer1 = torch.matmul(attention_probs1, value_layer1)
  # [bs, num_attention_heads, seq_length, attention_head_size] -> [bs, seq_length, num_attention_heads, attention_head_size]
  context_layer1 = context_layer1.permute(0, 2, 1, 3).contiguous()
  # [bs, seq_length, num_attention_heads, attention_head_size] -> [bs, seq_length, all_head_size]
  new_context_layer_shape1 = context_layer1.size()[:-2] + (self.all_head_size,)
  context_layer1 = context_layer1.view(*new_context_layer_shape1)
  ```
  
  
  
- **Pre-Training Tasks**：
  
  1. masked multi-modal modelling
  2. multi-modal alignment prediction
  
- **Pre-Training Weights**：see [https://github.com/google-research/vision_transformer](https://github.com/google-research/vision_transformer) for other more weights

  |       tasks       | model: bert_base_6layer_6conect |
  | :---------------: | :----------------------------------------------------------: |
  | Pertrained on Conceptual Captions for RefCOCO+ Train | [bert_base_6_layer_6_connect_freeze_0](https://drive.google.com/drive/folders/1JVM5WiolJJLnY9_lruxSaSop7IFX8a-v) |
  | Pretrained on RefCOCO+ for RefCOCO+ Eval | [refcoco+_bert_base_6layer_6conect-pretrained](https://drive.google.com/drive/folders/1GWY2fEbZCYHkcnxd0oysU0olfPdzcD3l) |
  | Pertrained on Conceptual Captions for Flickr30k Train | [bert_base_6_layer_6_connect_freeze_0](https://drive.google.com/drive/folders/1JVM5WiolJJLnY9_lruxSaSop7IFX8a-v) |
  | Pretrained on Flickr30k for Flickr30k Eval | [RetrievalFlickr30k_bert_base_6layer_6conect-pretrained](https://drive.google.com/drive/folders/18zUTF3ZyOEuOT1z1aykwtIkBUhfROmJo) |

- **Transfer Tasks**：
  - 视觉问答 VQA
    - dataset：VQA 2.0（1.1 million questions about COCO images each with 10 answers）
    
  - 视觉常识回答 [VCR]([VCR - Dropbox](https://www.dropbox.com/sh/9pgxc3njd3iq03o/AADXgnT1HmEdrds7aujTncBGa?dl=0))
  
  - **引用表达式理解 Grounding Referring Expressions**
    
    - dataset：[RefCOCO+]([referExpression - Dropbox](https://www.dropbox.com/sh/4jqadcfkai68yoe/AADHI6dKviFcraeCMdjiaDENa?dl=0))
    - train
      ```
      CUDA_VISIBLE_DEVICES=0 python vilbert_train_tasks.py --bert_model bert-base-uncased \
      --from_pretrained vilbert/pretrain/bert_base_6_layer_6_connect_freeze_0/pytorch_model_8.bin \
      --config_file vilbert/config/bert_base_6layer_6conect.json  --learning_rate 4e-5 --num_workers 0 \
      --tasks 4
      ```
      try different head_num
      ```
      CUDA_VISIBLE_DEVICES=0 python vilbert_train_tasks.py --bert_model bert-base-uncased \
      --from_pretrained vilbert/pretrain/bert_base_6_layer_6_connect_freeze_0/pytorch_model_8.bin --learning_rate 4e-5 --num_workers 0 --tasks 4 
      # bihead2
      --config_file vilbert/config/bert_base_6layer_6conect_bihead2.json
      # bihead32
      --config_file vilbert/config/bert_base_6layer_6conect_bihead32.json
      # vhead32
      --config_file vilbert/config/bert_base_6layer_6conect_vhead32.json
      # thead48
      --config_file vilbert/config/bert_base_6layer_6conect_thead48.json
      # bi32v32t48
      --config_file vilbert/config/bert_base_6layer_6conect_bi32v32t48.json
      ```

    - evaluation: 用源代码已有模型测试
      ```
      CUDA_VISIBLE_DEVICES=0 python vilbert_eval_tasks.py --bert_model bert-base-uncased --from_pretrained vilbert/pretrain/refcoco+_bert_base_6layer_6conect-pretrained/pytorch_model_19.bin --config_file vilbert/config/bert_base_6layer_6conect.json --task 4
      ```
    - evaluation: 用我跑的模型测试
      ```
      CUDA_VISIBLE_DEVICES=0 python vilbert_eval_tasks.py --bert_model bert-base-uncased --task 4 
      # 测src
      --config_file vilbert/config/bert_base_6layer_6conect.json 
      --from_pretrained logs/refcoco+-bert_base_6layer_6conect-train/pytorch_model_2.bin
      # 测bihead32
      --config_file vilbert/config/bert_base_6layer_6conect_bihead32.json 
      --from_pretrained logs/refcoco+-bert_base_6layer_6conect_bihead32-train/pytorch_model_4.bin
      # 测vhead32
      --config_file vilbert/config/bert_base_6layer_6conect_vhead32.json 
      --from_pretrained logs/refcoco+-bert_base_6layer_6conect_vhead32-train/pytorch_model_1.bin
      # 测thead48
      --config_file vilbert/config/bert_base_6layer_6conect_thead48.json 
      --from_pretrained logs/refcoco+-bert_base_6layer_6conect_thead48-train/pytorch_model_2.bin
      # 测bi=48,v=32,t=48
      --config_file vilbert/config/bert_base_6layer_6conect_bi32v32t48.json 
      --from_pretrained logs/refcoco+-bert_base_6layer_6conect_bi32v32t48-train/pytorch_model_3.bin
      ```
    
    | .json | Co_Att_Heads | Image_Att_Heads | Text_Att_Heads |                          valid(max)                          |                 eval                  | .log |
    | :---: | :----------: | :-------------: | :------------: | :----------------------------------------------------------: | :-----------------------------------: | :--: |
    |       |      2       |        8        |       12       |                        66.453(2.bin)                         |             66.453(2.bin)             |      |
    |  scr  |      8       |        8        |       12       |                         68.49(2.bin)                         |             68.507(2.bin)             |      |
    |       |      2       |        8        |       12       |                  69.576(3.bin)69.79(5.bin)                   |                                       |      |
    |       |      32      |        8        |       12       |                     68.24(4.bin)突然下降                     |             68.247(4.bin)             |      |
    |       |      8       |        8        |       3        | 68.823(3.bin)68.777(4.bin)68.86(5.bin)69.01(6.bin)69.36(8.bin) |                                       |      |
    |       |      8       |       32        |       12       |         68.19(1.bin)68.14(4.bin)61.07(7.bin)突然下降         |      68.210(1.bin)68.135(4.bin)       |      |
    |       |      8       |        8        |       48       |                   67.57(2.bin)67.38(3.bin)                   |             67.578(2.bin)             |      |
    |       |      32      |       32        |       48       |             67.066(2.bin)68.2(3.bin)68.05(4.bin)             | 67.048(2.bin)68.2(3.bin)68.033(4.bin) |      |
    
    | .json file  | Co_Att_Heads | Image_Att_Heads | Text_Att_Heads |        valid(max)        |
    | :---------: | :----------: | :-------------: | :------------: | :----------------------: |
    |     scr     |      8       |        8        |       12       | 68.49(me) / 68.61(paper) |
    |  _bihead2   |      2       |        -        |       -        |    66.45 **(-2.04)**     |
    |  _bihead32  |      32      |        -        |       -        |      68.24 (-0.25)       |
    |   _vhead2   |      -       |        2        |       -        |     69.79 (**+1.3**)     |
    |  _vhead32   |      -       |       32        |       -        |       68.19 (-0.3)       |
    |   _thead3   |      -       |        -        |       3        |    69.36 (**+0.87**)     |
    |  _thead48   |      -       |        -        |       48       |    67.57 **(-0.92)**     |
    | _bi32v32t48 |      32      |       32        |       48       |       68.2 (-0.29)       |
    
  - **基于标题的图像检索 Caption-Based Image Retrieval**
    
    - dataset：[Flickr30k](https://www.dropbox.com/sh/qqk1xlhkqjyek8q/AAADni5hVBV2PAC8R_13xpIja?dl=0)（31,000 images from Flickr with five captions each）
    - train
      ```
      CUDA_VISIBLE_DEVICES=0 python vilbert_train_tasks.py --bert_model bert-base-uncased --from_pretrained vilbert/pretrain/bert_base_6_layer_6_connect_freeze_0/pytorch_model_8.bin  --config_file vilbert/config/bert_base_6layer_6conect.json  --learning_rate 4e-5 --num_workers 1 --tasks 3
      ```
    - evaluation
      ```
      CUDA_VISIBLE_DEVICES=0 python vilbert_eval_retrieval.py --bert_model bert-base-uncased --from_pretrained vilbert/pretrain/RetrievalFlickr30k_bert_base_6layer_6conect-pretrained/pytorch_model_19.bin --config_file vilbert/config/bert_base_6layer_6conect.json --task 3 --split test --batch_size 1
      ```

>  3 ViLT
>
> 4 CrossViT

# Reference

- [1] [Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762)
- [2] [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (2021)](https://arxiv.org/abs/2010.11929)
- [3] [https://github.com/google-research/vision_transformer](https://github.com/google-research/vision_transformer)
- [4] [Multimodal Learning with Transformers: A Survey (2022)](https://arxiv.org/abs/2206.06488)
- [5] [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (2018)](https://arxiv.org/abs/1810.04805)
- [6] [ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks (2019)](https://arxiv.org/abs/1908.02265)
- [7] [https://github.com/google-research/vision_transformer](https://github.com/google-research/vision_transformer)
- [8] [ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision (2021)](https://arxiv.org/abs/2102.03334)
- [9] [CrossViT: Cross-Attention Multi-Scale Vision Transformer for Image Classification (2021)](https://arxiv.org/abs/2103.14899)

