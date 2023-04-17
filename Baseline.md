[toc]



## LXMERT

论文：[LXMERT: Learning Cross-Modality Encoder Representations from Transformers (EMNLP2019)](https://arxiv.org/abs/1908.07490)

代码：[airsplay/lxmert (github.com)](https://github.com/airsplay/lxmert)

my fork：[xxayt/LXMERT (github.com)](https://github.com/xxayt/LXMERT)

**选择原因**：结构简单（可改进的空间大）；数据集（VQA 2.0）测试在可接受范围

### 1 结构

<img src=".\image\LXMERT结构.png" alt="结构" style="zoom:60%;" />

- object-level image embeddings 图像嵌入层：RoI feature + position feature
- word-level sentence embeddings 文本嵌入层：word embedding + index embedding
- Object-Relationship Encoder：（$N_R$ 层）self-attention + FFN
- Language Encoder：（$N_L$ 层）self-attention + FFN
- Cross-Modality Encoder：（$N_X$ 层）cross-attention + self-attention + FFN

### 2 预训练

<img src=".\image\LXMERT预训练任务.png" alt="预训练任务" style="zoom:60%;" />

- 数据集：
- pre-train tasks：5个
  - 针对文本：Masked Cross-Modality LM
  - 针对图像：Masked Object Prediction via RoI-feature regression / via detected-label classification
  - 针对跨模态：Cross-Modality Matching，Image Question Answering (QA)
- 输入：图像（物体特征）和相关句子（例如，标题或问题）

### 3 微调

#### VQA

- 数据集：VQA 2.0
  - 图像：使用对MS COCO数据集，用faster-rcnn提取目标特征的图像（训练集17G，测试集8G），其中已包含VQA 2.0数据集的图像
  - 文本：VQA 2.0（训练集125M，测试集61M）
  
- 训练：
  - 超参：$4 \text{ epoch},32 \text{ batch size}, 5e-5 \text{lr}$ 
  - 我的显卡：2080Ti，显存11264MiB
  - 读取数据耗时：$>$ **100min**
  - 训练耗时：**80min/epoch**
  - 验证耗时：**约70min**
  
- VQA效果（上传 [EvalAI](https://eval.ai/web/challenges/challenge-page/830/overview) 提交验证，目前只有challenge 2021可以提交）

  |                  | src paper result (challenge 2019) | my result（challenge 2021） |
  | :--------------: | :-------------------------------: | :-------------------------: |
  | Local Validation |               69.9%               |           69.507%           |
  |     Test-Dev     |              72.42%               |           71.83%            |
  |  Test-Standard   |              72.54%               |           72.05%            |

  ```json
  [{"test-dev": {"yes/no": 87.28, "number": 54.13, "other": 62.65, "overall": 71.83}}, 
   {"test-standard": {"yes/no": 87.41, "number": 53.8, "other": 62.86, "overall": 72.05}}]
  ```

  - Leaderboard截图：

  ![VQA test-std result](D:\2Codefield\VS_code\python\GeWuLab\LXMERT\image\VQA test-std result.png)

#### GQA

- 数据集
  - 图像：分别对Visual Genome、MS COCO数据集，用faster-rcnn提取训练和测试的目标特征图像（训练集至少30G），
  - 文本：VQA 2.0（训练集180M，验证集25M，提交测试集590M）
- 训练：
  - 超参：$\text{epoch: }4,\text{batch size: 32},\text{lr: }1e-5$ 
  - 我的显卡：2080Ti，显存11264MiB

#### NLVR



## ViLBERT

论文：[ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks (NeurIPS2019)](https://arxiv.org/abs/1908.02265)

代码：[jiasenlu/vilbert_beta (github.com)](https://github.com/jiasenlu/vilbert_beta)

my fork：[xxayt/ViLBERT (github.com)](https://github.com/xxayt/ViLBERT)

### 1 结构

<img src="D:\2Codefield\VS_code\python\GeWuLab\LXMERT\image\ViLBERT结构.png" alt="ViLBERT结构" style="zoom:50%;" />

- Image embed：RoI feature + position feature
- Sentence embed
- Sentence Transformer：（$L-k$ 层）self-attention + FFN
- Co-Attentional Transformer Layers：（$k$ 层）cross-attention + self-attention + FFN

### 2 预训练

- 数据集：Conceptual Captions（330万对image-caption pairs）
- pre-train tasks：2个
  - masked multi-modal modelling
  - multi-modal alignment prediction
- 输入：图像（来自目标检测网络提取的**边框和视觉特征**）和相关句子（例如，标题或问题）

### 3 微调

#### VQA

- 数据集：VQA 2.0
  - 图像：COCO数据集，用faster-rcnn提取目标特征的图像。（还没下载，但是很大）
  - 文本：针对COCO数据集中图片的110万个问题，每个问题10个答案
- 训练：
  - 原文超参：$\text{epoch: }20,\text{batch size: 256},\text{init lr: }4e-5$ 
  - 还没尝试。。

#### VCR

#### Referring Expressions

#### Caption-Based Image Retrieval



## MCAN

论文：[Deep Modular Co-Attention Networks for Visual Question Answering (CVPR2019)](https://arxiv.org/abs/1906.10770)

代码：[MILVLG/mcan-vqa (github.com)](https://github.com/MILVLG/mcan-vqa)

my fork：[xxayt/MCAN (github.com)](https://github.com/xxayt/MCAN)

### 1 结构

<img src=".\image\MACN结构.png" alt="MACN结构" style="zoom:50%;" />

- 问题和图像的表示（Question and Image Representations）
- 深度协同注意力学习（Deep Co-Attention Learning）
- 多模态融合和分类输出（Multimodal Fusion and Output Classifier）

### 2 训练

#### VQA