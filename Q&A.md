[toc]

# Q&A of Transformer、Vision Transformer and Multimodal Trandformer

- **Reference**：
  - Vanilla Transformer：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)
  - Vision Transformer (ViT)：[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
  - Multimodal Trandformer：[Multimodal Learning with Transformers: A Survey](https://arxiv.org/abs/2206.06488)



---



# 1 Vanilla Transformer

## 1.1 Introduction

- Transformer：一种新的网络架构，完全基于注意力机制，去除了RNN和卷积。可描述input和output之间的**全局依赖关系**。
- RNN的特点(缺点)：无法并行，通常沿着input和output序列的位置计算

## 1.2 Model Architecture

<img src=".\image\transformer.png" alt="transformer" style="zoom:50%;" />

- **Encoder-Decoder**：$x$ 序列 $\stackrel{encoder}{\longrightarrow}$ $z$ 序列，$z$ 序列 $\stackrel{decoder}{\longrightarrow}$ $y$ 序列。需要以自回归（解码时一个个生成）的形式生成output

- **Encoder**：
  $$
  Z_1\leftarrow LN(MHA(Z_0)+Z_0)\\
  Z_2\leftarrow LN(FFN(Z_1)+Z_1)
  $$

  - 其中 $FFN(Z_1),Z_1,MHA(Z_0),Z_0\in R^{bz\times d_{model}=bz\times 512}$

- **使用 $\text{Layer Norm}$ 而不使用 $\text{Batch Norm}$ 的原因**：

  - $\text{Batch Norm}$：在训练时，对每个feature上的整个mini_batch序列进行归一化，并学到lambda和beta；在测试时，对每个feature上的全部样本进行归一化。
  - $\text{Layer Norm}$：对每个样本上的整个feature进行归一化，即针对某个样本内部计算均值和方差。
  - 原因：在CV领域，由于channel维度信息很重要，因此使用BN对每个feature维度归一化可以减少不同feature (channel)信息的损失。在NLP领域，一般序列长度不一致，且各样本的信息关联性不大，因此使用LN无需考虑样本间的依赖，也不需要计算全部样本的均值方差。

- **Decoder**：
  $$
  Z_1\leftarrow LN(Masked\_MHA(Z_0)+Z_0)\\
  Z_2\leftarrow LN(MHA(Z_1)+Z_1)\\
  Z_3\leftarrow LN(FFN(Z_2)+Z_2)
  $$

### 1.2.1 Attention

<img src=".\image\attention.png" alt="attention" style="zoom:50%;" />
$$
\text{Attention}(Q,K,V)=\text{softmax}(\dfrac{QK^T}{\sqrt{d_k}})V
$$

- **参数维度**：由于 $Q\in R^{N\times d_k}$，$K\in R^{M\times d_k}$，$V\in R^{M\times d_v}$，于是 $\text{Attention}(Q,K,V)\in R^{N\times d_k}$。在 $Encoder$的 $MHA$ 中 $N=M$
- **使用 $dot-product$ 注意力而不使用加法注意力(additive attention)的原因**：通过使用优化的矩阵乘法，更快更省空间。

- **需要 $Q$ 和 $K$ 做点积的原因**：得到的 $\text{softmax}(\dfrac{QK^T}{\sqrt{d_k}})$ 称为 $\text{attention score}$，视为对 $V$ 进行加权和的权重（来对 $V$ 进行提纯）。
- **$Q$ 和 $K$ 使用不同权重矩阵 $W_i^Q,W_i^K$ 的原因**：在不同空间上的投影，增加表达能力，使计算得到的 $\text{attention score}$ 矩阵的泛化能力更高。若拿 $K\cdot K^T$ 进行点积，得到的对称矩阵是投影到了同一个空间，所以泛化能力变差。
- **$\dfrac{1}{\sqrt{d_k}}$ 的原因**：由于 $Q,K$ 中的每个样本 $q,k$ 均已进行 $\text{Layer Norm}$（均值为0，方差为1），则对于 $q\cdot k=\sum\limits_{i=1}^{d_k}q_ik_i$
  - **缩放的原因**：若 $d_k$ 较大，则 $q\cdot k$ 的值会变大，使 $\text{softmax}$ 函数趋向于 $0$ 和 $1$，此时计算的梯度将会极小，无法继续训练。为了抵消影响，对点积的值进行缩放。
  - **缩放参数为 $\dfrac{1}{\sqrt{d_k}}$ 的原因**：通过计算， $q\cdot k$ 的均值为 $0$，方差为 $d_k$。为了使 $\operatorname{softmax}$ 更加平滑，需要控制方差为 $1$，正好需要缩放参数为 $\dfrac{1}{\sqrt{d_k}}$。


### 1.2.2 Multi-Head Self-Attention (MHSA)

<img src=".\image\multi-head attention.png" alt="multi-head attention" style="zoom:50%;" />
$$
\text{MultiHead}(Q, K, V ) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O\\
\text{where head}_i = \text{Attention}(QW^Q_i, KW^K_i, VW^V_i )
$$


- **参数维度**：由于 $Q\in R^{N\times d_{model}}$，$K\in R^{M\times d_{model}}$，$V\in R^{M\times d_{model}}$；

  而 $W^Q_i\in R^{d_{model}\times d_k}$，$W^K_i\in R^{d_{model}\times d_k}$，$W^V_i\in R^{d_{model}\times d_v}$，$W^O\in R^{hd_v\times d_{model}}$

  于是 $\text{head}_i=\text{Attention}(Q,K,V)\in R^{N\times d_v}$，$\text{Concat}\in R^{N\times hd_v}$，$\text{MultiHead}(Q,K,V)\in R^{N\times d_{model}}$。

- **使用多头注意力机制的原因**：$dot-product$注意力没有什么可学习参数，而投影到低维（$d_k=d_v=d_{model}/h=512/8=64$）,投影时的 $W,b$ 均是可学习参数。此外，每次投影可以学到不同子空间的信息，可捕捉多种特征信息。

- **Encoder中的MHA**：这里MHA中的 $query=key=value$ 均来自前一层的输出，即称为self-attention自注意机制。Encoder中的每个位置都可以处理Encoder前一层中的所有(all)位置。

- **Decoder中的Masked MHA1**：这里的Masked MHA1中的 $query=key=value$ ，即也为self-attention。
  $$
  \text{MSA}(Q,K,V)=\text{softmax}(\dfrac{QK^T}{\sqrt{d_k}}\odot M)V
  $$

  - **使用Mask的作用**：为防止Decoder中的信息在 $t$ 时刻看到 $t$ 时刻之后的输入，保证训练和预测时行为一致，并且保持自回归特性（解码时一个一个生成，无法并行化）。通过屏蔽（-1000）$\text{softmax}$ 中对应连接后方的值来实现。

- **Decoder中的MHA2**：这里MHA中的 $query$ 来自前面的Decoder，而 $key$ 和 $value$ 来自Encoder的输出，不是自注意力机制。这使得Decoder中的每个位置都可以覆盖输入序列中的所有(all)位置。这模仿了Seq2Seq模型中典型的编码器-解码器注意机制

### 1.2.3 Feed-Forward Networks (FFN)

$$
FFN(x)=\max(0,xW_1+b)W_2+b
$$

- **参数维度**：$W_1:[bz,512]\rightarrow[bz,2048]$，$W_1:[bz,2048]\rightarrow[bz,512]$
- **FFN的作用**：为了语义的转换，先将每个位置的特征向量映射到高维空间，在映射回原始维度。
- **“扩张-压缩”结构的原因**：
  1. 增强特征表达能力：MLP可在高维空间中对特征向量进行更加复杂的非线性变换，从而增强特征的表征能力
  2. 提高模型训练速度：通过压缩，可使模型参数更加紧凑，从而提高模型训练和推理的效率。

### 1.2.4 Pos Embedding

$$
Z\leftarrow X\oplus \text{Pos Embedding}
$$

- 为了利用时序信息，加入绝对位置信息pos embedding（这里通过加法操作）



---



# 2 Vision Transformer (ViT)

## 2.1 Introduction

- ViT：一种应用于图像patch序列的只用transformer的网络架构，可很好地执行图像分类。
- 主流使用方法：pre-train，在大型数据集上进行预训练；fine-turn，在较小的特定任务数据集上进行微调。
- **相同较小数据集下，ViT比ResNet差的原因**：
  - 缺少作为先验信息的归纳偏置(inductive biases)，CNN中两大归纳偏置为locality（靠近的物体相关性更强）和translation equirance（$f\cdot g = g\cdot f$）。
  - 因此ViT需在大规模数据集下进行预训练，来提升效果。实验表明针对大数据集的效果提升未达上限。

## 2.2 ViT Method

<img src=".\image\vit.png" alt="vit" style="zoom:50%;" />
$$
\begin{aligned}
Z_0& = [X_{class}; X^1_pE; X^2_pE;... ; X^N_pE] + E_{pos} \\
Z_l'& = MSA(LN(Z_{l−1})) + Z_{l−1} \\
Z_l& = MLP(LN(Z_l')) + Z_l' \\
y& = LN(Z^0_L)
\end{aligned}
$$

- **参数维度**：图像 $H=W=224,C=3$，**patch**大小 $P=16(/14/32)$，patch个数 $N=HW/P^2=196$

  $X_p^i\in R^{N\times (P^2\cdot C)=196\times 768}$，$E\in R^{768\times 768}:[bz,196,768]\rightarrow [bz,196,768]$，$X_{class}\in R^{1\times 768}$，$E_{pos}\in R^{197\times 768}$

  $Z_i\in R^{197\times 768}$ 

- **使用步骤**：将图像分割为多个patches，将这些patches的线性投影序列加入cls token和pos embedding后，作为Transformer Encoder的输入。最后取cls token的特征进行分类。训练时以监督学习的方式进行图像分类。

### 2.2.1 Embedding

- **将image分割为多个patch的原因**：传统CNN中，可有效捕捉图像的局部特征。但在Transformer中每个token都关注的是全局信息，因此需要将全局分割成更小的部分，以便Transformer可以更好地处理局部信息。此外Vanilla Transformer适用于NLP，其输入token为2维（不算batchsize），而传统RGB图像输入为3维，因此需要将输入变为2维，故在划分为patch后实现flatten操作进行降维。
- **class token的作用**：class token经过Transformer Encoder后，通过全局注意力，学到所有patch中的权重。然后将代表所有token信息的class token作为整个Transformer的输出特征，放入分类器MLP Head。
  - 不使用class token：类似ResNet的方法，将patch token在Transformer Encoder的输出作为图像特征放入全局平均池化(GAP)，通过调整learning rate，效果类似。
- **position embedding**：可视作一种额外的输入特征。
  - 作用：为了对seq中不同位置的信息进行区分和处理，从而更好地捕捉序列中的局部结构。具体来说，用于为input中的每个位置提供一些时间或空间信息，并将其映射到一个隐式的特征空间坐标上。
  - 方法：一维pos embedding，二维pos embedding，相对pos embedding

### 2.2.2 Transformer Encoder

> 类似[Vanilla Transformer中的Encoder](###1.2.2 Multi-Head Self-Attention (MHSA))

### 2.2.3 MLP Head

- **分类头将class token的特征作为输入**：在pre-train时，由一个带hidden layer的MLP实现。在fine-turn时，由一个线性层实现。

## 2.3 Fine-tuning

- 最后的分类器可直接替换为输出num_class维的线性层
- **图像尺寸与pre-train时不同时**：由于pre-train的pos-embedding无法与尺寸对应，因此其无法直接使用。可通过二维插值实现。

## 2.4 Comparision

- ViT和ResNet比较：大规模数据集训练时（ImageNet21K及更大数据集），ViT效果更好



---



# 3 MultiModel Transformer

> 本次调查的主要内容包括：
> (1)多模态学习、Transformer生态系统和多模态大数据时代的背景；
> (2)从几何拓扑角度对Vanilla Transformer、Vision Transformer和多模态Transformer进行理论回顾；
> (3)通过多模态预训练和特定多模态任务这两个重要范式回顾了多模态Transformer的应用；
> (4)对多模态transformer模型和应用所面临的共同挑战和共同设计的总结；
> (5)对社区的开放问题和潜在研究方向的讨论。

## 3.1 Multi-Head Attention Mechanism

- Transformer做多模态学习的优点：
  - self-attention的[输入模式](###3.2.1 Tokenized Input)，可简化对跨模态相关性的学习。
  - 对不同模态和任务建模时的可伸缩性。。。模型具有良好的泛化能力，不太容易收到特定模态中数据偏差（归纳偏置）的影响。
  - 几何拓扑角度：self-attention可将任何模态的输入建模为全连接图（读取全局信息），将每个输入token视为图中的节点；可罗选择建模空间。

## 3.2 Transformer understand（理论回顾）

> 从几何拓扑角度

### 3.2.1 Tokenized Input

- **Tokenized Input的作用**：通过标记化序列(tokenized sequences) 实现**可将任何模态输入建模为全连通图**。
- **Tokenized Input的优势**：
  - 简化跨模态引起的模态壁垒
  - 通过concat,stack和weighted sum等多种方式处理输入信息
  - 与特定token（例：[class], [MASK]）兼容
  - 帮助attention处理多模态数据。

### 3.2.2 Self-Attention

- **Self-Attention的作用**：允许使给定输入序列的每个元素关注所有其他元素，因此self-attention可视为将输入编码为一个全连通图。
- **Multi-Head Self-Attention的作用**：相当于一种整体思想，帮助模型共同处理来自多个表征空间的信息。
- **Masked SA的作用**：相当于向Transformer模型加入对应领域的先验知识。

## 3.3 Transformer in MultiModel task

### 3.3.1 MultiModel Input

- 此处的融合处理可视为一种**early fusion**
- 在输入层面：**将embedding后的token视为节点，将整个input视为全连接图**。这种通用的结构使得Transformer可在不同模态下有效。
- Special Token：
- 处理任意模态的输入，只需执行两个主要步骤：
  (1)令牌化(tokenize)输入；
  (2)选择一个embedding space来表示tokens，然后将数据输入Transformer。

### 3.3.2 Self-Attention Variants

- 



<img src=".\image\multimodel self-attention.png" alt="multimodel self-attention" style="zoom:40%;" />

1. Early Sum(token-wise, weighted)：
2. Early Concat；
3. Hierarchical Attention (multi-stream to one-stream)；
4. Hierarchical Attention (one-stream to multi-stream)；
5. Cross-Attention；
6. Cross-Attention to Concat

---

- 将注意力机制使用于多模态研究，有什么优势

将注意力机制用于多模态研究可以带来多个优势：

1. 提高模型**性能**：多模态数据通常包含不同类型的信息，例如图像和文本，它们之间的关系可能比单模态数据更为复杂。使用注意力机制可以使模型更好地捕捉这些关系，从而提高模型性能。
2. 增加模型**解释性**：在多模态数据中，不同模态之间的关系可能是非常重要的。注意力机制可以帮助我们定位模型在不同模态之间所关注的区域或特征，从而提高模型的解释性。
3. 提高模型**泛化能力**：在多模态研究中，模型的泛化能力往往比单模态数据更为关键，因为模型必须能够在新的多模态数据上进行良好的推广。注意力机制可以帮助模型更好地适应新的多模态数据，从而提高泛化能力。
4. **应用范围广泛**：注意力机制在各种任务中都可以应用，例如图像标注、视频分类、自然语言处理等，因此可以被广泛地应用于多模态研究中。

- 可以详细说说吗

当我们将注意力机制应用于多模态研究时，通常需要设计一个多模态注意力模型，该模型可以同时处理多个输入模态，并且能够对不同模态之间的关系进行建模。下面是一个可能的多模态注意力模型架构：

1. 模态嵌入层：对于每个输入模态，我们可以使用不同的模态嵌入器来将数据转换为低维向量表示，例如使用卷积神经网络（CNN）对图像进行特征提取，使用循环神经网络（RNN）对文本进行特征提取。
2. 多模态注意力层：该层的输入包括多个模态的嵌入向量和一个上下文向量。该层通过学习注意力权重来控制每个输入模态对上下文向量的贡献程度。例如，对于一张图片和一段文字描述，多模态注意力层可以学习哪些图片区域和文本单词与上下文相关，以及它们的相关性大小。
3. 多模态融合层：该层的输入包括每个输入模态的加权向量，该向量是上下文向量和注意力权重的乘积。多模态融合层将所有输入模态的加权向量进行加和或拼接，以生成一个表示多模态信息的向量。
4. 输出层：根据具体任务的需求，输出层可以是一个分类器、回归器或者一个生成模型。该层的输入是多模态融合层输出的向量，以及其他任务相关的元数据。

在多模态研究中，注意力机制可以使模型更加灵活和有效地利用不同模态之间的关系，从而提高模型性能、泛化能力和解释性。例如，在图像标注任务中，我们可以使用注意力机制来确定图像中的重要区域，并将它们与文本描述相匹配，从而生成更准确和具有解释性的图像标注结果。在视频分类任务中，我们可以使用注意力机制来确定哪些帧是重要的，并将它们作为视频的特征表示，从而提高分类性能。总之，注意力机制在多模态研究中具有广泛的应用前景。

- 那从特定模态的数据偏差角度呢

从特定模态的数据偏差角度看，将注意力机制应用于多模态研究可以带来以下优势：

1. 弥补特定模态的数据偏差：在多模态数据中，某些模态可能比其他模态更容易受到数据偏差的影响，例如图像可能会受到拍摄角度、光照等因素的影响，文本可能会受到语言习惯、文化背景等因素的影响。通过使用注意力机制，我们可以对每个模态的特征进行加权，使模型更关注那些对任务更有用的特征，从而减少特定模态的数据偏差对模型的影响。
2. 提高对少数类别的关注度：在一些多模态任务中，少数类别可能比多数类别更为重要，例如医疗图像中的异常病灶、安全监控中的异常事件等。使用注意力机制可以帮助模型更加关注这些少数类别所对应的模态特征，从而提高模型对这些重要类别的识别能力。
3. 提高模型鲁棒性：多模态数据的丰富性和复杂性可能导致模型对噪声和干扰非常敏感。通过使用注意力机制，我们可以让模型更关注那些对任务最为重要的特征，从而减少对噪声和干扰的敏感度，提高模型的鲁棒性。

总之，将注意力机制应用于多模态研究可以帮助我们更好地利用多模态数据中的信息，提高模型的性能、泛化能力和鲁棒性，同时减少特定模态数据偏差对模型的影响。
