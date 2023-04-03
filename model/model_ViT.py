# Vision Transformer
# 多尺度
from functools import partial
from collections import OrderedDict
import torch
import torch.nn as nn

# 随机深度的drop方法
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


# 采用drop方法的类
class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# 拆分patch + 展平处理
class Patch_Embedding(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_channel=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        # 卷积层kernel_size=stride, 代替划分
        self.proj = nn.Conv2d(in_channel, embed_dim, kernel_size=patch_size, stride=patch_size)
        # 如果norm_layer==None则此层为线性映射nn.Identity()
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        # ViT模型中输入图片的大小是固定的
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # [bs,3,224,224] -> [bs,768,14,14]
        x = self.proj(x)
        # 展平处理 flatten: [B, C, H, W] -> [B, C, HW]
        # [bs,768,14,14] -> [bs,768,196]
        x = x.flatten(2)
        # 调换维度 transpose: [B, C, HW] -> [B, HW, C]
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x


# 多头注意力机制模块
class Attention(nn.Module):
    def __init__(self,
                dim,   # 输入token的维度dim
                num_heads=8,  # head头数
                qkv_bias=False,  # 求qkv时是否加偏置
                qk_scale=None,
                attn_drop_ratio=0.,
                proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads  # 每个head的qkv对应的维度
        self.scale = qk_scale or head_dim ** -0.5  # Attention公式的系数中 \dfr{1}{\sqrt{d}}
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # 使用全连接层得到qkv
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)  # Wo：最后得到结果进行拼接，使用Wo进行映射
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape
        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        qkv = self.qkv(x)
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        # 调整数据维度顺序
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = qkv.permute(2, 0, 3, 1, 4)
        # 通过切片拆分成qkv
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        '''
        Attention公式: \text{softmax}(\dfrac{QK^T}{\sqrt{d_k}})V
        '''
        # 后两个维度进行调换
        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # 矩阵乘法@: 只对后两个维度进行矩阵乘法
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # dim=-1: 针对每行进行softmax操作
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        x = (attn @ v)
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        x = x.transpose(1, 2)
        # 拼接多头中最后两个维度的信息
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = x.reshape(B, N, C)
        # 全连接层进行映射
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# MLP Block
class MLP_Block(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.FC1 = nn.Linear(in_features, hidden_features)
        # GELU激活函数
        self.act = act_layer()
        self.FC2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.FC1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.FC2(x)
        x = self.drop(x)
        return x


# Encoder Block
class Encoder_Block(nn.Module):
    def __init__(self,
                dim,
                num_heads,
                mlp_ratio=4.,  # MLP隐藏层输出节点个数相对输入节点个数的倍数
                qkv_bias=False,
                qk_scale=None,
                drop_ratio=0.,
                attn_drop_ratio=0.,
                drop_path_ratio=0.,
                act_layer=nn.GELU,
                norm_layer=nn.LayerNorm):
        super(Encoder_Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = MLP_Block(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

# ViT汇总
class VisionTransformer(nn.Module):
    def __init__(self, img_size=224,
                patch_size=16, 
                in_channel=3, 
                num_classes=1000,
                embed_dim=768, 
                depth=12, 
                num_heads=12, 
                mlp_ratio=4.0, 
                qkv_bias=True,
                qk_scale=None, 
                representation_size=None, 
                drop_ratio=0.,
                attn_drop_ratio=0., 
                drop_path_ratio=0., 
                embed_layer=Patch_Embedding, 
                norm_layer=None,
                act_layer=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_channel (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer (Encoder_Block重复次数)
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)  # LayerNorm: 对每单个batch进行的归一化
        act_layer = act_layer or nn.GELU
        # 拆分成patch
        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_channel=in_channel, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        # 创建可训练的class_token参数
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # 第一个1代表batch_size维度
        # 创建可训练的position_embedding参数
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)
        # Encoder Block中drop_out/drop_path方法使用的drop参数是递增的
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        # 创建Encoder_Block列表
        self.Encoder_Block_List = nn.Sequential(*[
            Encoder_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                        norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        # Representation layer: 最后分类前可选是否加入pre-logits
        if representation_size:
            self.has_logits = True
            self.num_features = representation_size
            # 有序字典
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()
        # Classifier FC_last(s): 最后分类的全连接层
        self.FC_last = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        # Weight init: 权重初始化
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(_init_vit_weights)

    def forward(self, x):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        # [B,3,224,224] -> [B,196,768]
        x = self.patch_embed(x)  # [B, 196, 768]
        # [1,1,768] -> [B,1,768]: 在batch_size维度扩展至batch_size份
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        # 加上类别token
        # [B,196,768] -> [B,197,768]
        x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        # 加上位置编码
        # [B,197,768] -> [B,197,768]
        x = self.pos_drop(x + self.pos_embed)
        # [B,196,768] -> [B,197,768]
        x = self.Encoder_Block_List(x)
        x = self.norm(x)
        # 提取第二维第一列的cls_token
        # [B,197,768] -> [B,1,768]
        x = x[:, 0]
        # 可选的pre_logits
        # [B,1,768] -> [B,1,representation_size]
        x = self.pre_logits(x)
        # [B,1,representation_size] -> [B,1,num_classes]
        x = self.FC_last(x)
        return x


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


# ViT-B/16
def vit_base_patch16_224_ImageNet1k(num_classes: int = 1000):
    """
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1zqb08naP0RPqqfSXfkB2EA  密码: eu9f
    """
    model = VisionTransformer(img_size=224,
                                patch_size=16,
                                embed_dim=768,
                                depth=12,
                                num_heads=12,
                                representation_size=None,
                                num_classes=num_classes)
    return model


# ViT-B/16
def vit_base_patch16_224_ImageNet21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth
    """
    model = VisionTransformer(img_size=224,
                                patch_size=16,
                                embed_dim=768,
                                depth=12,
                                num_heads=12,
                                representation_size=768 if has_logits else None,
                                num_classes=num_classes)
    return model


# ViT-B/32
def vit_base_patch32_224_ImageNet1k(num_classes: int = 1000):
    """
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1hCv0U8pQomwAtHBYc4hmZg  密码: s5hl
    """
    model = VisionTransformer(img_size=224,
                                patch_size=32,
                                embed_dim=768,
                                depth=12,
                                num_heads=12,
                                representation_size=None,
                                num_classes=num_classes)
    return model


# ViT-B/32
def vit_base_patch32_224_ImageNet21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch32_224_in21k-8db57226.pth
    """
    model = VisionTransformer(img_size=224,
                                patch_size=32,
                                embed_dim=768,
                                depth=12,
                                num_heads=12,
                                representation_size=768 if has_logits else None,
                                num_classes=num_classes)
    return model


# ViT-L/16
def vit_large_patch16_224_ImageNet1k(num_classes: int = 1000):
    """
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1cxBgZJJ6qUWPSBNcE4TdRQ  密码: qqt8
    """
    model = VisionTransformer(img_size=224,
                                patch_size=16,
                                embed_dim=1024,
                                depth=24,
                                num_heads=16,
                                representation_size=None,
                                num_classes=num_classes)
    return model


# ViT-L/16: 太大了，权重文件1.2G
def vit_large_patch16_224_ImageNet21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch16_224_in21k-606da67d.pth
    """
    model = VisionTransformer(img_size=224,
                                patch_size=16,
                                embed_dim=1024,
                                depth=24,
                                num_heads=16,
                                representation_size=1024 if has_logits else None,
                                num_classes=num_classes)
    return model


# ViT-L/32
def vit_large_patch32_224_ImageNet21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_224_in21k-9046d2e7.pth
    """
    model = VisionTransformer(img_size=224,
                                patch_size=32,
                                embed_dim=1024,
                                depth=24,
                                num_heads=16,
                                representation_size=1024 if has_logits else None,
                                num_classes=num_classes)
    return model


# ViT-H/14
def vit_huge_patch14_224_ImageNet21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: converted weights not currently available, too large for github release hosting.
    """
    model = VisionTransformer(img_size=224,
                                patch_size=14,
                                embed_dim=1280,
                                depth=32,
                                num_heads=16,
                                representation_size=1280 if has_logits else None,
                                num_classes=num_classes)
    return model