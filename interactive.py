import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, feature_dim, num_heads):
        super(MultiHeadCrossAttention, self).__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.attention_head_size = int(feature_dim / num_heads)
        self.all_head_size = self.num_heads * self.attention_head_size
        self.query_weights = nn.Linear(feature_dim, self.all_head_size)
        self.key_weights = nn.Linear(feature_dim, self.all_head_size)
        self.value_weights = nn.Linear(feature_dim, self.all_head_size)
        self.out = nn.Linear(feature_dim, feature_dim)

    '''
    输入 (512维)
       ↓
    Linear → Q
    Linear → K
    Linear → V
       ↓
    拆成 4个头（每个128维）
       ↓
    Attention计算
       ↓
    拼接
       ↓
    Linear（self.out）
       ↓
    输出
    '''

    def transpose_for_scores(self, x):
        assert self.feature_dim % self.num_heads == 0  #必须能整除，否则无法平均分给每个head
        self.attention_head_size = self.feature_dim // self.num_heads
        new_x_shape = x.size()[:-1] + (self.num_heads, self.attention_head_size)    #取“除了最后一维之外的所有维度”new_x_shape = (32,) + (4, 128)
        x = x.view(new_x_shape)
        x = x.unsqueeze(2)  #在指定位置插入一个“长度为1的维度”用于调整张量形状，满足矩阵乘法
        x = x.permute(0, 1, 2, 3)   #permute 的作用改变维度顺序
        return x

    def forward(self, query, key, value):
        query_layer = self.transpose_for_scores(self.query_weights(query))
        key_layer = self.transpose_for_scores(self.key_weights(key))
        value_layer = self.transpose_for_scores(self.value_weights(value))
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))   #把 Key 的最后两个维度交换（转置）
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = F.softmax(attention_scores, dim=-1)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        attention_output = self.out(context_layer)
        return attention_output

'''模型首先通过线性变换生成 Query、Key 和 Value，然后通过计算 Q 和 K 的点积得到注意力分数，
并进行缩放和 softmax 得到权重，
最后对 Value 进行加权求和，从而得到融合后的特征表示。'''


class FeedForward(nn.Module):   #对交互结果进行“特征增强”
    def __init__(self, embed_size, forward_expansion):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_size, forward_expansion * embed_size)
        self.fc2 = nn.Linear(forward_expansion * embed_size, embed_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
'''FeedForward 是 Transformer 中的前馈网络模块，通过两层全连接层和非线性激活函数，对注意力输出进行进一步特征变换，从而提升模型的表达能力。'''


class CrossTransformerEncoder(nn.Module):
    def __init__(self, protein_feature_dim, rna_feature_dim, num_heads=4):
        super(CrossTransformerEncoder, self).__init__()
        self.protein_to_rna_fc = nn.Linear(protein_feature_dim, 512)
        self.rna_to_protein_fc = nn.Linear(rna_feature_dim, 512)    #特征对齐统一维度
        self.multihead_crossattention = MultiHeadCrossAttention(512, num_heads)
        self.pro_norm1 = nn.LayerNorm(512)
        self.rna_norm1 = nn.LayerNorm(512)
        self.pro_feed_forward = FeedForward(512, 4)
        self.rna_feed_forward = FeedForward(512, 4)
        self.pro_norm2 = nn.LayerNorm(512)
        self.rna_norm2 = nn.LayerNorm(512)
        self.dropout = nn.Dropout(0.2)  #随机丢掉一部分神经元（防止过拟合）

#Linear = 对齐维度
#Attention = 学交互关系
#FFN = 强化特征
#LayerNorm = 稳定训练
#Dropout = 防过拟合

#Attention 只做加权FFN 提升表达能力

    def forward(self, protein_features, rna_features):

        #特征对齐
        protein_features_aligned = F.relu(self.protein_to_rna_fc(protein_features))
        rna_features_aligned = F.relu(self.rna_to_protein_fc(rna_features))
        #Protein → Linear → ReLU → 512维
        #RNA → Linear → ReLU → 512维

        # shared_weights_cross_multihead_attention
        protein_attention_output = self.multihead_crossattention(
            query=protein_features_aligned,
            key=rna_features_aligned,
            value=rna_features_aligned
        )
        rna_attention_output = self.multihead_crossattention(
            query=rna_features_aligned,
            key=protein_features_aligned,
            value=protein_features_aligned
        )

        # add & norm
        protein_features_aligned = protein_features_aligned.unsqueeze(1)
        rna_features_aligned = rna_features_aligned.unsqueeze(1)    #[batch, 512] → [batch, 1, 512]
        pro_feature = self.dropout(self.pro_norm1(protein_attention_output + protein_features_aligned))
        rna_feature = self.dropout(self.rna_norm1(rna_attention_output + rna_features_aligned)) #输出 = Attention结果 + 原始输入
        pro_feature = pro_feature.squeeze(1)
        rna_feature = rna_feature.squeeze(1)    #再降回原维度

        # feed forward
        pro_feature_forward = self.pro_feed_forward(pro_feature)
        rna_feature_forward = self.rna_feed_forward(rna_feature)

        # add & norm
        pro_feature = self.dropout(self.pro_norm2(pro_feature_forward + pro_feature))
        rna_feature = self.dropout(self.rna_norm2(rna_feature_forward + rna_feature))

        return pro_feature, rna_feature

        #输出增强后的Protein和rna特征
'''
输入:
Protein: [batch, d1]
RNA:     [batch, d2]

↓ Linear

[batch, 512]

↓ Attention

[batch, 1, 512]

↓ squeeze

[batch, 512]

↓ FFN

[batch, 512]

输出:
[batch, 512]
'''