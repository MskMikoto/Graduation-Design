import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,),
                 dilation=(1,), if_bias=False, relu=True, same_padding=True, bn=True):
        super(Conv1d, self).__init__()
        p0 = int((kernel_size[0] - 1) / 2) if same_padding else 0   #为了保持序列长度不变，采用 same padding 策略，使卷积前后特征长度一致
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=p0,
                              dilation=dilation, bias=True if if_bias else False)   #卷积在此处学习某一段组合是否重要
        self.bn = nn.BatchNorm1d(out_channels) if bn else None  #标准化，防止梯度爆炸和训练不稳定 x → (x - 均值) / 方差
        self.relu = nn.ReLU(inplace=True) if relu else None     #f(x) = max(0, x)引入非线性变换

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        x = F.dropout(x, 0.2)   #随机丢掉20%的神经元，防止过拟合
        return x

'''
输入 x
 ↓
Conv1d（提取局部特征）
 ↓
BatchNorm（稳定训练）
 ↓
ReLU（非线性）
 ↓
Dropout（防过拟合）
 ↓
输出
'''

class HMCN(nn.Module):      #用多尺度卷积提取局部特征 + 用线性层提取全局特征 + 融合
    def __init__(self, in_channel, out_channel):
        super(HMCN, self).__init__()

        #只做特征变换，不提取上下文，线性变换（通道映射）
        self.conv0 = nn.Sequential(
            Conv1d(in_channel, out_channel, kernel_size=(1,), same_padding=True),
        )

        #捕捉短距离依赖（局部模式）
        self.conv1 = nn.Sequential(
            Conv1d(in_channel, out_channel, kernel_size=(1,), same_padding=True),
            Conv1d(out_channel, out_channel, kernel_size=(3,), same_padding=True),
        )

        #捕捉中距离依赖
        self.conv2 = nn.Sequential(
            Conv1d(in_channel, out_channel, kernel_size=(1,), same_padding=True),
            Conv1d(out_channel, out_channel, kernel_size=(5,), same_padding=True),
            Conv1d(out_channel, out_channel, kernel_size=(5,), same_padding=True),
        )

        #捕捉长距离依赖（全局结构）
        self.conv3 = nn.Sequential(
            Conv1d(in_channel, out_channel, kernel_size=(1,), same_padding=True),
            Conv1d(out_channel, out_channel, kernel_size=(7,), same_padding=True),
            Conv1d(out_channel, out_channel, kernel_size=(7,), same_padding=True),
            Conv1d(out_channel, out_channel, kernel_size=(7,), same_padding=True),
        )
        #把输入直接映射成4个分支拼接后的维度，本质是提取全局特征，不依赖局部卷积
        self.linear = nn.Linear(in_channel, out_channel * 4)

    def forward(self, x):
        x0 = self.conv0(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = torch.cat([x0, x1, x2, x3], dim=1)     #把四个分支的输出拼接起来
        #把不同感受野（不同尺度）的特征整合到一起

        x = x.squeeze(2)    #把最后一维去掉
        x = self.linear(x)  #全局特征建模
        x = x.unsqueeze(2)  #给最后加回一个维度
        x = x + x4          #残差

        return x
        #HMCN 提取出来的“独立特征表示”
