from interactive import *
from independent import *


class DNN(nn.Module):
    def __init__(self, inputs_dim, hidden_units):
        super(DNN, self).__init__()
        self.inputs_dim = inputs_dim
        self.hidden_units = hidden_units
        self.dropout = nn.Dropout(0.5)

        self.hidden_units = [inputs_dim] + list(self.hidden_units)
        self.linear = nn.ModuleList([
            nn.Linear(self.hidden_units[i], self.hidden_units[i + 1]) for i in range(len(self.hidden_units) - 1)
        ])

        self.activation = nn.Softmax()

    def forward(self, X):
        inputs = X
        for i in range(len(self.linear)):
            fc = self.linear[i](inputs)
            gate_fc = self.activation(fc)
            fc = fc * gate_fc   #输出特征 = 原始特征 × 重要性权重原本大的值 → 仍然保留较多 原本小的值 → 被进一步压缩
            fc = self.dropout(fc)
            inputs = fc

        return inputs


class DBENet_NPI(nn.Module):
    def __init__(self, protein_input_dim, rna_input_dim, protein_out_dim=128, rna_out_dim=128,
                 dnn_hidden_units=(512, 256, 128)):
        super(DBENet_NPI, self).__init__()
        self.protein_input_dim = protein_input_dim
        self.rna_input_dim = rna_input_dim
        self.interactive_attention1 = CrossTransformerEncoder(protein_input_dim, rna_input_dim)
        self.interactive_attention2 = CrossTransformerEncoder(512, 512)
        self.interactive_attention3 = CrossTransformerEncoder(512, 512)
        self.interactive_attention4 = CrossTransformerEncoder(512, 512)
        self.hmcn_pro = HMCN(protein_input_dim, protein_out_dim)
        self.hmcn_rna = HMCN(rna_input_dim, rna_out_dim)
        self.linear = nn.Linear(2048, 1024)
        self.dnn = DNN(protein_out_dim * 4 + rna_out_dim * 4, dnn_hidden_units)
        self.residual_connection = nn.Linear(protein_out_dim * 4 + rna_out_dim * 4, dnn_hidden_units[-1])
        self.linear_out = nn.Linear(dnn_hidden_units[-1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        protein_feature, rna_feature = x[:, :self.protein_input_dim], x[:, self.protein_input_dim:]

        # interactive feature encoder (MCANet)
        interactive_pro1, interactive_rna1 = self.interactive_attention1(protein_feature, rna_feature)
        interactive_pro2, interactive_rna2 = self.interactive_attention2(interactive_pro1, interactive_rna1)
        interactive_pro3, interactive_rna3 = self.interactive_attention3(interactive_pro2, interactive_rna2)
        interactive_pro4, interactive_rna4 = self.interactive_attention4(interactive_pro3, interactive_rna3)
        #四层交互编码

        interactive_feature = torch.cat([interactive_rna4, interactive_pro4], dim=1)
        #沿着特征维拼接，形成一个完整的“交互特征向量”

        # independent feature encoder (HMCNet)
        #protein自身的局部模式，RNA自身的结构/性质特征
        independent_pro = protein_feature.unsqueeze(2)      #给protein特征增加一个维度
        independent_rna = rna_feature.unsqueeze(2)          #给RNA特征增加一个维度
        independent_pro = self.hmcn_pro(independent_pro)    #独立特征提取
        independent_rna = self.hmcn_rna(independent_rna)
        independent_pro = independent_pro.squeeze(2)        #CNN提完特征后，把结果恢复成二维向量
        independent_rna = independent_rna.squeeze(2)
        independent_feature = torch.cat([independent_rna, independent_pro], dim=1)
        #形成完整的独立特征向量

        # concatenate
        all_features = torch.cat([interactive_feature, independent_feature], dim=1)
        #将交互特征和独立特征拼接起来

        # ncRPI prediction module
        concatenated = self.linear(all_features)    #对融合特征做线性映射
        dnn_output = self.dnn(concatenated)
        #Linear + Softmax门控 + Dropout
        #高层非线性特征融合 + 重要特征筛选
        #从 1024 维融合特征里，进一步筛出最适合做 interaction prediction 的高层表示

        residual_output = self.residual_connection(concatenated)
        #残差

        final_input = torch.add(dnn_output, residual_output)
        #DNN主路 + 残差支路

        prediction = self.linear_out(final_input)
        #最终线性输出层，logit（未归一化分数）

        return self.sigmoid(prediction)
