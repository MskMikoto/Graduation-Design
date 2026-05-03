import os
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler
from utils.sequence_encoder import ProEncoder, RNAEncoder

'''
Choose DATA_SET = ['RPI488', 'Example', 'RPI1847', 'RPI7317', 'NPInter']
'''
DATA_SET = 'Example'
script_dir, script_name = os.path.split(os.path.abspath(sys.argv[0]))
parent_dir = os.path.dirname(script_dir)
DATA_BASE_PATH = parent_dir + '/'
BASE_PATH = parent_dir + '/' + DATA_SET + '/'


def read_data_seq(path):
    """
    读取序列文件（.fa）
    :param path: 序列文件路径
    :return: 返回序列字典,键为序列名称,值为与其对应的字符串, {'name': 'seq',...}
    """
    seq_dict = {}
    with open(path, 'r') as f:
        name = ''
        for line in f:
            line = line.strip()
            if line[0] == '>':
                name = line[1:]
                seq_dict[name] = ''
            else:
                if line.startswith('XXX'):
                    seq_dict.pop(name)
                else:
                    seq_dict[name] = line
    return seq_dict


def load_data(data_set):
    """
    从指定的数据集中加载数据,返回蛋白质序列信息、RNA序列信息
    """
    pro_seqs = read_data_seq(DATA_BASE_PATH + data_set + '/' + data_set + '_protein_seq_cleaned.fa')
    rna_seqs = read_data_seq(DATA_BASE_PATH + data_set + '/' + data_set + '_rna_seq_cleaned.fa')

    return pro_seqs, rna_seqs


def encoding_data(pro_seqs, rna_seqs, PE, RE):
    """
    编码蛋白质和RNA序列信息
    """
    pro_only_seqs_encoded = {}  # 存储仅编码蛋白质序列的结果
    rna_only_seqs_encoded = {}  # 存储仅编码RNA序列的结果
    pro_seqs_structs_encoded = {}  # 存储编码蛋白质序列及结构的结果。
    rna_seqs_structs_encoded = {}  # 存储编码RNA序列及结构的结果。
    for pro_name, pro_seq in pro_seqs.items():
        pro_only_seqs_encoded[pro_name] = PE.encode_conjoint(pro_seq)
    for rna_name, rna_seq in rna_seqs.items():
        rna_only_seqs_encoded[rna_name] = RE.encode_conjoint(rna_seq)

    return pro_only_seqs_encoded, rna_only_seqs_encoded


def standardization(data):
    """
    对数据进行标准化处理。
    """
    scaler = StandardScaler()
    scaler.fit(data)
    data = scaler.transform(data)
    return data


# 加载蛋白质序列信息、RNA序列信息
pro_seqs, rna_seqs = load_data(DATA_SET)

# PE: 蛋白质编码对象; RE: RNA编码对象
PE = ProEncoder(3, 3, True)
RE = RNAEncoder(4, 4, True)

# 编码蛋白质和RNA的信息
print("encoding protein data and rna data.\n")
pro_only_seqs_encoded, rna_only_seqs_encoded = encoding_data(pro_seqs, rna_seqs, PE, RE)

# 提取字典中的名称和编码向量，转换为列表
pro_name_list = list(pro_only_seqs_encoded.keys())
rna_name_list = list(rna_only_seqs_encoded.keys())
pro_data_list = list(pro_only_seqs_encoded.values())
rna_data_list = list(rna_only_seqs_encoded.values())

# 对每一行数据进行归一化处理
pro_data_array = np.array(pro_data_list)
rna_data_array = np.array(rna_data_list)
# pro_data_array = standardization(pro_data_array)
# rna_data_array = standardization(rna_data_array)

pro_data_list = list(pro_data_array)
rna_data_list = list(rna_data_array)

# 保存编码后的RNA数据到单独的文件中
# 更新后的保存RNA编码数据函数
# def save_encoded_rna_to_files(name, data, output_filename):
#     with open(output_filename, 'w') as f:
#         for name, vector in zip(name, data):
#             # 写入RNA名称，以FASTA格式标准的'>'开始
#             f.write(f">{name}\n")
#             # 将编码向量转换为字符串，数值间以空格分隔，每个数值格式化为小数点后三位
#             vector_str = ' '.join(f"{x:.3f}" for x in vector)
#             f.write(vector_str + "\n")
#
#
# # 更新后的保存蛋白质编码数据函数
# def save_encoded_pro_to_files(name, data, output_filename):
#     with open(output_filename, 'w') as f:
#         for name, vector in zip(name, data):
#             # 写入蛋白质名称，以FASTA格式标准的'>'开始
#             f.write(f">{name}\n")
#             # 将编码向量转换为字符串，数值间以空格分隔，每个数值格式化为小数点后三位
#             vector_str = ' '.join(f"{x:.3f}" for x in vector)
#             f.write(vector_str + "\n")
#
#
# # 保存RNA编码数据
# save_encoded_rna_to_files(rna_name_list, rna_data_list, BASE_PATH + "ncRNA_1_4_mer.txt")
# save_encoded_pro_to_files(pro_name_list, pro_data_list, BASE_PATH + "protein_1_3_mer.txt")
