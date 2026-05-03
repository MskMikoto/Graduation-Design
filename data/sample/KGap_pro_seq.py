import itertools
import os
import re
import sys
import numpy as np
import pandas as pd

'''
Choose DATA_SET = ['RPI488', 'Example', 'RPI1847', 'RPI7317', 'NPInter']
'''
script_dir, script_name = os.path.split(os.path.abspath(sys.argv[0]))
parent_dir = os.path.dirname(script_dir)
DATA_SET = 'Example'
BASE_PATH = parent_dir + '/' + DATA_SET + '/'

# 蛋白质字母表
# ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'
ALPHABET = 'GFKDS'

# 氨基酸分组
group = {
    'G': 'GAVLMI',  # Aliphatic
    'F': 'FYW',  # Aromatic
    'K': 'KRH',  # Positive charger
    'D': 'DE',  # Negative charger
    'S': 'STCPNQ'  # uncharged
}


# 读取蛋白质FASTA文件
def readProteinFasta(file):
    """
    返回一个包含 Nr 个蛋白质数据的列表
    """
    with open(file) as f:
        records = f.read()

    if re.search('>', records) is None:
        print('The input protein sequence must be in FASTA format.')
        sys.exit(1)

    records = records.split('>')[1:]
    myFasta = []

    groupKey = group.keys()
    index = {}
    for key in groupKey:
        for aa in group[key]:
            index[aa] = key

    for fasta in records:
        array = fasta.split('\n')
        name, sequence = array[0].split()[0], re.sub('[^ACDEFGHIKLMNPQRSTVWY-]', '-', ''.join(array[1:]).upper())
        # 简化序列
        simplified_sequence = ''
        for aa in sequence:
            simplified_sequence += index.get(aa, '-')
        myFasta.append([name, simplified_sequence])
    return myFasta


# 计算k-mers
def kmers(seq, k):
    """
    将序列 seq 分割成长度为 k 的子序列，返回一个包含所有子序列的列表
    """
    v = []
    for i in range(len(seq) - k + 1):
        v.append(seq[i:i + k])
    return v


# 计算 g-MonoKGap
def MonoKGap(x, g):
    """
    计算序列 x 的 g-MonoKGap，返回一个包含所有 g-MonoKGap 的列表
    """
    t = []
    m = list(itertools.product(ALPHABET, repeat=2))
    L_sequence = len(x) - g - 1

    for i in range(1, g + 1, 1):
        V = kmers(x, i + 2)
        for gGap in m:
            C = 0
            for v in V:
                if v[0] == gGap[0] and v[-1] == gGap[1]:
                    C += 1
            t.append(C / L_sequence)
    return t


# 生成 g-MonoKGap 矩阵
def MonoKGap_vector(input_data, g):  # g=4, 4*25=100
    """
    返回一个包含所有 g-MonoKGap 的列表，列表中的每个元素是一个包含所有 g-MonoKGap 的子序列的列表
    """
    fastas = readProteinFasta(input_data)
    vector = []
    header = ['#']
    for f in range(g * 5 * 5):
        header.append('Mono.' + str(f))
    vector.append(header)

    for i in fastas:
        name, sequence = i[0], re.sub('-', '', i[1])
        sample = [name]
        each_vec = MonoKGap(sequence, g)
        sample.extend(each_vec)
        vector.append(sample)

    return vector


# 计算 g-MonoDiKGap
def MonoDiKGap(x, g):
    """
    计算序列 x 的 g-MonoDiKGap，返回一个包含所有 g-MonoDiKGap 的列表
    """
    t = []
    m = list(itertools.product(ALPHABET, repeat=3))
    L_sequence = len(x) - g - 2

    for i in range(1, g + 1, 1):
        V = kmers(x, i + 3)
        for gGap in m:
            C = 0
            for v in V:
                if v[0] == gGap[0] and v[-2] == gGap[1] and v[-1] == gGap[2]:
                    C += 1
            t.append(C / L_sequence)
    return t


# 生成 g-MonoDiKGap 矩阵
def MonoDiKGap_vector(input_data, g):
    """
    返回一个包含所有 g-MonoDiKGap 的列表，列表中的每个元素是一个包含所有 g-MonoDiKGap 的子序列的列表
    """
    fastas = readProteinFasta(input_data)
    vector = []
    header = ['#']
    for f in range(g * 5 * 5 * 5):
        header.append('MonoDi.' + str(f))
    vector.append(header)

    for i in fastas:
        name, sequence = i[0], re.sub('-', '', i[1])
        sample = [name]
        each_vec = MonoDiKGap(sequence, g)
        sample.extend(each_vec)
        vector.append(sample)

    return vector


# 主程序
vector1 = MonoKGap_vector(BASE_PATH + "/{}_protein_seq_cleaned.fa".format(DATA_SET), g=4)  # 5*5 * 4 = 100
vector2 = MonoDiKGap_vector(BASE_PATH + "/{}_protein_seq_cleaned.fa".format(DATA_SET), g=2)  # 5*5*5 * 2 = 250

data_MonoKGap = np.matrix(vector1[1:])[:, 1:]  # matrix: (Nr, 100)
data_MonoDiKGap = np.matrix(vector2[1:])[:, 1:]  # matrix: (Nr, 250)

data_KGap_protein_seq = np.hstack((data_MonoKGap, data_MonoDiKGap))  # matrix: (Nr, 250)
data_KGap_protein_seq_csv = pd.DataFrame(data_KGap_protein_seq)

# output_file = os.path.join(BASE_PATH, "data_KGap_protein_seq.csv")
# data_KGap_protein_seq_csv.to_csv(output_file, index=False, header=False)
