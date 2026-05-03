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

ALPHABET = 'ACGU'


def readRNAFasta(file):
    """
    返回一个包含Nr个RNA数据的列表，格式为[['name','seq'],['name','seq'],...,['name','seq']]
    """
    with open(file) as f:
        records = f.read()

    if re.search('>', records) == None:
        print('The input RNA sequence must be fasta format.')
        sys.exit(1)
    records = records.split('>')[1:]
    myFasta = []
    for fasta in records:
        array = fasta.split('\n')
        name, sequence = array[0].split()[0], re.sub('[^ACGU-]', '-', ''.join(array[1:]).upper())
        myFasta.append([name, sequence])
    return myFasta


def kmers(seq, k):
    """
    将序列seq分割成长度为k的子序列，返回一个包含所有子序列的列表
    """
    v = []
    for i in range(len(seq) - k + 1):
        v.append(seq[i:i + k])
    return v


def MonoKGap(x, g):  # g=4,4*16=64
    """
    计算序列x的g-MonoKGap，返回一个包含所有g-MonoKGap的列表 len(list)=64
    """
    t = []
    # 创建一个包含所有可能的双字符组合m，使用itertools.product函数结合生物序列的字母表（ALPHABET）
    m = list(itertools.product(ALPHABET, repeat=2))  # [('A','A'),('A','C'),('A','G'),...,('U','U')]
    # 计算序列x可以提取的子序列的个数
    L_sequence = (len(x) - g - 1)
    # 遍历了序列x中所有长度为i+2的子序列（其中i从1到g），并统计了每个子序列以m列表中的双字符为起始和结束字符的数量
    for i in range(1, g + 1, 1):
        V = kmers(x, i + 2)
        for gGap in m:
            C = 0
            for v in V:
                if v[0] == gGap[0] and v[-1] == gGap[1]:
                    C += 1
            t.append(C / L_sequence)
    return t


def MonoKGap_vector(input_data, g):  # g=4,4*16=64
    """
    返回一个包含所有g-MonoKGap的列表，列表中的每个元素是一个包含所有g-MonoKGap的子序列的列表，
    样式 : [['#','Mono.0',...,'Mono.63'],['name.0','value.0',...,'value63'],...,['name.Nr',...,'value.63']]
    """
    fastas = readRNAFasta(input_data)
    vector = []
    header = ['#']  # ['#','Mono.0',...,'Mono.63']
    for f in range(g * 4 * 4):
        header.append('Mono.' + str(f))
    vector.append(header)
    sample = []
    for i in fastas:
        name, sequence = i[0], re.sub('-', '', i[1])
        sample = [name]
        each_vec = MonoKGap(sequence, g)
        sample = sample + each_vec
        vector.append(sample)
    return vector


def MonoDiKGap(x, g):  # g=2,2*64=128
    """
    计算序列x的g-MonoDiKGap，返回一个包含所有g-MonoDiKGap的列表 len(list)=128
    """
    t = []
    # 生成所有可能的g-MonoDiKGap的起始字符和结束字符组合 [('A','A','A'),('A','A','C'),('A','A','G'),...,('U','U','U')]
    m = list(itertools.product(ALPHABET, repeat=3))
    # 计算序列x的有效长度L_sequence，即len(x)-g-2
    L_sequence = (len(x) - g - 2)
    # 遍历序列x中所有长度为i+3的子序列（其中i从1到g），并统计了每个子序列以m列表中的三字符为起始、中间和结束字符的数量
    for i in range(1, g + 1, 1):
        V = kmers(x, i + 3)
        for gGap in m:
            C = 0
            for v in V:
                if v[0] == gGap[0] and v[-2] == gGap[1] and v[-1] == gGap[2]:
                    C += 1
            t.append(C / L_sequence)
    return t


def MonoDiKGap_vector(input_data, g):  # g=2,2*64=128
    """
    返回一个包含所有g-MonoDiKGap的列表，列表中的每个元素是一个包含所有g-MonoDiKGap的子序列的列表
    """
    fastas = readRNAFasta(input_data)
    vector = []
    header = ['#']
    for f in range(g * 4 * 4 * 4):
        header.append('MonoDi.' + str(f))
    vector.append(header)
    sample = []
    for i in fastas:
        name, sequence = i[0], re.sub('-', '', i[1])
        sample = [name]
        each_vec = MonoDiKGap(sequence, g)
        sample = sample + each_vec
        vector.append(sample)
    return vector


vector1 = MonoKGap_vector(BASE_PATH + "/{}_rna_seq_cleaned.fa".format(DATA_SET), g=4)  # 16*4=64
vector2 = MonoDiKGap_vector(BASE_PATH + "/{}_rna_seq_cleaned.fa".format(DATA_SET), g=2)  # 64*2=128

data_MonoKGap = np.matrix(vector1[1:])[:, 1:]  # matrix: (Nr,64)
data_MonoDiKGap = np.matrix(vector2[1:])[:, 1:]  # matrix: (Nr,128)
data_KGap_rna_seq = np.hstack((data_MonoKGap, data_MonoDiKGap))  # matrix: (Nr,192)
data_KGap_rna_seq_csv = pd.DataFrame(data_KGap_rna_seq)
# csv_data.to_csv('PN1_MG.csv')
