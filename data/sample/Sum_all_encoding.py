from Seq import *
from KGap_pro_seq import *
from KGap_rna_seq import *
from Physicochemical_Properties import *

from sklearn.preprocessing import StandardScaler


def save_encoded_rnas_to_files(name, data, output_filename):
    with open(output_filename, 'w') as f:
        for name, vector in zip(name, data):
            # 写入RNA名称，以FASTA格式标准的'>'开始
            f.write(f">{name}\n")
            # 将编码向量转换为字符串，数值间以空格分隔，每个数值格式化为小数点后三位
            vector_str = ' '.join(f"{x:.3f}" for x in vector)
            f.write(vector_str + "\n")


def save_encoded_pros_to_files(name, data, output_filename):
    with open(output_filename, 'w') as f:
        for name, vector in zip(name, data):
            # 写入蛋白质名称，以FASTA格式标准的'>'开始
            f.write(f">{name}\n")
            # 将编码向量转换为字符串，数值间以空格分隔，每个数值格式化为小数点后三位
            vector_str = ' '.join(f"{x:.3f}" for x in vector)
            f.write(vector_str + "\n")


def standardization(data):
    """
    对数据进行标准化处理。
    """
    scaler = StandardScaler()
    scaler.fit(data)
    data = scaler.transform(data)
    return data


# kmer_seq 399-D 340-D
pro_name_list = pro_name_list
pro_data_list = pro_data_list
pro_dara_list = standardization(pro_data_list)  # 399-D
rna_name_list = rna_name_list
rna_data_list = rna_data_list
rna_data_list = standardization(rna_data_list)  # 340-D

# KGap_pro_seq  350-D
KGap_pro_fea_vec = data_KGap_protein_seq.tolist()
KGap_pro_fea_vec = [np.array(i, dtype=float) for i in KGap_pro_fea_vec]
KGap_pro_fea_vec = standardization(KGap_pro_fea_vec)

# KGap_rna_seq 192-D
KGap_rna_fea_vec = data_KGap_rna_seq.tolist()
KGap_rna_fea_vec = [np.array(i, dtype=float) for i in KGap_rna_fea_vec]
KGap_rna_fea_vec = standardization(KGap_rna_fea_vec)

# pro_Physicochemical_Properties_fea_vector 80-D
pro_Physicochemical_Properties_fea_vector = pro_Physicochemical_Properties_fea_vector.tolist()
pro_Physicochemical_Properties_fea_vector = [np.array(i, dtype=float) for i in pro_Physicochemical_Properties_fea_vector]
pro_Physicochemical_Properties_fea_vector = standardization(pro_Physicochemical_Properties_fea_vector)

# rna_Physicochemical_Properties_fea_vector 20-D
rna_Physicochemical_Properties_fea_vector = rna_Physicochemical_Properties_fea_vector.tolist()
rna_Physicochemical_Properties_fea_vector = [np.array(i, dtype=float) for i in rna_Physicochemical_Properties_fea_vector]
rna_Physicochemical_Properties_fea_vector = standardization(rna_Physicochemical_Properties_fea_vector)

# 399+350+80=829
pro_data_list = [np.concatenate((pro_data_list[i], KGap_pro_fea_vec[i], pro_Physicochemical_Properties_fea_vector[i])) for i in range(len(pro_data_list))]
# 340+192+20=552
rna_data_list = [np.concatenate((rna_data_list[i], KGap_rna_fea_vec[i], rna_Physicochemical_Properties_fea_vector[i])) for i in range(len(rna_data_list))]


# 保存RNA和蛋白质的编码数据
save_encoded_rnas_to_files(rna_name_list, rna_data_list, BASE_PATH + "ncRNA_1_4_mer.txt")
save_encoded_pros_to_files(pro_name_list, pro_data_list, BASE_PATH + "protein_1_3_mer.txt")
