import numpy as np
import pandas as pd


def get_data(data_name):

    rna_mer = {}
    with open('{}/ncRNA_1_4_mer.txt'.format(data_name), 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                key = line[1:]
                rna_mer[key] = []
            else:
                rna_mer[key].extend([float(x) for x in line.split()])

    pro_mer = {}
    with open('{}/protein_1_3_mer.txt'.format(data_name), 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                key = line[1:]
                pro_mer[key] = []
            else:
                pro_mer[key].extend([float(x) for x in line.split()])

    interaction = pd.read_excel("{}/{}.xlsx".format(data_name, data_name))

    all_data = []
    for i in range(len(interaction)):
        rna_name = str(interaction["RNA names"][i])
        pro_name = str(interaction["Protein names"][i])
        all_data.append([interaction["label"][i]] + pro_mer[pro_name] + rna_mer[rna_name])

    pd_data = pd.DataFrame(all_data)
    pd_data.to_csv("{}/sample.txt".format(data_name), sep="\t", columns=None, header=None)
    print(pd_data)


'''
Choose DATA_SET = ['RPI488', 'Example', 'RPI1847', 'RPI7317', 'NPInter']
'''
DATA_SET = 'Example'
get_data(DATA_SET)


