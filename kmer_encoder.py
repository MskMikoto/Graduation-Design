import numpy as np
from collections import defaultdict
from itertools import product


def generate_k_mers(alphabet, k):
    return [''.join(p) for p in product(alphabet, repeat=k)]


def kmer_encode(sequence, alphabet, k_range=(1, 3), normalize=True):
    k_mer_maps = {}
    for k in range(k_range[0], k_range[1] + 1):
        k_mers = generate_k_mers(alphabet, k)
        k_mer_maps[k] = {km: i for i, km in enumerate(k_mers)}

    features = []
    for k in range(k_range[0], k_range[1] + 1):
        counts = defaultdict(int)
        for i in range(len(sequence) - k + 1):
            kmer = sequence[i:i + k]
            if kmer in k_mer_maps[k]:
                counts[kmer] += 1
        vec = [counts.get(km, 0) for km in k_mer_maps[k].keys()]
        if normalize and max(vec) > 0:
            vec = [v / max(vec) for v in vec]
        features.extend(vec)
    return np.array(features)


def read_fasta(file_path):
    sequences = {}
    with open(file_path, 'r') as f:
        name = None
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                name = line[1:].split()[0]
                sequences[name] = ''
            elif name:
                sequences[name] += line
    return sequences


PROTEIN_ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'
RNA_ALPHABET = 'AUCG'


def encode_protein_sequences(sequences, k_range=(1, 3)):
    return np.array([kmer_encode(seq.upper(), PROTEIN_ALPHABET, k_range) 
                     for seq in sequences])


def encode_rna_sequences(sequences, k_range=(1, 4)):
    return np.array([kmer_encode(seq.upper().replace('T', 'U'), RNA_ALPHABET, k_range) 
                     for seq in sequences])
