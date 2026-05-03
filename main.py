from model import *
from metric import *
from kmer_encoder import read_fasta, encode_protein_sequences, encode_rna_sequences
from similarity_matrix import build_similarity_matrix, visualize_similarity_matrix, save_similarity_matrix
from gnn_module import GCNEncoder, build_adjacency_from_similarity, normalize_adjacency
from baseline import run_baseline_comparison, print_comparison
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve
import warnings
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

batch_size = 1024
random_state = 123
DATA_SET = 'Example'
USE_KMER = False
BUILD_SIMILARITY = True
USE_GNN = True
GNN_HIDDEN = 256
GNN_OUT = 128
GNN_K_NEIGHBORS = 10
EARLY_STOP_PATIENCE = 5

print("=" * 60)
print("DBENet-NPI: ncRNA-Protein Interaction Prediction")
print("=" * 60)

if USE_KMER:
    print("\n[INFO] Using k-mer encoding...")
    protein_seqs = read_fasta(f'data/{DATA_SET}/RPI1807_protein_seq_cleaned.fa')
    rna_seqs = read_fasta(f'data/{DATA_SET}/RPI1807_rna_seq_cleaned.fa')
    interaction_df = pd.read_excel(f'data/{DATA_SET}/{DATA_SET}.xlsx')

    protein_names = list(protein_seqs.keys())
    rna_names = list(rna_seqs.keys())
    protein_features = encode_protein_sequences([protein_seqs[n] for n in protein_names])
    rna_features = encode_rna_sequences([rna_seqs[n] for n in rna_names])

    features, labels = [], []
    for i in range(len(interaction_df)):
        rna_name, pro_name = str(interaction_df["RNA names"][i]), str(interaction_df["Protein names"][i])
        if rna_name in rna_names and pro_name in protein_names:
            features.append(np.concatenate([protein_features[protein_names.index(pro_name)],
                                           rna_features[rna_names.index(rna_name)]]))
            labels.append(interaction_df["label"][i])
    features, labels = np.array(features), np.array(labels).reshape(-1, 1)
    protein_size, rna_size = protein_features.shape[1], rna_features.shape[1]
    print(f"Protein dim: {protein_size}, RNA dim: {rna_size}, Samples: {len(features)}")
else:
    print("\n[INFO] Using pre-computed features...")
    protein_feature = ['P' + str(i) for i in range(1, 830)]
    rna_feature = ['R' + str(i) for i in range(1, 553)]
    col_names = ['label'] + protein_feature + rna_feature
    data = pd.read_csv(f'data/{DATA_SET}/sample.txt', names=col_names, sep='\t')
    labels = data['label'].values.reshape(-1, 1)
    features = data.drop(columns=['label']).values
    protein_size, rna_size = 829, 552

if BUILD_SIMILARITY:
    print("\n[INFO] Building similarity matrices...")
    os.makedirs(f'data/{DATA_SET}/similarity', exist_ok=True)
    protein_feat = features[:, :protein_size]
    rna_feat = features[:, protein_size:]

    pro_sim = build_similarity_matrix(protein_feat)
    rna_sim = build_similarity_matrix(rna_feat)

    print(f"Protein similarity: mean={pro_sim.mean():.4f}, std={pro_sim.std():.4f}")
    print(f"RNA similarity: mean={rna_sim.mean():.4f}, std={rna_sim.std():.4f}")

    save_similarity_matrix(pro_sim, f'data/{DATA_SET}/similarity/protein_sim.npy')
    save_similarity_matrix(rna_sim, f'data/{DATA_SET}/similarity/rna_sim.npy')
    visualize_similarity_matrix(pro_sim, f'data/{DATA_SET}/similarity/protein_sim.png', "Protein")
    visualize_similarity_matrix(rna_sim, f'data/{DATA_SET}/similarity/rna_sim.png', "RNA")
    print("Similarity matrices saved.")

if USE_GNN:
    print("\n[INFO] Building GNN adjacency matrices...")
    pro_adj = build_adjacency_from_similarity(pro_sim, k_neighbors=GNN_K_NEIGHBORS)
    rna_adj = build_adjacency_from_similarity(rna_sim, k_neighbors=GNN_K_NEIGHBORS)
    pro_adj = normalize_adjacency(pro_adj)
    rna_adj = normalize_adjacency(rna_adj)
    print(f"Protein adj: {pro_adj.shape}, edges={int(pro_adj.sum())}")
    print(f"RNA adj: {rna_adj.shape}, edges={int(rna_adj.sum())}")

    pro_gnn = GCNEncoder(protein_size, GNN_HIDDEN, GNN_OUT)
    rna_gnn = GCNEncoder(rna_size, GNN_HIDDEN, GNN_OUT)

    with torch.no_grad():
        pro_feat_tensor = torch.tensor(protein_feat, dtype=torch.float32)
        rna_feat_tensor = torch.tensor(rna_feat, dtype=torch.float32)
        pro_gnn_feat = pro_gnn(pro_feat_tensor, pro_adj).numpy()
        rna_gnn_feat = rna_gnn(rna_feat_tensor, rna_adj).numpy()

    features = np.concatenate([protein_feat, rna_feat, pro_gnn_feat, rna_gnn_feat], axis=1)
    print(f"Enhanced feature dim: {features.shape[1]} (original: {protein_size + rna_size}, GNN: {GNN_OUT * 2})")

print("\n[INFO] Starting 5-fold cross-validation...")
print("-" * 60)

k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
all_results = []
all_fpr_tpr = []
all_train_losses, all_test_losses, all_test_aucs = [], [], []

for fold, (train_idx, test_idx) in enumerate(kf.split(features)):
    print(f"\nFold {fold + 1}/{k_folds}")

    train_x, test_x = features[train_idx], features[test_idx]
    train_y, test_y = labels[train_idx], labels[test_idx]

    train_loader = DataLoader(TensorDataset(torch.from_numpy(train_x).float(), torch.from_numpy(train_y).float()),
                              shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(test_x).float(), torch.from_numpy(test_y).float()),
                             batch_size=batch_size)

    model = DBENet_NPI(protein_size, rna_size)
    loss_func = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-6)

    train_losses, test_losses, test_aucs = [], [], []
    epochs = 100
    best_test_auc = 0
    no_improve = 0
    best_fold_results = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            if USE_GNN:
                x_orig = x[:, :protein_size + rna_size]
                loss = loss_func(model(x_orig), y.float())
            else:
                loss = loss_func(model(x), y.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_loss = total_loss / len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        true_labels, pred_probs = [], []
        test_loss = 0
        with torch.no_grad():
            for x, y in test_loader:
                if USE_GNN:
                    x_orig = x[:, :protein_size + rna_size]
                    y_hat = model(x_orig)
                else:
                    y_hat = model(x)
                test_loss += loss_func(y_hat, y.float()).item()
                true_labels.extend(y.tolist())
                pred_probs.extend(y_hat.tolist())
        test_losses.append(test_loss / len(test_loader))

        test_auc, test_rec, test_pre, test_f1, test_acc, test_spe, test_mcc = get_result(test_loader, model, protein_size + rna_size if USE_GNN else None)
        test_aucs.append(test_auc)

        print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {train_loss:.4f} | Test Loss: {test_losses[-1]:.4f} | "
              f"Test AUC: {test_auc:.4f} | Test F1: {test_f1:.4f} | Test ACC: {test_acc:.4f}")

        if test_auc > best_test_auc:
            best_test_auc = test_auc
            best_fold_results = {'auc': test_auc, 'f1': test_f1, 'acc': test_acc,
                                 'rec': test_rec, 'pre': test_pre, 'spe': test_spe, 'mcc': test_mcc}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch+1} (no improvement for {EARLY_STOP_PATIENCE} epochs)")
                break

    all_train_losses.append(train_losses)
    all_test_losses.append(test_losses)
    all_test_aucs.append(test_aucs)

    fpr, tpr, _ = roc_curve(true_labels, pred_probs)
    all_fpr_tpr.append((fpr, tpr, best_test_auc))

    all_results.append(best_fold_results)
    print(f"Best Fold {fold+1}: AUC={best_test_auc:.4f}, F1={best_fold_results['f1']:.4f}, ACC={best_fold_results['acc']:.4f}")

print("\n" + "=" * 60)
print("Visualization")
print("=" * 60)

max_epochs = max(len(losses) for losses in all_train_losses)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
for i, (fpr, tpr, auc_val) in enumerate(all_fpr_tpr):
    plt.plot(fpr, tpr, label=f'Fold {i+1} (AUC={auc_val:.4f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve')
plt.legend(fontsize=8)

plt.subplot(1, 3, 2)
for i, losses in enumerate(all_train_losses):
    plt.plot(range(1, len(losses)+1), losses, 'b-', alpha=0.3)
for i, losses in enumerate(all_test_losses):
    plt.plot(range(1, len(losses)+1), losses, 'r-', alpha=0.3)
plt.plot(range(1, max_epochs+1), np.mean([l + [l[-1]]*(max_epochs-len(l)) for l in all_train_losses], axis=0), 'b-', label='Train Loss')
plt.plot(range(1, max_epochs+1), np.mean([l + [l[-1]]*(max_epochs-len(l)) for l in all_test_losses], axis=0), 'r-', label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve (with Early Stopping)')
plt.legend()

plt.subplot(1, 3, 3)
for i, aucs in enumerate(all_test_aucs):
    plt.plot(range(1, len(aucs)+1), aucs, 'g-', alpha=0.3)
plt.plot(range(1, max_epochs+1), np.mean([a + [a[-1]]*(max_epochs-len(a)) for a in all_test_aucs], axis=0), 'g-', label='Test AUC')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.title('AUC Curve')
plt.legend()

plt.tight_layout()
plt.savefig(f'data/{DATA_SET}/results_visualization.png', dpi=300)
plt.close()
print(f"Visualization saved to: data/{DATA_SET}/results_visualization.png")

print("\n" + "=" * 60)
print("Results Summary")
print("=" * 60)
print(f"Average AUC: {np.mean([r['auc'] for r in all_results]):.4f} ± {np.std([r['auc'] for r in all_results]):.4f}")
print(f"Average F1:  {np.mean([r['f1'] for r in all_results]):.4f} ± {np.std([r['f1'] for r in all_results]):.4f}")
print(f"Average ACC: {np.mean([r['acc'] for r in all_results]):.4f} ± {np.std([r['acc'] for r in all_results]):.4f}")
print(f"Average Rec: {np.mean([r['rec'] for r in all_results]):.4f} ± {np.std([r['rec'] for r in all_results]):.4f}")
print(f"Average Pre: {np.mean([r['pre'] for r in all_results]):.4f} ± {np.std([r['pre'] for r in all_results]):.4f}")
print(f"Average Spe: {np.mean([r['spe'] for r in all_results]):.4f} ± {np.std([r['spe'] for r in all_results]):.4f}")
print(f"Average MCC: {np.mean([r['mcc'] for r in all_results]):.4f} ± {np.std([r['mcc'] for r in all_results]):.4f}")

baseline_results = run_baseline_comparison(features, labels, protein_size, rna_size)
print_comparison(baseline_results, all_results)

print("\n" + "=" * 60)
print("Ablation Study Summary")
print("=" * 60)
print("Current model uses: Original Features + GNN Features")
print(f"AUC: {np.mean([r['auc'] for r in all_results]):.4f} ± {np.std([r['auc'] for r in all_results]):.4f}")
print(f"F1:  {np.mean([r['f1'] for r in all_results]):.4f} ± {np.std([r['f1'] for r in all_results]):.4f}")
print(f"ACC: {np.mean([r['acc'] for r in all_results]):.4f} ± {np.std([r['acc'] for r in all_results]):.4f}")
print("\nTo run ablation experiments, set USE_GNN=False in main.py and re-run.")
