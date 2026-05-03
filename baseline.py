import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score


def run_baseline_comparison(features, labels, protein_size, rna_size, n_folds=5, random_state=123):
    X = features[:, :protein_size + rna_size]
    y = labels.ravel()
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    rf_metrics = {'auc': [], 'f1': [], 'acc': []}
    for train_idx, test_idx in kf.split(X):
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X[train_idx], y[train_idx])
        pred = rf.predict(X[test_idx])
        prob = rf.predict_proba(X[test_idx])[:, 1]
        rf_metrics['auc'].append(roc_auc_score(y[test_idx], prob))
        rf_metrics['f1'].append(f1_score(y[test_idx], pred))
        rf_metrics['acc'].append(accuracy_score(y[test_idx], pred))

    return {k: (np.mean(v), np.std(v)) for k, v in rf_metrics.items()}


def print_comparison(baseline_results, dbenet_results):
    print("\n" + "=" * 60)
    print("Model Comparison")
    print("=" * 60)
    print(f"{'Model':<15} {'AUC':<20} {'F1':<20} {'ACC':<20}")
    print("-" * 60)

    auc_mean, auc_std = baseline_results['auc']
    f1_mean, f1_std = baseline_results['f1']
    acc_mean, acc_std = baseline_results['acc']
    print(f"{'Random Forest':<15} {auc_mean:.4f}±{auc_std:.4f}      {f1_mean:.4f}±{f1_std:.4f}      {acc_mean:.4f}±{acc_std:.4f}")

    auc = np.mean([r['auc'] for r in dbenet_results])
    auc_std = np.std([r['auc'] for r in dbenet_results])
    f1 = np.mean([r['f1'] for r in dbenet_results])
    f1_std = np.std([r['f1'] for r in dbenet_results])
    acc = np.mean([r['acc'] for r in dbenet_results])
    acc_std = np.std([r['acc'] for r in dbenet_results])
    print(f"{'DBENet-NPI':<15} {auc:.4f}±{auc_std:.4f}      {f1:.4f}±{f1_std:.4f}      {acc:.4f}±{acc_std:.4f}")
    print("=" * 60)
