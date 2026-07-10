#!/usr/bin/env python3
"""
Step 6: Feature ranking for the seizure detector (RF importance + redundancy).

Goal
    Rank the 28 baseline features so we can keep the 12-16 most useful, honest
    ones for the MLP. "Useful" here means two things at once, and we refuse to
    look at only the first:

      (1) Discriminative strength  -- how much a feature helps separate
          seizure from normal windows.
      (2) Non-redundancy           -- whether that strength is *new* signal or
          a duplicate of a feature we already kept.

Why not just sort by RandomForest.feature_importances_?
    Two known failure modes bite us directly, given how these 28 features were
    built (feature-extraction-plan in memory):

      * Impurity (Gini) importance is SPLIT across correlated features. If two
        features are near-duplicates, each RF tree picks one at random per
        split, so both look ~half as important as the shared signal really is.
        A genuinely strong feature can look mediocre just because it has a twin.

      * Several of our features are redundant *by construction*, not by luck:
          - hjorth_activity == variance == std**2   (monotone twin of std)
          - the 6 FZ-CZ ratios are algebraic functions of the FZ-CZ band powers
          - broadband_power ~= sum of the band powers of the same channel
          - band powers across T7-P7 / T8-P8 / FZ-CZ co-move with global state
        Ranking on raw importance alone would silently spend 2-3 of our 16
        precious slots on the same underlying signal.

Method (all on split=='train' ONLY -- see feature-selection-strategy)
    A. Redundancy structure
         Spearman correlation (rank-based, so it is immune to the heavy skew of
         power features and to monotone transforms like std<->variance).
         Hierarchical clustering (average linkage on 1-|rho|) groups features
         that carry essentially the same information. Each cluster -> one rep.

    B. Strength, two independent estimators
         - Gini importance:        fast, biased toward correlated/high-cardinality
                                    features; reported for reference.
         - Permutation importance: honest, model-agnostic. Measured on a
                                    PATIENT-disjoint hold-out carved out of the
                                    train split (GroupShuffleSplit), so it
                                    reflects cross-patient generalisation -- the
                                    same thing we ultimately care about -- while
                                    val/test stay fully locked.

    C. De-correlated shortlist
         Within each correlation cluster, keep the single feature with the
         highest permutation importance. This is what stops two "strong but
         correlated" features from both being selected. Report the survivors
         ranked, so choosing the final 12-16 is just "take the top k".

Class imbalance (~1% seizure)
    class_weight='balanced' in the RF, plus a stratified subsample of the
    (huge) negative class so the majority does not swamp the signal and so
    permutation importance runs in reasonable time. Positives are kept in full.

Outputs (results/feature_selection/)
    feature_ranking.csv          per-feature: gini, permutation mean/std, cluster
    correlation_matrix.csv       28x28 Spearman rho
    clusters.txt                 human-readable cluster membership + reps
    ranking.png / dendrogram.png / correlation_heatmap.png
    feature_selection_report.md  the narrative + recommended shortlist

Nothing here touches val or test. This step only *ranks*; the final feature
count is chosen afterwards.

Usage
    python scripts/04_feature_selection.py
    python scripts/04_feature_selection.py --neg-per-pos 20 --corr-threshold 0.9
"""

import argparse
import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_train(dataset_path):
    """Load dataset.npz and return the TRAIN split only.

    Returns X (train features), y (train labels), groups (patient_id per row,
    for patient-disjoint sub-splitting), is_aug (per-row augmentation flag), and
    feature_names.

    We deliberately read only split=='train'. val/test must stay untouched
    until final evaluation (feature-selection-strategy in memory). Augmented
    rows are confined to train by 03_build_dataset.py. We carry the per-row
    is_augmented flag through so that permutation importance can be *scored* on
    real windows only: augmentation is a time-shift oversampling trick for the
    minority class, so its distribution differs systematically from real seizure
    windows. Fitting the RF on augmented rows is fine (and matches how the MLP
    will train), but ranking features by their effect on the augmented
    distribution would bias the selection. See compute_importances.
    """
    d = np.load(dataset_path, allow_pickle=True)
    tr = (d['split'] == 'train')
    X = d['features'][tr].astype(np.float64)
    y = d['labels'][tr].astype(np.int64)
    groups = d['patient_id'][tr]
    is_aug = d['is_augmented'][tr].astype(bool)
    names = [str(n) for n in d['feature_names']]
    if len(names) != X.shape[1]:
        names = [f'f{i}' for i in range(X.shape[1])]
    return X, y, groups, is_aug, names


def subsample_negatives(X, y, groups, is_aug, neg_per_pos, seed):
    """Keep all positives, randomly keep neg_per_pos negatives per positive.

    Severe imbalance (~1% positive) hurts twice: the RF's impurity/permutation
    signal gets dominated by the majority class, and permutation importance on
    1.4M rows is slow. We keep every seizure window and a stratified random
    subsample of normal windows. Subsampling is done GLOBALLY (not per patient)
    with a fixed seed; groups and the is_augmented flag are carried through so
    the later patient-disjoint split still sees every patient it should and can
    restrict permutation scoring to real windows.
    """
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n_keep = min(len(neg_idx), neg_per_pos * len(pos_idx))
    neg_keep = rng.choice(neg_idx, size=n_keep, replace=False)
    keep = np.concatenate([pos_idx, neg_keep])
    keep.sort()
    return X[keep], y[keep], groups[keep], is_aug[keep]


def spearman_matrix(X):
    """Spearman rho matrix, computed as Pearson on column ranks.

    Rank-based on purpose: power features are heavily right-skewed and std vs
    hjorth_activity differ by a monotone transform (square). Spearman sees both
    as the redundancy they are; Pearson would under-report them.
    """
    from scipy.stats import rankdata
    ranks = np.column_stack([rankdata(X[:, j]) for j in range(X.shape[1])])
    rho = np.corrcoef(ranks, rowvar=False)
    rho = np.nan_to_num(rho, nan=0.0)  # constant columns -> 0 correlation
    np.fill_diagonal(rho, 1.0)
    return rho


def cluster_features(rho, names, threshold):
    """Hierarchical clustering on 1-|rho|; cut so each cluster is 'redundant'.

    Distance = 1 - |rho|: two features are close when they carry the same
    information regardless of sign (a ratio and its inverse are anti-correlated
    but still redundant). Average linkage, then cut the tree at distance
    (1 - threshold): e.g. threshold=0.9 groups features whose |rho| >= ~0.9.

    Returns (labels, linkage_matrix): labels[j] is the cluster id of feature j.
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    dist = 1.0 - np.abs(rho)
    dist = (dist + dist.T) / 2.0          # enforce symmetry
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method='average')  # average linkage on a distance
    labels = fcluster(Z, t=1.0 - threshold, criterion='distance')
    return labels, Z


def compute_importances(X, y, groups, is_aug, seed, n_estimators, n_repeats):
    """Gini + patient-disjoint permutation importance from a single RF.

    Split the (subsampled) train rows by PATIENT into an RF-fit set and a
    held-out importance set (GroupShuffleSplit, ~30% of patients held out). The
    RF never sees the held-out patients, so permutation importance measured
    there reflects cross-patient generalisation -- exactly the deployment
    condition -- instead of memorised training rows. val/test remain locked.

    Augmentation asymmetry (why the hold-out is filtered to real windows):
        Augmented copies are time-shifted duplicates added ONLY to positive
        windows to help the model learn the minority class. Two consequences:
          * The RF *fit* set may keep them -- that mirrors how the downstream
            MLP will actually be trained (on the augmented train split).
          * The *scoring* set must NOT: permutation importance is meant to
            estimate how much each feature helps on the REAL window
            distribution we deploy against. Scoring on augmented rows would
            measure "generalisation to our augmentation recipe" instead, and
            since ~66% of train positives are augmented, that bias is large.
        So we score permutation importance on real windows only
        (``~is_aug`` within the held-out patients).

    Returns dict with gini, perm_mean, perm_std, and hold-out bookkeeping.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import GroupShuffleSplit

    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    fit_idx, hold_idx = next(gss.split(X, y, groups))

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight='balanced',
        max_features='sqrt',
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=seed,
    )
    rf.fit(X[fit_idx], y[fit_idx])
    gini = rf.feature_importances_

    # Restrict the scoring set to REAL (non-augmented) held-out windows so the
    # importance reflects the true deployment distribution, not the training-
    # time augmentation. The RF still trained on whatever fit_idx contained.
    hold_real = hold_idx[~is_aug[hold_idx]]

    # Permutation importance scored on held-out REAL patients. Use average
    # precision (area under PR curve): with ~1% positives, PR-based scoring is
    # far more informative than accuracy, which a trivial all-negative model
    # would ace.
    perm = permutation_importance(
        rf, X[hold_real], y[hold_real],
        scoring='average_precision',
        n_repeats=n_repeats,
        n_jobs=-1,
        random_state=seed,
    )

    held_groups = sorted(set(groups[hold_real].tolist()))
    return {
        'gini': gini,
        'perm_mean': perm.importances_mean,
        'perm_std': perm.importances_std,
        'held_out_patients': held_groups,
        'n_fit': len(fit_idx),
        'n_hold': len(hold_real),
        'n_hold_pos_real': int((y[hold_real] == 1).sum()),
        'n_fit_aug': int(is_aug[fit_idx].sum()),
    }


def pick_representatives(labels, perm_mean, names):
    """Within each cluster, the highest-permutation-importance feature wins.

    This is the de-correlation step: correlated features share a cluster, and
    only the strongest survivor represents it. Ties (equal permutation score)
    break on name order for determinism.
    """
    reps = {}
    for cid in sorted(set(labels)):
        members = [j for j in range(len(names)) if labels[j] == cid]
        best = max(members, key=lambda j: (perm_mean[j], -j))
        reps[cid] = best
    return reps


def main():
    ap = argparse.ArgumentParser(description='Rank features via RF + redundancy.')
    ap.add_argument('--dataset', default='data/processed/dataset.npz')
    ap.add_argument('--neg-per-pos', type=int, default=20,
                    help='Negatives kept per positive when subsampling (default 20)')
    ap.add_argument('--corr-threshold', type=float, default=0.9,
                    help='|Spearman rho| at/above which features are one cluster (default 0.9)')
    ap.add_argument('--n-estimators', type=int, default=400)
    ap.add_argument('--n-repeats', type=int, default=10,
                    help='Permutation-importance repeats (default 10)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--outdir', default='results/feature_selection')
    args = ap.parse_args()

    dataset_path = project_root / args.dataset
    outdir = project_root / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Step 6: Feature ranking (RF importance + redundancy)")
    print("=" * 70)

    # --- Load TRAIN only -----------------------------------------------------
    X, y, groups, is_aug, names = load_train(dataset_path)
    n_feat = X.shape[1]
    print(f"\nTrain rows: {len(y):,}  features: {n_feat}")
    print(f"  positives: {int((y == 1).sum()):,}  "
          f"negatives: {int((y == 0).sum()):,}  "
          f"({100 * (y == 1).mean():.2f}% positive)")
    print(f"  positives -- real: {int(((y == 1) & ~is_aug).sum()):,}, "
          f"augmented: {int(((y == 1) & is_aug).sum()):,}")
    print(f"  patients: {sorted(set(groups.tolist()))}")

    # --- Subsample negatives -------------------------------------------------
    Xs, ys, gs, aug_s = subsample_negatives(
        X, y, groups, is_aug, args.neg_per_pos, args.seed)
    print(f"\nAfter subsampling (neg_per_pos={args.neg_per_pos}): {len(ys):,} rows "
          f"({int((ys == 1).sum()):,} pos, {int((ys == 0).sum()):,} neg)")
    print(f"  positives -- real: {int(((ys == 1) & ~aug_s).sum()):,}, "
          f"augmented: {int(((ys == 1) & aug_s).sum()):,}")

    # --- Redundancy structure (Spearman + clustering) ------------------------
    # Correlation/redundancy is measured on REAL windows only, for the same
    # reason permutation importance is (below): augmented positives are time-
    # shifted duplicates that over-sample the seizure region of feature space
    # and could inflate correlations that only hold during seizures. Redundancy
    # should reflect the real deployment distribution, not the augmentation
    # recipe. The RF *fit* still uses augmented rows; every *measurement* does not.
    print("\nComputing Spearman correlation + clustering (REAL windows only) ...")
    Xs_real = Xs[~aug_s]
    rho = spearman_matrix(Xs_real)
    labels, Z = cluster_features(rho, names, args.corr_threshold)
    n_clusters = len(set(labels))
    print(f"  {n_clusters} clusters at |rho| >= {args.corr_threshold} "
          f"(from {len(Xs_real):,} real rows)")

    # Flag the near-perfect correlations explicitly -- these are the by-design
    # duplicates we predicted; seeing them confirms the pipeline is sane.
    print("\nHighly correlated pairs (|rho| >= 0.95):")
    found = False
    for i in range(n_feat):
        for j in range(i + 1, n_feat):
            if abs(rho[i, j]) >= 0.95:
                found = True
                print(f"    {rho[i, j]:+.3f}  {names[i]:32s} <-> {names[j]}")
    if not found:
        print("    (none)")

    # --- Importances ---------------------------------------------------------
    print("\nFitting RandomForest + permutation importance "
          "(patient-disjoint hold-out, REAL windows only) ...")
    imp = compute_importances(Xs, ys, gs, aug_s, args.seed,
                              args.n_estimators, args.n_repeats)
    print(f"  RF fit on {imp['n_fit']:,} rows "
          f"(real+augmented; {imp['n_fit_aug']:,} augmented)")
    print(f"  permutation scored on {imp['n_hold']:,} held-out REAL windows "
          f"({imp['n_hold_pos_real']:,} positive)")
    print(f"  held-out patients: {imp['held_out_patients']}")

    gini = imp['gini']
    perm_mean = imp['perm_mean']
    perm_std = imp['perm_std']

    # --- Representatives (de-correlation) ------------------------------------
    reps = pick_representatives(labels, perm_mean, names)
    is_rep = np.zeros(n_feat, dtype=bool)
    for j in reps.values():
        is_rep[j] = True

    # --- Assemble ranking table ---------------------------------------------
    order = np.argsort(-perm_mean)  # rank by honest permutation importance
    perm_rank = np.empty(n_feat, dtype=int)
    perm_rank[order] = np.arange(1, n_feat + 1)

    # --- Write CSVs ----------------------------------------------------------
    csv_path = outdir / 'feature_ranking.csv'
    with open(csv_path, 'w') as f:
        f.write("perm_rank,feature,perm_importance_mean,perm_importance_std,"
                "gini_importance,cluster,is_cluster_representative\n")
        for j in order:
            f.write(f"{perm_rank[j]},{names[j]},{perm_mean[j]:.6f},"
                    f"{perm_std[j]:.6f},{gini[j]:.6f},{labels[j]},"
                    f"{int(is_rep[j])}\n")
    print(f"\nWrote {csv_path}")

    corr_path = outdir / 'correlation_matrix.csv'
    with open(corr_path, 'w') as f:
        f.write("," + ",".join(names) + "\n")
        for i in range(n_feat):
            f.write(names[i] + "," +
                    ",".join(f"{rho[i, j]:.4f}" for j in range(n_feat)) + "\n")
    print(f"Wrote {corr_path}")

    # --- Clusters (human-readable) -------------------------------------------
    clusters_path = outdir / 'clusters.txt'
    with open(clusters_path, 'w') as f:
        f.write(f"Correlation clusters (|Spearman rho| >= {args.corr_threshold})\n")
        f.write(f"{n_clusters} clusters over {n_feat} features.\n")
        f.write("Representative = highest permutation importance in the cluster.\n\n")
        for cid in sorted(set(labels)):
            members = [j for j in range(n_feat) if labels[j] == cid]
            members.sort(key=lambda j: -perm_mean[j])
            f.write(f"Cluster {cid}  ({len(members)} feature"
                    f"{'s' if len(members) > 1 else ''}):\n")
            for j in members:
                tag = "  <== REP" if is_rep[j] else ""
                f.write(f"    perm={perm_mean[j]:.5f}  gini={gini[j]:.5f}  "
                        f"{names[j]}{tag}\n")
            f.write("\n")
    print(f"Wrote {clusters_path}")

    # --- Console: full ranking + de-correlated shortlist ---------------------
    print("\n" + "=" * 70)
    print("Full ranking by permutation importance (honest, cross-patient)")
    print("=" * 70)
    print(f"{'rk':>3} {'feature':34s} {'perm(mean+/-std)':>22s} "
          f"{'gini':>7s} {'clus':>4s} {'rep':>3s}")
    for j in order:
        print(f"{perm_rank[j]:>3} {names[j]:34s} "
              f"{perm_mean[j]:>10.5f} +/-{perm_std[j]:<7.5f} "
              f"{gini[j]:>7.4f} {labels[j]:>4} {'*' if is_rep[j] else '':>3}")

    # De-correlated shortlist: only representatives, ranked. We additionally
    # flag which representatives carry a POSITIVE and STABLE permutation score
    # (perm_mean > perm_std > 0): a representative whose mean importance is at or
    # below zero, or smaller than its own noise, is not evidence of real signal
    # and should NOT be selected just to hit a target count. The final k is
    # chosen by the val sweep among these stable representatives -- not by a
    # mechanical "take the top 16".
    rep_order = [j for j in order if is_rep[j]]
    stable = [j for j in rep_order if perm_mean[j] > perm_std[j] > 0
              and perm_mean[j] > 0]
    print("\n" + "=" * 70)
    print(f"DE-CORRELATED SHORTLIST ({len(rep_order)} cluster representatives)")
    print(f"  {len(stable)} of them have a positive, stable importance "
          f"(perm_mean > perm_std > 0) -- the only ones worth selecting.")
    print("  Final k is decided by the val sweep among the stable ones, NOT by "
          "a fixed top-N.")
    print("=" * 70)
    for rank, j in enumerate(rep_order, 1):
        mark = "OK  " if (perm_mean[j] > perm_std[j] > 0 and perm_mean[j] > 0) \
            else "weak"
        print(f"  {rank:>2}. [{mark}] {names[j]:34s} perm={perm_mean[j]:.5f}  "
              f"gini={gini[j]:.4f}  (cluster {labels[j]})")

    # --- Plots (best-effort; skip if matplotlib missing) ---------------------
    try:
        _make_plots(outdir, rho, Z, names, labels, order, perm_mean,
                    perm_std, gini, is_rep, args.corr_threshold)
        print(f"\nPlots written to {outdir}")
    except Exception as e:  # pragma: no cover - plotting is non-essential
        print(f"\n(Plotting skipped: {e})")

    # --- Markdown report -----------------------------------------------------
    _write_report(outdir, args, names, order, perm_mean, perm_std, gini,
                  labels, is_rep, reps, rho, imp, len(ys),
                  int((ys == 1).sum()), n_clusters, rep_order)
    print(f"Wrote {outdir / 'feature_selection_report.md'}")

    print("\nDone. Ranking only -- val/test untouched. Importance was scored on "
          "REAL (non-augmented) held-out windows. Choose the final k from the "
          "STABLE representatives via the val sweep -- do not pad to a fixed count.")


def _make_plots(outdir, rho, Z, names, labels, order, perm_mean, perm_std,
                gini, is_rep, threshold):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram

    n = len(names)

    # 1. Permutation-importance bar chart (ranked), reps highlighted.
    fig, ax = plt.subplots(figsize=(9, 8))
    ys = np.arange(n)[::-1]
    colors = ['#2c7fb8' if is_rep[j] else '#bdbdbd' for j in order]
    ax.barh(ys, perm_mean[order], xerr=perm_std[order],
            color=colors, ecolor='#888', capsize=2)
    ax.set_yticks(ys)
    ax.set_yticklabels([names[j] for j in order], fontsize=7)
    ax.set_xlabel('Permutation importance (avg-precision drop)')
    ax.set_title('Feature ranking (blue = cluster representative)')
    fig.tight_layout()
    fig.savefig(outdir / 'ranking.png', dpi=130)
    plt.close(fig)

    # 2. Dendrogram.
    fig, ax = plt.subplots(figsize=(11, 6))
    dendrogram(Z, labels=names, leaf_font_size=7, ax=ax,
               color_threshold=1.0 - threshold)
    ax.axhline(1.0 - threshold, ls='--', c='r', lw=1,
               label=f'cut @ |rho|={threshold}')
    ax.set_ylabel('distance (1 - |Spearman rho|)')
    ax.set_title('Feature clustering (redundancy structure)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / 'dendrogram.png', dpi=130)
    plt.close(fig)

    # 3. Correlation heatmap.
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(rho, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, fontsize=6, rotation=90)
    ax.set_yticklabels(names, fontsize=6)
    ax.set_title('Spearman correlation matrix')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(outdir / 'correlation_heatmap.png', dpi=130)
    plt.close(fig)


def _write_report(outdir, args, names, order, perm_mean, perm_std, gini,
                  labels, is_rep, reps, rho, imp, n_rows, n_pos,
                  n_clusters, rep_order):
    lines = []
    lines.append("# Feature Selection Report (Step 6)\n")
    lines.append("Ranking only. `val`/`test` untouched; everything below is "
                 "computed on `split=='train'`.\n")
    lines.append("## Setup\n")
    lines.append(f"- Rows used (after negative subsampling): **{n_rows:,}** "
                 f"({n_pos:,} positive)")
    lines.append(f"- Negatives per positive: `{args.neg_per_pos}`")
    lines.append(f"- RF: `{args.n_estimators}` trees, `class_weight=balanced`, "
                 f"`max_features=sqrt`, `min_samples_leaf=5`")
    lines.append(f"- Permutation importance: `average_precision`, "
                 f"`{args.n_repeats}` repeats, on a **patient-disjoint** hold-out "
                 f"scored on **real (non-augmented) windows only**")
    lines.append(f"- RF fit rows: {imp['n_fit']:,} "
                 f"(real+augmented, {imp['n_fit_aug']:,} augmented); "
                 f"hold-out real rows: {imp['n_hold']:,} "
                 f"({imp['n_hold_pos_real']:,} seizure)")
    lines.append(f"- Held-out patients: `{imp['held_out_patients']}`")
    lines.append(f"- Correlation/redundancy also measured on **real windows "
                 f"only**; augmented copies are used for the RF *fit* only")
    lines.append(f"- Correlation clusters at |Spearman rho| >= "
                 f"`{args.corr_threshold}`: **{n_clusters}**\n")

    lines.append("## Why this method, not raw `feature_importances_`\n")
    lines.append("Impurity importance is split across correlated features and "
                 "several of these 28 are redundant by construction "
                 "(`hjorth_activity == std**2`; the 6 ratios are functions of "
                 "the FZ-CZ band powers; `broadband ~= sum(bands)`). So we rank "
                 "on **permutation importance** (honest, cross-patient) and use "
                 "**Spearman clustering** to keep only one representative per "
                 "redundant group.\n")

    lines.append("## Full ranking (by permutation importance)\n")
    lines.append("| rk | feature | perm mean | perm std | gini | cluster | rep |")
    lines.append("|---:|---|---:|---:|---:|---:|:--:|")
    for rank, j in enumerate(order, 1):
        lines.append(f"| {rank} | `{names[j]}` | {perm_mean[j]:.5f} | "
                     f"{perm_std[j]:.5f} | {gini[j]:.4f} | {labels[j]} | "
                     f"{'*' if is_rep[j] else ''} |")

    lines.append("\n## Highly correlated pairs (|rho| >= 0.95)\n")
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if abs(rho[i, j]) >= 0.95:
                pairs.append((rho[i, j], names[i], names[j]))
    if pairs:
        lines.append("| rho | feature A | feature B |")
        lines.append("|---:|---|---|")
        for r, a, b in sorted(pairs, key=lambda t: -abs(t[0])):
            lines.append(f"| {r:+.3f} | `{a}` | `{b}` |")
    else:
        lines.append("_None._")

    lines.append(f"\n## De-correlated shortlist ({len(rep_order)} representatives)\n")
    lines.append("Redundancy already removed. Importance was scored on **real "
                 "(non-augmented) held-out windows only**, so the ranking "
                 "reflects the true seizure distribution, not the training "
                 "augmentation. Do **not** pad to a fixed count: keep only "
                 "representatives whose permutation importance is **positive "
                 "and stable** (mean > std), then let the validation sweep pick "
                 "the final `k` among those.\n")
    lines.append("| pick | feature | perm | perm std | gini | cluster | stable |")
    lines.append("|---:|---|---:|---:|---:|---:|:--:|")
    for rank, j in enumerate(rep_order, 1):
        stable = "yes" if perm_mean[j] > perm_std[j] and perm_mean[j] > 0 else ""
        lines.append(f"| {rank} | `{names[j]}` | {perm_mean[j]:.5f} | "
                     f"{perm_std[j]:.5f} | {gini[j]:.4f} | {labels[j]} | "
                     f"{stable} |")

    lines.append("\n## Files\n")
    lines.append("- `feature_ranking.csv` -- per-feature scores + cluster + rep flag")
    lines.append("- `correlation_matrix.csv` -- 28x28 Spearman rho")
    lines.append("- `clusters.txt` -- cluster membership, reps marked")
    lines.append("- `ranking.png`, `dendrogram.png`, `correlation_heatmap.png`")

    (outdir / 'feature_selection_report.md').write_text(
        "\n".join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
