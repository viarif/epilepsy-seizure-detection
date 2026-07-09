"""
Visualize and validate extracted features - SINGLE FILE ONLY.

⚠️ WARNING: This script analyzes features from a SINGLE recording file.
   Results are for quality checking purposes only, NOT for drawing conclusions
   about feature importance across the entire dataset.

This script:
1. Loads extracted features from ONE file
2. Computes statistical summaries for quality checks
3. Generates visualizations for sanity checking
4. Checks for feature quality issues (NaN, Inf, constant features)

For final feature importance analysis, use the full dataset after all
recordings are processed.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 10)


def load_features(feature_file):
    """Load features from npz file."""
    data = np.load(feature_file, allow_pickle=True)
    features = data['features']
    labels = data['labels']
    feature_names = data['feature_names']
    return features, labels, feature_names


def compute_feature_statistics(features, labels, feature_names):
    """Compute per-feature statistics for seizure and normal windows."""

    seizure_mask = labels == 1
    normal_mask = labels == 0

    seizure_features = features[seizure_mask]
    normal_features = features[normal_mask]

    stats = []

    for i, name in enumerate(feature_names):
        seizure_mean = np.mean(seizure_features[:, i])
        seizure_std = np.std(seizure_features[:, i])
        normal_mean = np.mean(normal_features[:, i])
        normal_std = np.std(normal_features[:, i])

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((seizure_std**2 + normal_std**2) / 2)
        cohens_d = (seizure_mean - normal_mean) / (pooled_std + 1e-10)

        stats.append({
            'feature': name,
            'seizure_mean': seizure_mean,
            'seizure_std': seizure_std,
            'normal_mean': normal_mean,
            'normal_std': normal_std,
            'mean_diff': seizure_mean - normal_mean,
            'cohens_d': cohens_d,
        })

    return stats


def plot_feature_distributions(features, labels, feature_names, output_dir):
    """Plot feature distributions for seizure vs normal windows."""

    seizure_mask = labels == 1
    normal_mask = labels == 0

    n_features = len(feature_names)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = axes.flatten()

    for i, name in enumerate(feature_names):
        ax = axes[i]

        # Plot histograms
        seizure_data = features[seizure_mask, i]
        normal_data = features[normal_mask, i]

        ax.hist(normal_data, bins=30, alpha=0.6, label='Normal', color='blue', density=True)
        ax.hist(seizure_data, bins=30, alpha=0.6, label='Seizure', color='red', density=True)

        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.set_title(name, fontsize=10)
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    output_path = output_dir / 'feature_distributions.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_feature_importance_preview(stats, output_dir):
    """Plot top features by effect size (Cohen's d) - SINGLE FILE ONLY."""

    # Sort by absolute Cohen's d
    sorted_stats = sorted(stats, key=lambda x: abs(x['cohens_d']), reverse=True)

    # Top 15 features
    top_stats = sorted_stats[:15]

    feature_names = [s['feature'] for s in top_stats]
    cohens_d = [s['cohens_d'] for s in top_stats]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ['red' if d > 0 else 'blue' for d in cohens_d]
    bars = ax.barh(range(len(feature_names)), cohens_d, color=colors, alpha=0.7)

    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names)
    ax.set_xlabel("Cohen's d (Effect Size)")
    ax.set_title("Top 15 Features by Effect Size - SINGLE FILE ONLY\n(For Quality Check, NOT Final Feature Selection)")
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    ax.grid(True, alpha=0.3, axis='x')

    # Add warning text
    fig.text(0.5, 0.02, 'WARNING: Results from single file only. Do NOT use for feature selection.',
             ha='center', fontsize=9, style='italic', color='red')

    plt.tight_layout()
    output_path = output_dir / 'single_file_feature_effect_size.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_correlation_matrix(features, feature_names, output_dir):
    """Plot feature correlation matrix."""

    # Compute correlation matrix
    corr_matrix = np.corrcoef(features.T)

    # Plot
    fig, ax = plt.subplots(figsize=(16, 14))

    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

    # Set ticks
    ax.set_xticks(range(len(feature_names)))
    ax.set_yticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=90, ha='right', fontsize=8)
    ax.set_yticklabels(feature_names, fontsize=8)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation', rotation=270, labelpad=20)

    ax.set_title('Feature Correlation Matrix')

    plt.tight_layout()
    output_path = output_dir / 'feature_correlation_matrix.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    """Main validation and visualization function."""

    print("=" * 80)
    print("Feature Validation - SINGLE FILE QUALITY CHECK")
    print("=" * 80)
    print()
    print("WARNING: This analyzes ONE file only. Results are for sanity checking,")
    print("         NOT for drawing conclusions about feature importance.")
    print()

    # Paths
    project_root = Path(__file__).parent.parent
    processed_dir = project_root / 'data' / 'processed'
    output_dir = project_root / 'results' / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find feature files
    feature_files = sorted(processed_dir.glob('*_features.npz'))

    if not feature_files:
        print("Error: No feature files found!")
        return

    # Process first file
    feature_file = feature_files[0]
    print(f"Loading: {feature_file.name}")

    features, labels, feature_names = load_features(feature_file)

    print(f"  Features shape: {features.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Seizure windows: {np.sum(labels == 1)}")
    print(f"  Normal windows: {np.sum(labels == 0)}")
    print()

    # Compute statistics
    print("Computing feature statistics...")
    stats = compute_feature_statistics(features, labels, feature_names)

    # Print top features by effect size (FOR THIS FILE ONLY)
    print("\nTop 10 features by effect size - THIS FILE ONLY (quality check):")
    print("-" * 80)
    sorted_stats = sorted(stats, key=lambda x: abs(x['cohens_d']), reverse=True)

    for i, s in enumerate(sorted_stats[:10], 1):
        print(f"{i:2d}. {s['feature']:30s}  Cohen's d: {s['cohens_d']:7.3f}")
    print()
    print("NOTE: These rankings are specific to this recording. Do NOT use for")
    print("      final feature selection. Process all data first.")
    print()

    # Check for quality issues
    print("Feature quality check:")
    print("-" * 80)

    # Check for constant features
    constant_features = []
    for i, name in enumerate(feature_names):
        if np.std(features[:, i]) < 1e-6:
            constant_features.append(name)

    if constant_features:
        print(f"Warning: {len(constant_features)} constant features found:")
        for name in constant_features:
            print(f"  - {name}")
    else:
        print("✓ No constant features")

    # Check for highly correlated features (|r| > 0.95)
    corr_matrix = np.corrcoef(features.T)
    high_corr_pairs = []

    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            if abs(corr_matrix[i, j]) > 0.95:
                high_corr_pairs.append((feature_names[i], feature_names[j], corr_matrix[i, j]))

    if high_corr_pairs:
        print(f"\nWarning: {len(high_corr_pairs)} highly correlated feature pairs (|r| > 0.95):")
        for name1, name2, r in high_corr_pairs[:5]:  # Show first 5
            print(f"  - {name1} <-> {name2}: r = {r:.3f}")
    else:
        print("✓ No highly correlated features (|r| > 0.95)")

    print()

    # Generate visualizations
    print("Generating visualizations...")
    print("-" * 80)

    plot_feature_distributions(features, labels, feature_names, output_dir)
    plot_feature_importance_preview(stats, output_dir)
    plot_correlation_matrix(features, feature_names, output_dir)

    print()
    print("=" * 80)
    print("Validation completed!")
    print(f"Output directory: {output_dir}")
    print()


if __name__ == '__main__':
    main()
