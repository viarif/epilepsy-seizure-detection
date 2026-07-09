import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# 加载数据
data_path = r'E:\summer\data\processed\chb01_03_windows.npz'
data = np.load(data_path)

windows = data['X']
labels = data['y']
channels = data['channels']
sfreq = data['sfreq']

print("=" * 60)
print("File Overview")
print("=" * 60)
print(f"Windows shape: {windows.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Channels: {channels}")
print(f"Sampling rate: {sfreq} Hz")
print(f"\nLabel distribution:")

unique, counts = np.unique(labels, return_counts=True)
for label, count in zip(unique, counts):
    label_name = "Seizure" if label == 1 else "Normal"
    print(f"  {label_name} (label={label}): {count} windows ({count/len(labels)*100:.2f}%)")

# 可视化
fig = plt.figure(figsize=(16, 10))
fig.suptitle(f'CHB01_03 EEG Windows Visualization ({len(windows)} windows total)',
             fontsize=14, fontweight='bold')

# 选择窗口
np.random.seed(42)
seizure_indices = np.where(labels == 1)[0]
normal_indices = np.where(labels == 0)[0]

if len(seizure_indices) > 0:
    selected_seizure = np.random.choice(seizure_indices, min(2, len(seizure_indices)), replace=False)
else:
    selected_seizure = []

selected_normal = np.random.choice(normal_indices, min(2, len(normal_indices)), replace=False)

selected_indices = list(selected_seizure) + list(selected_normal)
titles = ['Seizure Window'] * len(selected_seizure) + ['Normal Window'] * len(selected_normal)

# 每个窗口创建子图，每个通道单独一行
n_windows = len(selected_indices)
n_channels = windows.shape[1]

for win_idx, (window_idx, title) in enumerate(zip(selected_indices, titles)):
    window_data = windows[window_idx]
    time_axis = np.arange(window_data.shape[1]) / sfreq

    for ch_idx in range(n_channels):
        ax = plt.subplot(n_windows, n_channels, win_idx * n_channels + ch_idx + 1)

        # 绘制信号
        signal = window_data[ch_idx, :]
        ax.plot(time_axis, signal, linewidth=0.8)

        # 标题和标签
        if win_idx == 0:
            ax.set_title(f'{channels[ch_idx]}', fontsize=10, fontweight='bold')

        if ch_idx == 0:
            ax.set_ylabel(f'{title}\nWindow {window_idx}\nAmplitude', fontsize=8)
        else:
            ax.set_ylabel('Amplitude', fontsize=8)

        if win_idx == n_windows - 1:
            ax.set_xlabel('Time (s)', fontsize=8)

        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

        # 显示数值范围
        ax.text(0.98, 0.95, f'Range: [{signal.min():.1e}, {signal.max():.1e}]',
                transform=ax.transAxes, fontsize=6, va='top', ha='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()

# 保存图像
output_path = r'E:\summer\data\processed\chb01_03_visualization_fixed.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nVisualization saved to: {output_path}")

plt.show()

print("\nVisualization complete!")
