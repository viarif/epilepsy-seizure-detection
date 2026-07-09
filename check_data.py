import numpy as np

# 加载数据
data_path = r'E:\summer\data\processed\chb01_03_windows.npz'
data = np.load(data_path)

windows = data['X']
labels = data['y']

print("=" * 60)
print("Data Statistics Check")
print("=" * 60)

# 检查几个窗口的统计信息
print(f"\nWindow shape: {windows.shape}")
print(f"Data type: {windows.dtype}")

# 检查整个数据集的统计
print(f"\nOverall statistics:")
print(f"  Min: {windows.min():.6f}")
print(f"  Max: {windows.max():.6f}")
print(f"  Mean: {windows.mean():.6f}")
print(f"  Std: {windows.std():.6f}")

# 检查第一个窗口的详细信息
print(f"\nFirst window (index 0):")
for ch in range(windows.shape[1]):
    print(f"  Channel {ch}: min={windows[0, ch, :].min():.6f}, "
          f"max={windows[0, ch, :].max():.6f}, "
          f"mean={windows[0, ch, :].mean():.6f}, "
          f"std={windows[0, ch, :].std():.6f}")

# 检查一个发作窗口
seizure_idx = np.where(labels == 1)[0]
if len(seizure_idx) > 0:
    idx = seizure_idx[0]
    print(f"\nFirst seizure window (index {idx}):")
    for ch in range(windows.shape[1]):
        print(f"  Channel {ch}: min={windows[idx, ch, :].min():.6f}, "
              f"max={windows[idx, ch, :].max():.6f}, "
              f"mean={windows[idx, ch, :].mean():.6f}, "
              f"std={windows[idx, ch, :].std():.6f}")

# 打印前10个样本值
print(f"\nFirst 10 samples of window 0, channel 0:")
print(windows[0, 0, :10])
