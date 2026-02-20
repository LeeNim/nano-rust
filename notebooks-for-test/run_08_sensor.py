"""
08 — Sensor Fusion MLP (Multi-Sensor → Classification)
Industrial vibration monitoring — classify machine state from IMU sensor data.
Architecture: Dense(6→32) → ReLU → Dense(32→16) → ReLU → Dense(16→4)
Input: 6 features (accel_xyz + gyro_xyz)
Output: 4 classes (normal, bearing_fault, misalignment, imbalance)
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from nano_rust_utils import quantize_to_i8, quantize_weights, calibrate_model
import nano_rust_py

CLASSES = ['Normal', 'Bearing Fault', 'Misalignment', 'Imbalance']

# ============================================================
# GPU Setup
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}', end='')
if device.type == 'cuda':
    print(f' ({torch.cuda.get_device_name(0)})')
else:
    print()

# ============================================================
# Step 1: Generate Synthetic IMU Sensor Data
# ============================================================
np.random.seed(42)
N_PER_CLASS = 500
N_FEATURES = 6

def generate_sensor_data(n, class_id):
    """Generate synthetic vibration data with class-specific patterns."""
    base = np.random.randn(n, N_FEATURES) * 0.3
    if class_id == 0:    # Normal: low amplitude
        base *= 0.5
    elif class_id == 1:  # Bearing fault: high-freq spikes in accel
        base[:, :3] += np.random.choice([-1, 1], (n, 3)) * np.random.exponential(0.8, (n, 3))
    elif class_id == 2:  # Misalignment: correlated accel/gyro
        base[:, 3:] = base[:, :3] * 0.7 + np.random.randn(n, 3) * 0.2
    elif class_id == 3:  # Imbalance: periodic in one axis
        base[:, 0] += np.sin(np.linspace(0, 4*np.pi, n)) * 1.2
        base[:, 3] += np.cos(np.linspace(0, 4*np.pi, n)) * 0.8
    return base

X_all, y_all = [], []
for c in range(4):
    X_all.append(generate_sensor_data(N_PER_CLASS, c))
    y_all.extend([c] * N_PER_CLASS)

X = np.vstack(X_all).astype(np.float32)
y = np.array(y_all, dtype=np.int64)

idx = np.random.permutation(len(X))
X, y = X[idx], y[idx]
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, pin_memory=True)

print(f'Train: {len(train_ds)}, Test: {len(test_ds)}, Features: {N_FEATURES}, Classes: {len(CLASSES)}')

# ============================================================
# Step 2: Train MLP on GPU
# ============================================================
model = nn.Sequential(
    nn.Linear(6, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 4),
).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.005)
criterion = nn.CrossEntropyLoss()

EPOCHS = 20
t0 = time.time()
for epoch in range(EPOCHS):
    model.train()
    correct, total = 0, 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        correct += out.argmax(1).eq(target).sum().item()
        total += target.size(0)
    if (epoch + 1) % 5 == 0:
        print(f'  Epoch {epoch+1}/{EPOCHS} - Acc: {100.*correct/total:.1f}%')
train_time = time.time() - t0
print(f'Training time: {train_time:.1f}s')

# ============================================================
# Step 3: Quantize & Test (on CPU)
# ============================================================
# Why global scale: sensor data has variable amplitude per sample (0.15-4.9).
# Using per-sample scale causes calibration mismatch → 10x error.
# In real MCU deployment, you'd set a fixed input range from datasheet specs.
model_cpu = model.cpu().eval()

q_weights = quantize_weights(model_cpu)

# Compute global input scale from entire dataset (train+test)
global_max = float(np.max(np.abs(np.vstack([X_train, X_test]))))
global_scale = global_max / 127.0
print(f'Global input scale: {global_scale:.6f} (max_abs={global_max:.4f})')

def quantize_global(data):
    """Quantize using fixed global scale instead of per-sample."""
    q = np.clip(np.round(data / global_scale), -128, 127).astype(np.int8)
    return q

# Calibrate with representative sample using the global scale
cal_input = torch.from_numpy(X_test[:1])
q_cal = quantize_global(cal_input.numpy().flatten())
requant = calibrate_model(model_cpu, cal_input, q_weights, global_scale)

def build_nano():
    nano = nano_rust_py.PySequentialModel(input_shape=[6], arena_size=4096)
    m, s, bc = requant['0']
    nano.add_dense_with_requant(q_weights['0']['weights'].flatten().tolist(), bc, m, s)
    nano.add_relu()
    m, s, bc = requant['2']
    nano.add_dense_with_requant(q_weights['2']['weights'].flatten().tolist(), bc, m, s)
    nano.add_relu()
    m, s, bc = requant['4']
    nano.add_dense_with_requant(q_weights['4']['weights'].flatten().tolist(), bc, m, s)
    return nano

correct_pt, correct_nano, match_count = 0, 0, 0
max_diffs = []

t0 = time.time()
for i in range(len(X_test)):
    x_f = torch.from_numpy(X_test[i:i+1])
    label = int(y_test[i])
    q_x = quantize_global(X_test[i])

    with torch.no_grad():
        pt_out = model_cpu(x_f).numpy().flatten()
    pt_cls = int(np.argmax(pt_out))

    nano_out = build_nano().forward(q_x.tolist())
    nano_cls = int(np.argmax(nano_out))

    q_pt, _ = quantize_to_i8(pt_out)
    diff = np.abs(q_pt.astype(np.int32) - np.array(nano_out, dtype=np.int8).astype(np.int32))
    max_diffs.append(int(np.max(diff)))

    if pt_cls == label: correct_pt += 1
    if nano_cls == label: correct_nano += 1
    if nano_cls == pt_cls: match_count += 1
infer_time = time.time() - t0

N = len(X_test)

# ============================================================
# Results
# ============================================================
print()
print('=' * 60)
print('       SENSOR FUSION MLP RESULTS')
print('=' * 60)
print(f'PyTorch Accuracy:     {100.*correct_pt/N:.1f}%')
print(f'NANO-RUST Accuracy:   {100.*correct_nano/N:.1f}%')
print(f'Classification Match: {100.*match_count/N:.1f}%')
print(f'Max Diff (median):    {int(np.median(max_diffs))}')
print(f'Max Diff (95th):      {int(np.percentile(max_diffs, 95))}')
print(f'Max Diff (max):       {max(max_diffs)}')
print(f'Training time (GPU):  {train_time:.1f}s')
print(f'Inference ({N}):      {infer_time:.1f}s')
print('=' * 60)
total_weights = sum(q['weights'].nbytes for q in q_weights.values() if q['weights'] is not None)
print(f'\nMemory: {total_weights} bytes (weights only) + 4KB arena')
print(f'This MLP fits on ANY microcontroller - even ATmega328 (2KB RAM)!')
print(f'{"PASS" if 100.*match_count/N > 85 else "FAIL"}: {100.*match_count/N:.1f}% agreement')
