"""
06 — MNIST Digit Classification with GPU Training
Train CNN on MNIST → Quantize to i8 → Run inference in NANO-RUST
Architecture: Conv2D(1→8) → ReLU → Pool → Conv2D(8→16) → ReLU → Pool → Flatten → Dense(784→10)
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from nano_rust_utils import quantize_to_i8, quantize_weights, calibrate_model
import nano_rust_py

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
# Step 1: Load MNIST
# ============================================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, pin_memory=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, pin_memory=True, num_workers=0)
print(f'Train: {len(train_dataset)}, Test: {len(test_dataset)}')

# ============================================================
# Step 2: Define & Train CNN (on GPU)
# ============================================================
model = nn.Sequential(
    nn.Conv2d(1, 8, 3, 1, 1),   # [1,28,28]→[8,28,28]
    nn.ReLU(),
    nn.MaxPool2d(2, 2),          # →[8,14,14]
    nn.Conv2d(8, 16, 3, 1, 1),  # →[16,14,14]
    nn.ReLU(),
    nn.MaxPool2d(2, 2),          # →[16,7,7]
    nn.Flatten(),                # →[784]
    nn.Linear(16 * 7 * 7, 10),
).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

EPOCHS = 3
t0 = time.time()
for epoch in range(EPOCHS):
    model.train()
    correct, total = 0, 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        correct += output.argmax(1).eq(target).sum().item()
        total += target.size(0)
    print(f'  Epoch {epoch+1}/{EPOCHS} - Acc: {100.*correct/total:.1f}%')
train_time = time.time() - t0
print(f'Training time: {train_time:.1f}s')

# ============================================================
# Step 3: PyTorch Test Accuracy
# ============================================================
model.eval()
correct_pt, total_pt = 0, 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        correct_pt += model(data).argmax(1).eq(target).sum().item()
        total_pt += target.size(0)
pt_accuracy = 100. * correct_pt / total_pt
print(f'PyTorch Test Accuracy: {pt_accuracy:.2f}%')

# ============================================================
# Step 4: Quantize & Calibrate (on CPU)
# ============================================================
model_cpu = model.cpu().eval()  # Move to CPU for quantization

q_weights = quantize_weights(model_cpu)
cal_image = test_dataset[0][0].unsqueeze(0)  # [1, 1, 28, 28]
q_cal, cal_scale = quantize_to_i8(cal_image.numpy().flatten())
requant = calibrate_model(model_cpu, cal_image, q_weights, cal_scale)

print(f'Quantized layers: {list(q_weights.keys())}')
for name, info in q_weights.items():
    if info['weights'] is not None:
        print(f'  {name} ({info["type"]}): {info["weights"].shape}, scale={info["weight_scale"]:.6f}')

# ============================================================
# Step 5: Build NANO-RUST model & Test
# ============================================================
def build_nano_model():
    nano = nano_rust_py.PySequentialModel(input_shape=[1, 28, 28], arena_size=131072)
    # Conv2d(1→8, 3x3, stride=1, pad=1)
    m, s, bc = requant['0']
    nano.add_conv2d_with_requant(
        q_weights['0']['weights'].flatten().tolist(), bc, 1, 8, 3, 3, 1, 1, m, s)
    nano.add_relu()
    nano.add_max_pool2d(2, 2, 2)
    # Conv2d(8→16, 3x3, stride=1, pad=1)
    m, s, bc = requant['3']
    nano.add_conv2d_with_requant(
        q_weights['3']['weights'].flatten().tolist(), bc, 8, 16, 3, 3, 1, 1, m, s)
    nano.add_relu()
    nano.add_max_pool2d(2, 2, 2)
    # Flatten + Dense(784→10)
    nano.add_flatten()
    m, s, bc = requant['7']
    nano.add_dense_with_requant(
        q_weights['7']['weights'].flatten().tolist(), bc, m, s)
    return nano

N_TEST = min(1000, len(test_dataset))
correct_nano, match_count = 0, 0
max_diffs = []

t0 = time.time()
for i in range(N_TEST):
    image, label = test_dataset[i]
    q_image, _ = quantize_to_i8(image.numpy().flatten())

    nano_out = build_nano_model().forward(q_image.tolist())
    nano_cls = int(np.argmax(nano_out))

    with torch.no_grad():
        pt_out = model_cpu(image.unsqueeze(0)).numpy().flatten()
    pt_cls = int(np.argmax(pt_out))

    q_pt, _ = quantize_to_i8(pt_out)
    diff = np.abs(q_pt.astype(np.int32) - np.array(nano_out, dtype=np.int8).astype(np.int32))
    max_diffs.append(int(np.max(diff)))

    if nano_cls == label: correct_nano += 1
    if nano_cls == pt_cls: match_count += 1
infer_time = time.time() - t0

nano_accuracy = 100. * correct_nano / N_TEST
agreement = 100. * match_count / N_TEST

# ============================================================
# Results
# ============================================================
print()
print('=' * 60)
print('       MNIST CLASSIFICATION RESULTS')
print('=' * 60)
print(f'PyTorch Accuracy:      {pt_accuracy:.2f}%')
print(f'NANO-RUST Accuracy:    {nano_accuracy:.2f}% (on {N_TEST} samples)')
print(f'Classification Match:  {agreement:.1f}%')
print(f'Max i8 Diff (median):  {int(np.median(max_diffs))}')
print(f'Max i8 Diff (95th):    {int(np.percentile(max_diffs, 95))}')
print(f'Max i8 Diff (max):     {max(max_diffs)}')
print(f'Training time (GPU):   {train_time:.1f}s')
print(f'Inference time ({N_TEST} samples): {infer_time:.1f}s')
print('=' * 60)
print(f'{"PASS" if agreement > 90 else "FAIL"}: {agreement:.1f}% classification agreement')

# Memory Budget
total_flash = 0
for name, info in q_weights.items():
    if info['weights'] is not None:
        w_bytes = info['weights'].nbytes
        b_bytes = len(requant[name][2]) if name in requant and not isinstance(requant[name][0], str) else 0
        total_flash += w_bytes + b_bytes
arena_est = 32768
print(f'\nMemory: Flash={total_flash} bytes ({total_flash/1024:.1f}KB), Arena={arena_est/1024:.0f}KB')
