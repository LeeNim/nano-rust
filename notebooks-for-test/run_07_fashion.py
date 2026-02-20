"""
07 — Fashion-MNIST Classification with GPU Training
Same CNN as MNIST but on fashion items (10 categories).
Categories: T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Boot
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

CLASSES = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot']

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
# Step 1: Load Fashion-MNIST
# ============================================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])
train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=256, shuffle=True, pin_memory=True, num_workers=0)
test_loader = DataLoader(test_data, batch_size=1000, pin_memory=True, num_workers=0)
print(f'Train: {len(train_data)}, Test: {len(test_data)}')

# ============================================================
# Step 2: Train CNN on GPU
# ============================================================
model = nn.Sequential(
    nn.Conv2d(1, 8, 3, 1, 1),  nn.ReLU(), nn.MaxPool2d(2, 2),
    nn.Conv2d(8, 16, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
    nn.Flatten(),
    nn.Linear(16*7*7, 10),
).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

EPOCHS = 5
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
pt_acc = 100. * correct_pt / total_pt
print(f'PyTorch Test Accuracy: {pt_acc:.2f}%')

# ============================================================
# Step 4: Quantize & Calibrate (on CPU)
# ============================================================
model_cpu = model.cpu().eval()

q_weights = quantize_weights(model_cpu)
cal_img = test_data[0][0].unsqueeze(0)
q_cal, cal_scale = quantize_to_i8(cal_img.numpy().flatten())
requant = calibrate_model(model_cpu, cal_img, q_weights, cal_scale)

# ============================================================
# Step 5: NANO-RUST Test
# ============================================================
def build_nano():
    nano = nano_rust_py.PySequentialModel(input_shape=[1, 28, 28], arena_size=131072)
    m, s, bc = requant['0']
    nano.add_conv2d_with_requant(q_weights['0']['weights'].flatten().tolist(), bc, 1, 8, 3, 3, 1, 1, m, s)
    nano.add_relu()
    nano.add_max_pool2d(2, 2, 2)
    m, s, bc = requant['3']
    nano.add_conv2d_with_requant(q_weights['3']['weights'].flatten().tolist(), bc, 8, 16, 3, 3, 1, 1, m, s)
    nano.add_relu()
    nano.add_max_pool2d(2, 2, 2)
    nano.add_flatten()
    m, s, bc = requant['7']
    nano.add_dense_with_requant(q_weights['7']['weights'].flatten().tolist(), bc, m, s)
    return nano

N_TEST = 1000
correct_nano, match_count = 0, 0
max_diffs = []

t0 = time.time()
for i in range(N_TEST):
    img, label = test_data[i]
    q_img, _ = quantize_to_i8(img.numpy().flatten())
    nano_out = build_nano().forward(q_img.tolist())
    nano_cls = int(np.argmax(nano_out))

    with torch.no_grad():
        pt_out = model_cpu(img.unsqueeze(0)).numpy().flatten()
    pt_cls = int(np.argmax(pt_out))

    q_pt, _ = quantize_to_i8(pt_out)
    diff = np.abs(q_pt.astype(np.int32) - np.array(nano_out, dtype=np.int8).astype(np.int32))
    max_diffs.append(int(np.max(diff)))

    if nano_cls == label: correct_nano += 1
    if nano_cls == pt_cls: match_count += 1
infer_time = time.time() - t0

nano_acc = 100. * correct_nano / N_TEST
agreement = 100. * match_count / N_TEST

# ============================================================
# Results
# ============================================================
print()
print('=' * 60)
print('       FASHION-MNIST RESULTS')
print('=' * 60)
print(f'PyTorch Accuracy:     {pt_acc:.2f}%')
print(f'NANO-RUST Accuracy:   {nano_acc:.2f}% (n={N_TEST})')
print(f'Classification Match: {agreement:.1f}%')
print(f'Max Diff (median):    {int(np.median(max_diffs))}')
print(f'Max Diff (95th):      {int(np.percentile(max_diffs, 95))}')
print(f'Max Diff (max):       {max(max_diffs)}')
print(f'Training time (GPU):  {train_time:.1f}s')
print(f'Inference ({N_TEST}): {infer_time:.1f}s')
print('=' * 60)
print(f'{"PASS" if agreement > 85 else "FAIL"}: {agreement:.1f}% agreement')
