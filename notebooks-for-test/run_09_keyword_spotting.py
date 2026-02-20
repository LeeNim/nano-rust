"""
09 — Keyword Spotting (Voice Recognition via MFCC + MLP)

Scenario: Detect 10 spoken keywords from audio — a core TinyML use case for
          wake-word detection and voice-controlled IoT devices.

Architecture: Dense(416→128) → ReLU → Dense(128→64) → ReLU → Dense(64→10)
Input:  13 MFCC coefficients × 32 time frames = 416 features (flattened)
Output: 10 keywords: yes, no, up, down, left, right, on, off, stop, go

Pipeline:
  1. Download Google Speech Commands v0.02 via torchaudio
  2. Extract MFCC features (13 coefficients, 32 frames)
  3. Train MLP on GPU
  4. Quantize to i8 → verify against NANO-RUST engine
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.utils.data import DataLoader, Dataset
from nano_rust_utils import quantize_to_i8, quantize_weights, calibrate_model
import nano_rust_py

# Target keywords (matches Google Speech Commands "core" set)
KEYWORDS = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
N_MFCC = 13       # Number of MFCC coefficients
N_FRAMES = 32     # Fixed number of time frames after resampling
N_FEATURES = N_MFCC * N_FRAMES  # 416

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
# Step 1: Load & Preprocess Speech Commands Dataset
# ============================================================
# Download dataset manually if not present
import tarfile, urllib.request, soundfile as sf
from pathlib import Path

DATASET_URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
# Try torchaudio's download location first, then custom location
DATASET_DIR = Path("./data/SpeechCommands/speech_commands_v0.02")
if not DATASET_DIR.exists():
    DATASET_DIR = Path("./data/speech_commands_v002")

if not DATASET_DIR.exists():
    archive = Path("./data/speech_commands_v0.02.tar.gz")
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        print("Downloading Speech Commands dataset (~2.3GB)...")
        urllib.request.urlretrieve(DATASET_URL, str(archive))
        print("Download complete.")
    print("Extracting dataset...")
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    with tarfile.open(str(archive), 'r:gz') as tar:
        tar.extractall(str(DATASET_DIR))
    print("Extraction complete.")

# Read validation/test split files
val_list = (DATASET_DIR / "validation_list.txt").read_text().strip().split('\n')
test_list = (DATASET_DIR / "testing_list.txt").read_text().strip().split('\n')
val_set = set(val_list)
test_set = set(test_list)


class KeywordDataset(Dataset):
    """Loads .wav files directly, extracts MFCC features."""

    def __init__(self, subset: str = 'training'):
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=16000,
            n_mfcc=N_MFCC,
            melkwargs={'n_fft': 400, 'hop_length': 160, 'n_mels': 23}
        )
        self.files = []
        self.labels = []

        for keyword in KEYWORDS:
            keyword_dir = DATASET_DIR / keyword
            if not keyword_dir.exists():
                continue
            for wav_file in keyword_dir.glob("*.wav"):
                rel_path = f"{keyword}/{wav_file.name}"
                if subset == 'testing' and rel_path not in test_set:
                    continue
                elif subset == 'validation' and rel_path not in val_set:
                    continue
                elif subset == 'training':
                    if rel_path in test_set or rel_path in val_set:
                        continue
                self.files.append(str(wav_file))
                self.labels.append(KEYWORDS.index(keyword))

        print(f'  {subset}: {len(self.files)} samples')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Load wav with soundfile (avoids torchcodec)
        audio_np, sample_rate = sf.read(self.files[idx], dtype='float32')
        waveform = torch.from_numpy(audio_np).unsqueeze(0)  # [1, samples]

        # Pad/truncate to 1 second (16000 samples)
        if waveform.shape[1] < 16000:
            waveform = torch.nn.functional.pad(waveform, (0, 16000 - waveform.shape[1]))
        else:
            waveform = waveform[:, :16000]

        # Extract MFCC: [1, n_mfcc, time] → [n_mfcc, time]
        mfcc = self.mfcc_transform(waveform).squeeze(0)

        # Resize time dimension to fixed N_FRAMES
        if mfcc.shape[1] != N_FRAMES:
            mfcc = torch.nn.functional.interpolate(
                mfcc.unsqueeze(0), size=N_FRAMES, mode='linear', align_corners=False
            ).squeeze(0)

        # Flatten to 1D: [N_MFCC * N_FRAMES]
        features = mfcc.flatten()

        # Normalize to zero-mean, unit-variance
        features = (features - features.mean()) / (features.std() + 1e-8)

        return features, self.labels[idx]


print('Loading Speech Commands dataset...')
train_ds = KeywordDataset('training')
test_ds = KeywordDataset('testing')

train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, pin_memory=True, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, pin_memory=True, num_workers=0)

# ============================================================
# Step 2: Train MLP on GPU
# ============================================================
model = nn.Sequential(
    nn.Linear(N_FEATURES, 128),  # 416→128
    nn.ReLU(),
    nn.Linear(128, 64),          # 128→64
    nn.ReLU(),
    nn.Linear(64, len(KEYWORDS)),  # 64→10
).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

EPOCHS = 10
t0 = time.time()
for epoch in range(EPOCHS):
    model.train()
    correct, total = 0, 0
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(features)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        correct += out.argmax(1).eq(labels).sum().item()
        total += labels.size(0)
    if (epoch + 1) % 2 == 0:
        print(f'  Epoch {epoch+1}/{EPOCHS} - Train Acc: {100.*correct/total:.1f}%')
train_time = time.time() - t0
print(f'Training time: {train_time:.1f}s')

# ============================================================
# Step 3: PyTorch Test Accuracy
# ============================================================
model.eval()
correct_pt, total_pt = 0, 0
with torch.no_grad():
    for features, labels in test_loader:
        features, labels = features.to(device), labels.to(device)
        correct_pt += model(features).argmax(1).eq(labels).sum().item()
        total_pt += labels.size(0)
pt_acc = 100. * correct_pt / total_pt
print(f'PyTorch Test Accuracy: {pt_acc:.2f}%')

# ============================================================
# Step 4: Quantize & Calibrate (on CPU)
# ============================================================
model_cpu = model.cpu().eval()

q_weights = quantize_weights(model_cpu)

# Use global scale from test features for consistent quantization
# Why: MFCC features are normalized per-sample, but range varies slightly
all_test_features = []
for i in range(min(100, len(test_ds))):
    feat, _ = test_ds[i]
    all_test_features.append(feat.numpy())
all_test_np = np.vstack(all_test_features)
global_max = float(np.max(np.abs(all_test_np)))
global_scale = global_max / 127.0
print(f'Global input scale: {global_scale:.6f} (max_abs={global_max:.4f})')

def quantize_global(data):
    """Quantize using fixed global scale."""
    return np.clip(np.round(data / global_scale), -128, 127).astype(np.int8)

# Calibrate with representative sample
cal_input = test_ds[0][0].unsqueeze(0)
requant = calibrate_model(model_cpu, cal_input, q_weights, global_scale)

# ============================================================
# Step 5: Build NANO-RUST Model & Test
# ============================================================
def build_nano():
    nano = nano_rust_py.PySequentialModel(input_shape=[N_FEATURES], arena_size=16384)
    # Dense(416→128)
    m, s, bc = requant['0']
    nano.add_dense_with_requant(q_weights['0']['weights'].flatten().tolist(), bc, m, s)
    nano.add_relu()
    # Dense(128→64)
    m, s, bc = requant['2']
    nano.add_dense_with_requant(q_weights['2']['weights'].flatten().tolist(), bc, m, s)
    nano.add_relu()
    # Dense(64→10)
    m, s, bc = requant['4']
    nano.add_dense_with_requant(q_weights['4']['weights'].flatten().tolist(), bc, m, s)
    return nano

N_TEST = min(500, len(test_ds))
correct_nano, match_count = 0, 0
max_diffs = []

t0 = time.time()
for i in range(N_TEST):
    feat, label = test_ds[i]
    feat_np = feat.numpy()
    q_feat = quantize_global(feat_np)

    nano_out = build_nano().forward(q_feat.tolist())
    nano_cls = int(np.argmax(nano_out))

    with torch.no_grad():
        pt_out = model_cpu(feat.unsqueeze(0)).numpy().flatten()
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
print('       KEYWORD SPOTTING RESULTS')
print('=' * 60)
print(f'Keywords:             {", ".join(KEYWORDS)}')
print(f'PyTorch Accuracy:     {pt_acc:.2f}%')
print(f'NANO-RUST Accuracy:   {nano_acc:.2f}% (n={N_TEST})')
print(f'Classification Match: {agreement:.1f}%')
print(f'Max Diff (median):    {int(np.median(max_diffs))}')
print(f'Max Diff (95th):      {int(np.percentile(max_diffs, 95))}')
print(f'Max Diff (max):       {max(max_diffs)}')
print(f'Training time (GPU):  {train_time:.1f}s')
print(f'Inference ({N_TEST}): {infer_time:.1f}s')
print('=' * 60)
total_weights = sum(q['weights'].nbytes for q in q_weights.values() if q['weights'] is not None)
print(f'\nMemory: {total_weights} bytes ({total_weights/1024:.1f}KB weights) + 16KB arena')
print(f'Fits ESP32 (520KB RAM)? {"YES" if total_weights + 16384 < 520*1024 else "NO"}')
print(f'{"PASS" if agreement > 85 else "FAIL"}: {agreement:.1f}% agreement')
