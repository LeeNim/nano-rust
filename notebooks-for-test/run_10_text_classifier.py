"""
10 — Text Classification (Sentiment Analysis via Bag-of-Words + MLP)

Scenario: Classify text into 5 categories using fixed-size feature vectors.
           Demonstrates NLP on TinyML — realistic for on-device text classification
           (e.g., command parsing, alert classification on ESP32).

Architecture: Dense(200→128) → ReLU → Dense(128→64) → ReLU → Dense(64→5)
Input:  200-dim bag-of-words vector (top-200 vocabulary)
Output: 5 categories from AG News dataset subset

Pipeline:
  1. Load AG News dataset (4 categories) via torchtext or built-in
  2. Build bag-of-words features (top-200 words)
  3. Train MLP on GPU
  4. Quantize to i8 → verify against NANO-RUST engine
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
from collections import Counter
import re

CATEGORIES = ['World', 'Sports', 'Business', 'Sci/Tech']
VOCAB_SIZE = 200   # Fixed feature vector size
N_CLASSES = len(CATEGORIES)

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
# Step 1: Generate / Load Text Data
# ============================================================
# Using synthetic text data that mimics real-world patterns.
# Why synthetic: Avoids large dataset downloads while demonstrating
# the exact same quantization + inference pipeline.
# In production, replace with real AG News / custom dataset.

np.random.seed(42)

# Word pools per category — realistic vocabulary distributions
WORD_POOLS = {
    0: ['war', 'peace', 'president', 'election', 'government', 'minister',
        'country', 'treaty', 'united', 'nations', 'policy', 'crisis',
        'diplomacy', 'summit', 'conflict', 'border', 'refugee', 'military',
        'security', 'alliance', 'vote', 'parliament', 'democracy', 'law'],
    1: ['game', 'team', 'player', 'score', 'win', 'match', 'champion',
        'league', 'season', 'goal', 'coach', 'tournament', 'final',
        'record', 'olympic', 'medal', 'race', 'training', 'stadium',
        'football', 'basketball', 'tennis', 'soccer', 'athlete'],
    2: ['market', 'stock', 'price', 'company', 'profit', 'revenue',
        'growth', 'economy', 'trade', 'investment', 'bank', 'finance',
        'quarter', 'earnings', 'share', 'billion', 'dollar', 'ceo',
        'merger', 'acquisition', 'startup', 'venture', 'inflation', 'tax'],
    3: ['software', 'computer', 'data', 'internet', 'technology', 'research',
        'science', 'algorithm', 'network', 'digital', 'system', 'device',
        'robot', 'artificial', 'intelligence', 'quantum', 'chip', 'cloud',
        'cyber', 'innovation', 'machine', 'learning', 'neural', 'genome'],
}

# Common neutral words (shared across categories)
COMMON_WORDS = ['the', 'is', 'was', 'are', 'been', 'have', 'had', 'will',
                'said', 'new', 'year', 'first', 'also', 'would', 'could',
                'after', 'more', 'about', 'between', 'has', 'their', 'from',
                'other', 'been', 'made', 'world', 'time', 'just', 'most']

def generate_text(n_per_class, n_words_range=(20, 50)):
    """Generate synthetic text samples with category-specific vocabulary."""
    texts, labels = [], []
    for c in range(N_CLASSES):
        pool = WORD_POOLS[c]
        for _ in range(n_per_class):
            n_words = np.random.randint(*n_words_range)
            # Mix: 60% category words + 40% common words
            n_cat = int(n_words * 0.6)
            n_common = n_words - n_cat
            words = (list(np.random.choice(pool, n_cat, replace=True)) +
                     list(np.random.choice(COMMON_WORDS, n_common, replace=True)))
            np.random.shuffle(words)
            texts.append(' '.join(words))
            labels.append(c)
    return texts, labels

# Generate datasets
train_texts, train_labels = generate_text(1000)
test_texts, test_labels = generate_text(200)

# Build vocabulary from training data
word_counts = Counter()
for text in train_texts:
    word_counts.update(text.lower().split())
vocab = [w for w, _ in word_counts.most_common(VOCAB_SIZE)]
word2idx = {w: i for i, w in enumerate(vocab)}
print(f'Vocabulary size: {len(vocab)} words')
print(f'Train: {len(train_texts)}, Test: {len(test_texts)}')

def text_to_bow(text):
    """Convert text to bag-of-words vector."""
    bow = np.zeros(VOCAB_SIZE, dtype=np.float32)
    words = text.lower().split()
    for w in words:
        if w in word2idx:
            bow[word2idx[w]] += 1
    # Normalize by document length
    if bow.sum() > 0:
        bow /= bow.sum()
    return bow

# Convert to feature matrices
X_train = np.array([text_to_bow(t) for t in train_texts])
X_test = np.array([text_to_bow(t) for t in test_texts])
y_train = np.array(train_labels, dtype=np.int64)
y_test = np.array(test_labels, dtype=np.int64)

# Shuffle training data
idx = np.random.permutation(len(X_train))
X_train, y_train = X_train[idx], y_train[idx]

train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
test_ds_torch = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_ds_torch, batch_size=256, pin_memory=True)

# ============================================================
# Step 2: Train MLP on GPU
# ============================================================
model = nn.Sequential(
    nn.Linear(VOCAB_SIZE, 128),    # 200→128
    nn.ReLU(),
    nn.Linear(128, 64),            # 128→64
    nn.ReLU(),
    nn.Linear(64, N_CLASSES),      # 64→4
).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.002)
criterion = nn.CrossEntropyLoss()

EPOCHS = 20
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
    if (epoch + 1) % 5 == 0:
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

# Global scale from all test features
global_max = float(np.max(np.abs(np.vstack([X_train, X_test]))))
global_scale = global_max / 127.0
print(f'Global input scale: {global_scale:.6f} (max_abs={global_max:.4f})')

def quantize_global(data):
    """Quantize using fixed global scale."""
    return np.clip(np.round(data / global_scale), -128, 127).astype(np.int8)

cal_input = torch.from_numpy(X_test[:1])
requant = calibrate_model(model_cpu, cal_input, q_weights, global_scale)

# ============================================================
# Step 5: NANO-RUST Test
# ============================================================
def build_nano():
    nano = nano_rust_py.PySequentialModel(input_shape=[VOCAB_SIZE], arena_size=8192)
    m, s, bc = requant['0']
    nano.add_dense_with_requant(q_weights['0']['weights'].flatten().tolist(), bc, m, s)
    nano.add_relu()
    m, s, bc = requant['2']
    nano.add_dense_with_requant(q_weights['2']['weights'].flatten().tolist(), bc, m, s)
    nano.add_relu()
    m, s, bc = requant['4']
    nano.add_dense_with_requant(q_weights['4']['weights'].flatten().tolist(), bc, m, s)
    return nano

N_TEST = len(X_test)
correct_nano, match_count = 0, 0
max_diffs = []

t0 = time.time()
for i in range(N_TEST):
    q_feat = quantize_global(X_test[i])
    label = int(y_test[i])

    nano_out = build_nano().forward(q_feat.tolist())
    nano_cls = int(np.argmax(nano_out))

    with torch.no_grad():
        pt_out = model_cpu(torch.from_numpy(X_test[i:i+1])).numpy().flatten()
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
print('       TEXT CLASSIFICATION RESULTS')
print('=' * 60)
print(f'Categories:           {", ".join(CATEGORIES)}')
print(f'Vocab size:           {VOCAB_SIZE} words')
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
print(f'\nMemory: {total_weights} bytes ({total_weights/1024:.1f}KB weights) + 8KB arena')
print(f'Fits ESP32 (520KB)? {"YES" if total_weights + 8192 < 520*1024 else "NO"}')
print(f'{"PASS" if agreement > 85 else "FAIL"}: {agreement:.1f}% agreement')
