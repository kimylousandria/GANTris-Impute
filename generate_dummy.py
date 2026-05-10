import pandas as pd
import random

# Konfigurasi
NUM_SAMPLES = 2000
SEQ_LENGTH = 15
BASES = ['A', 'T', 'C', 'G']

print("⚙️ Membangun dataset DNA sintetik...")

data = []
for _ in range(NUM_SAMPLES):
    seq = "".join(random.choices(BASES, k=SEQ_LENGTH))
    data.append(seq)

# Simpan ke CSV
df = pd.DataFrame(data, columns=["sequence"])
df.to_csv("dna_dataset.csv", index=False)

print(f"✅ Berhasil! File dna_dataset.csv dengan {NUM_SAMPLES} baris telah dibuat.")