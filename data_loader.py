import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# Konfigurasi Arsitektur
SEQ_LENGTH = 15
BASES = 4
BASE_MAP = {'A': 0, 'T': 1, 'C': 2, 'G': 3}

class DNADataset(Dataset):
    def __init__(self, csv_file="dna_dataset.csv"):
        try:
            self.data = pd.read_csv(csv_file)
            self.sequences = self.data['sequence'].astype(str).values
        except FileNotFoundError:
            raise FileNotFoundError(f"❌ File {csv_file} hilang. Jalankan generate_dummy.py dulu.")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx].upper()
        tensor = torch.zeros(SEQ_LENGTH, BASES)
        
        # Proses One-Hot Encoding On-the-Fly
        for i, char in enumerate(seq[:SEQ_LENGTH]):
            if char in BASE_MAP:
                tensor[i, BASE_MAP[char]] = 1.0
            else:
                tensor[i, :] = 0.25 # Nilai probabilitas rata untuk karakter tak dikenal (N)
                
        return tensor

def get_dataloader(csv_file="dna_dataset.csv", batch_size=32):
    dataset = DNADataset(csv_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)