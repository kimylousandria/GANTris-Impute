import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os

# ==========================================
# 1. KONFIGURASI GLOBAL & HYPERPARAMETER
# ==========================================
torch.manual_seed(42)
SEQ_LENGTH = 17
BASES = 4
BASE_MAP = {'A': 0, 'T': 1, 'C': 2, 'G': 3}

# ==========================================
# 2. DATASET PUMP (Sinkronisasi 17x4)
# ==========================================
class DNADataset(Dataset):
    def __init__(self, csv_file='dna_dataset.csv'):
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"❌ File {csv_file} tidak ditemukan. Jalankan skrip fetch_data.py terlebih dahulu.")
        
        df = pd.read_csv(csv_file).drop_duplicates()
        self.data = df['sequence'].tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Memaksa panjang absolut 17 karakter untuk keamanan dimensi
        seq = self.data[idx].upper().ljust(SEQ_LENGTH, 'A')
        tensor = torch.zeros(BASES, SEQ_LENGTH)
        for i, base in enumerate(seq[:SEQ_LENGTH]):
            if base in BASE_MAP:
                # Relaxed One-Hot Encoding untuk fleksibilitas gradien
                tensor[:, i] = 0.01 
                tensor[BASE_MAP[base], i] = 0.97
        return tensor

def get_dataloader(batch_size=64):
    dataset = DNADataset('dna_dataset.csv')
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# ==========================================
# 3. ARSITEKTUR KOMPUTASI (SNGAN + Conv1D)
# ==========================================
class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super().__init__()
        self.z_dim = z_dim
        # Transformasi noise laten ke dimensi spasial
        self.fc = nn.Linear(z_dim, 128 * SEQ_LENGTH)
        
        # Jaringan Konvolusi 1D Ramping (Tanpa Attention)
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, BASES, kernel_size=3, padding=1)
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 128, SEQ_LENGTH)
        x = self.conv_blocks(x)
        # Menghasilkan output probabilitas mutlak (jumlah = 1.0) per posisi
        return F.softmax(x, dim=1) 

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        # Seluruh lapisan diisolasi dengan Spectral Normalization
        self.conv_blocks = nn.Sequential(
            spectral_norm(nn.Conv1d(BASES, 64, kernel_size=3, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv1d(64, 128, kernel_size=3, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = spectral_norm(nn.Linear(128 * SEQ_LENGTH, 1))

    def forward(self, x):
        out = self.conv_blocks(x)
        out = out.view(out.size(0), -1)
        # Output Logit mentah tak terbatas (Tanpa Sigmoid)
        return self.fc(out) 

# ==========================================
# 4. MESIN TRAINING & OPTIMASI (TTUR)
# ==========================================
def train_sngan(epochs=1000, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Memulai Eksekusi SNGAN pada prosesor: {device}")

    dataloader = get_dataloader(batch_size)
    z_dim = 100
    
    netG = Generator(z_dim).to(device)
    netC = Critic().to(device)
    
    # Kecepatan belajar asimetris: Critic lebih cepat dari Generator (TTUR)
    optC = optim.Adam(netC.parameters(), lr=0.0004, betas=(0.0, 0.9))
    optG = optim.Adam(netG.parameters(), lr=0.0001, betas=(0.0, 0.9))

    N_CRITIC = 2 

    for epoch in range(epochs):
        for i, real_dna in enumerate(dataloader):
            real_dna = real_dna.to(device)
            b_size = real_dna.size(0)

            # --- FASE 1: EVALUASI CRITIC ---
            optC.zero_grad()
            z = torch.randn(b_size, z_dim, device=device)
            fake_dna = netG(z).detach()
            
            real_validity = netC(real_dna)
            fake_validity = netC(fake_dna)
            
            # Formulasi Matematis Hinge Loss
            loss_C_real = torch.mean(F.relu(1.0 - real_validity))
            loss_C_fake = torch.mean(F.relu(1.0 + fake_validity))
            loss_C = loss_C_real + loss_C_fake
            
            loss_C.backward()
            optC.step()

            # --- FASE 2: EVOLUSI GENERATOR ---
            if i % N_CRITIC == 0:
                optG.zero_grad()
                z = torch.randn(b_size, z_dim, device=device)
                gen_dna = netG(z)
                
                fake_validity_G = netC(gen_dna)
                
                # Generator berusaha mendorong skor Critic menjadi sepositif mungkin
                loss_G = -torch.mean(fake_validity_G)
                
                loss_G.backward()
                optG.step()

        # Pemantauan Evaluasi
        if epoch % 100 == 0:
            print(f"Epoch [{epoch:04d}/{epochs}] | Loss Critic: {loss_C.item():.4f} | Loss Gen: {loss_G.item():.4f}")

    # Ekspor Otak Komputasi
    torch.save(netG.state_dict(), 'sngan_generator_optimal.pth')
    print("✅ Konvergensi Selesai! Model Generator diekspor sebagai 'sngan_generator_optimal.pth'")

if __name__ == "__main__":
    train_sngan(epochs=1000, batch_size=64)