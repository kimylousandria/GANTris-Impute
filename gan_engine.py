import torch
import torch.nn as nn

# Parameter global berdasarkan data kita (Panjang DNA = 12, Basa = 4)
SEQ_LENGTH = 12
BASES = 4
INPUT_SIZE = SEQ_LENGTH * BASES # 12 x 4 = 48 neuron input

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # Membuat arsitektur Neural Network untuk Generator
        self.model = nn.Sequential(
            nn.Linear(INPUT_SIZE, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, INPUT_SIZE),
            nn.Sigmoid() # Memastikan output probabilitas berada di antara 0 dan 1
        )

    def forward(self, x):
        # Flatten input matriks (12,4) menjadi vektor (48,)
        x = x.view(x.size(0), -1) 
        out = self.model(x)
        # Kembalikan ke bentuk asal (Batch, 12, 4)
        return out.view(x.size(0), SEQ_LENGTH, BASES)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # Membuat arsitektur Neural Network untuk Discriminator
        self.model = nn.Sequential(
            nn.Linear(INPUT_SIZE, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid() # Output 1 (Asli) atau 0 (Palsu)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)

if __name__ == "__main__":
    print("--- Menginisialisasi Mesin GAN ---")
    
    # Membuat objek Generator dan Discriminator
    gen = Generator()
    disc = Discriminator()
    
    # Membuat tensor DNA dummy (batch_size=1, seq_len=12, bases=4)
    dummy_dna = torch.zeros((1, SEQ_LENGTH, BASES))
    
    # Mengetes apakah Generator bisa memproses data
    generated_dna = gen(dummy_dna)
    
    # Mengetes apakah Discriminator bisa menilai data
    judgement = disc(generated_dna)
    
    print("Mesin berhasil dibangun tanpa error!")
    print(f"Bentuk Output Generator: {generated_dna.shape}")
    print(f"Skor Kepalsuan dari Discriminator: {judgement.item():.4f}")