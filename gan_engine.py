import torch
import torch.nn as nn

SEQ_LENGTH = 17
BASES = 7
INPUT_DIM = SEQ_LENGTH * BASES

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        # Menambah kedalaman (Depth) untuk menangkap pola DNA BRCA1 yang kompleks
        self.main = nn.Sequential(
            nn.Linear(INPUT_DIM, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, INPUT_DIM)
        )

    def forward(self, x):
        b, s, f = x.shape
        flat_x = x.view(b, -1)
        logits = self.main(flat_x)
        out = torch.softmax(logits.view(b, s, f), dim=-1)
        return out

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # Mengimbangi kepintaran Generator dengan lapisan ekstra dan Dropout
        self.main = nn.Sequential(
            nn.Linear(INPUT_DIM, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),  # Mencegah Overfitting pada data kanker
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        b = x.size(0)
        flat_x = x.view(b, -1)
        return self.main(flat_x)