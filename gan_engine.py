import torch
import torch.nn as nn

# Hyperparameters
SEQ_LENGTH = 15
BASES = 4

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(SEQ_LENGTH * BASES, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, SEQ_LENGTH * BASES),
            nn.Softmax(dim=-1) # Output berupa probabilitas 0-1
        )

    def forward(self, x):
        # x shape: [batch, seq, bases]
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        out = self.main(x)
        return out.view(batch_size, SEQ_LENGTH, BASES)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(SEQ_LENGTH * BASES, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.main(x)