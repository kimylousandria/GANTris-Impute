import torch
import torch.nn as nn
import torch.optim as optim
from gan_engine import Generator, Discriminator
from data_loader import get_loader

def train_gan(epochs=1000, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Memulai Training menggunakan: {device}")

    netG = Generator().to(device)
    netD = Discriminator().to(device)
    criterion = nn.BCELoss()
    
    # Taktik Asymmetric LR: Mempercepat Generator untuk memukul balik Discriminator
    LR_G = 0.0005
    LR_D = 0.0001 
    
    optimizerG = optim.Adam(netG.parameters(), lr=LR_G, betas=(0.5, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=LR_D, betas=(0.5, 0.999))

    dataloader = get_loader(batch_size=batch_size)

    for epoch in range(epochs):
        for i, real_dna in enumerate(dataloader):
            real_dna = real_dna.to(device)
            b_size = real_dna.size(0)
            
            # FASE 1: MELATIH DISCRIMINATOR
            netD.zero_grad()
            
            # Relaxed Label Smoothing untuk meredam overconfidence Discriminator
            label_real = torch.full((b_size, 1), 0.8, device=device) 
            label_fake = torch.full((b_size, 1), 0.1, device=device)
            
            output_real = netD(real_dna)
            errD_real = criterion(output_real, label_real)
            
            # Wajib menggunakan dimensi 17 untuk sinkronisasi tensor
            noise = torch.randn(b_size, 17, 4).to(device)
            fake_dna = netG(noise)
            output_fake = netD(fake_dna.detach())
            errD_fake = criterion(output_fake, label_fake)
            
            errD = errD_real + errD_fake
            errD.backward()
            optimizerD.step()

            # FASE 2: MELATIH GENERATOR
            netG.zero_grad()
            output = netD(fake_dna)
            
            errG = criterion(output, label_real) 
            errG.backward()
            optimizerG.step()

        if epoch % 100 == 0:
            print(f"Epoch [{epoch:04d}/{epochs}] | Loss_D: {errD.item():.4f} | Loss_G: {errG.item():.4f}")

    torch.save(netG.state_dict(), "generator_model.pth")
    print("✅ Training Selesai! Model Generator berhasil disimpan sebagai 'generator_model.pth'")

if __name__ == "__main__":
    train_gan(epochs=1000, batch_size=32)