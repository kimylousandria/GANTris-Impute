import torch
import torch.nn as nn
import torch.optim as optim
from gan_engine import Generator, Discriminator
from data_loader import get_dataloader

def train_gan(epochs=1000, batch_size=32):
    # 1. Device Selection & Initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Memulai Training menggunakan: {device}")

    netG = Generator().to(device)
    netD = Discriminator().to(device)
    
    criterion = nn.BCELoss()
    
    # 2. REVISI ARSITEKTUR: Asymmetric Learning Rates
    # Generator diberi LR standar, Discriminator diperlambat 4x lipat
    # Dampak: Mencegah Discriminator mengenali pola secara instan.
    LR_G = 0.0002
    LR_D = 0.00005 
    
    optimizerG = optim.Adam(netG.parameters(), lr=LR_G, betas=(0.5, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=LR_D, betas=(0.5, 0.999))

    dataloader = get_dataloader(batch_size=batch_size)

    # 3. Training Loop Utama
    for epoch in range(epochs):
        for i, real_dna in enumerate(dataloader):
            real_dna = real_dna.to(device)
            b_size = real_dna.size(0)
            
          
            netD.zero_grad()
            
            # REVISI ARSITEKTUR: Label Smoothing
            # Target tidak lagi absolut 1.0 dan 0.0 untuk menghindari Overconfidence
            label_real = torch.full((b_size, 1), 0.9, device=device) 
            label_fake = torch.full((b_size, 1), 0.1, device=device)
            
            # A. Tes dengan DNA Asli
            output_real = netD(real_dna)
            errD_real = criterion(output_real, label_real)
            
            # B. Tes dengan DNA Palsu buatan Generator
            noise = torch.randn(b_size, 15, 4).to(device)
            fake_dna = netG(noise)
            output_fake = netD(fake_dna.detach())
            errD_fake = criterion(output_fake, label_fake)
            
            # C. Hitung total kesalahan dan update bobot Discriminator
            errD = errD_real + errD_fake
            errD.backward()
            optimizerD.step()

            # ==========================================
            # FASE 2: Melatih Generator (Si Pemalsu)
            # ==========================================
            netG.zero_grad()
            
            # Generator menguji ulang karyanya ke Discriminator yang baru di-update
            output = netD(fake_dna)
            
            # Generator ingin menipu Discriminator agar mengira karyanya asli (0.9)
            errG = criterion(output, label_real) 
            errG.backward()
            optimizerG.step()

        # 4. Monitoring Strategy
        if epoch % 100 == 0:
            print(f"Epoch [{epoch:04d}/{epochs}] | Loss_D: {errD.item():.4f} | Loss_G: {errG.item():.4f}")

    # 5. Ekspor Otak AI
    torch.save(netG.state_dict(), "generator_model.pth")
    print("✅ Training Selesai! Model Generator berhasil disimpan sebagai 'generator_model.pth'")

if __name__ == "__main__":
    # Jalankan simulasi lokal 1000 epoch untuk melihat osilasi Loss
    train_gan(epochs=1000, batch_size=32)