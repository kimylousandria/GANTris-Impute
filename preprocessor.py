import numpy as np
from Bio import SeqIO

def dna_to_onehot(sequence):
    """Mengubah string DNA menjadi matriks One-Hot Encoding."""
    # Kamus pemetaan basa nitrogen ke angka
    mapping = {
        'A': [1, 0, 0, 0],
        'T': [0, 1, 0, 0],
        'C': [0, 0, 1, 0],
        'G': [0, 0, 0, 1],
        'N': [0, 0, 0, 0], # Data hilang (Missing Value)
        '-': [0, 0, 0, 0]  # Data gap
    }
    
    # Looping setiap huruf di sekuens dan ubah jadi array
    one_hot = [mapping.get(base.upper(), [0, 0, 0, 0]) for base in sequence]
    return np.array(one_hot)

def process_fasta(file_path):
    """Membaca file .fasta dan mengubah isinya menjadi matriks."""
    processed_data = {}
    try:
        # Membaca file menggunakan Biopython
        for record in SeqIO.parse(file_path, "fasta"):
            seq_id = record.id
            sequence = str(record.seq)
            
            # Konversi ke One-Hot
            one_hot_matrix = dna_to_onehot(sequence)
            processed_data[seq_id] = one_hot_matrix
            
            print(f"Berhasil memproses: {seq_id} | Panjang: {len(sequence)} bp")
            
        return processed_data
    
    except FileNotFoundError:
        print(f"Error: File {file_path} tidak ditemukan. Cek kembali path foldernya.")
        return None

# Blok testing


if __name__ == "__main__":
    print("--- Memulai Uji Coba Preprocessor ---")
    data = process_fasta("sample.fasta")
    
    if data:
        for seq_id, matrix in data.items():
            print(f"\nID: {seq_id}")
            print("Matriks One-Hot (5 baris pertama):")
            print(matrix[:5]) # Menampilkan 5 basa pertama
            print(f"Shape Matriks: {matrix.shape}") # Harus (panjang_dna, 4)