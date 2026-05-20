import streamlit as st
import torch
import numpy as np
import time

# ─────────────────────────────────────────────
#  PARAM & MODEL
# ─────────────────────────────────────────────
SEQ_LENGTH = 17
BASES = 7

class Generator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.main = torch.nn.Sequential(
            torch.nn.Linear(SEQ_LENGTH * BASES, 256),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(256, 128),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(128, SEQ_LENGTH * BASES)
        )
        
    def forward(self, x):
        b, s, f = x.shape
        logits = self.main(x.view(b, -1)).view(b, s, f)
        return torch.softmax(logits, dim=-1)

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="GANTris-Impute",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  LOAD GLOBAL CSS
# ─────────────────────────────────────────────
try:
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    st.error("⚠️ File 'style.css' tidak ditemukan!")

# ─────────────────────────────────────────────
#  SESSION STATE — inisialisasi di awal
# ─────────────────────────────────────────────
if "dna_input" not in st.session_state:
    st.session_state["dna_input"] = "ATGACNNNTGCN"
if "has_run" not in st.session_state:
    st.session_state["has_run"] = False
if "pred_result" not in st.session_state:
    st.session_state["pred_result"] = None
if "clean_seq_result" not in st.session_state:
    st.session_state["clean_seq_result"] = None

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
BASES_MAP = {0: 'A', 1: 'T', 2: 'C', 3: 'G'}
COLOR_MAP = {'A': '#f43f5e', 'T': '#38bdf8', 'C': '#34d399', 'G': '#fbbf24'}

def colorize_dna(seq: str) -> str:
    html = ""
    for ch in seq.upper():
        if ch in COLOR_MAP:
            html += f'<span class="b{ch}">{ch}</span>'
        elif ch == 'N':
            html += f'<span class="bN">N</span>'
        else:
            html += f'<span style="color:#475569">{ch}</span>'
    return html

def make_ticker(repeat=4):
    template = "ATGCNNATCGGCATNNNATCGCATGNNATCGCATG──"
    colored  = colorize_dna(template) * repeat
    return f'<div class="dna-ticker"><div class="dna-ticker-inner">{colored}</div></div>'

def encode_input(seq: str) -> torch.Tensor:
    base_to_idx = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
    tensor = torch.zeros(1, SEQ_LENGTH, BASES)
    for i, ch in enumerate(seq.upper()[:SEQ_LENGTH]):
        if ch in base_to_idx:
            tensor[0, i, base_to_idx[ch]] = 1.0
        else:
            tensor[0, i, :] = 0.25
    return tensor

@st.cache_resource
def load_model():
    return Generator()


#  SIDEBAR
with st.sidebar:
    st.markdown('<p class="hero-title" style="font-size:20px;">🧬 GANTris</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">IMPUTATION ENGINE v2.0</p>', unsafe_allow_html=True)
    st.divider()

    st.markdown('<div class="section-label">⬡ BASE LEGEND</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="legend-row">
        <div class="legend-item"><div class="legend-dot" style="background:#f43f5e"></div><span>A — Adenin</span></div>
        <div class="legend-item"><div class="legend-dot" style="background:#38bdf8"></div><span>T — Timin</span></div>
        <div class="legend-item"><div class="legend-dot" style="background:#34d399"></div><span>C — Sitosin</span></div>
        <div class="legend-item"><div class="legend-dot" style="background:#fbbf24"></div><span>G — Guanin</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="section-label">⚙ PENGATURAN</div>', unsafe_allow_html=True)
    show_matrix = st.toggle("Tampilkan Matriks Probabilitas", value=True)
    show_stats  = st.toggle("Tampilkan Statistik", value=True)
    conf_filter = st.slider("Filter Keyakinan Minimum (%)", 0, 100, 0)

    st.divider()
    # Tombol reset di sidebar
    if st.button("🔄 Reset Input", use_container_width=True):
        st.session_state["dna_input"]       = "ATGACNNNTGCN"
        st.session_state["has_run"]         = False
        st.session_state["pred_result"]     = None
        st.session_state["clean_seq_result"] = None
        st.rerun()

# ─────────────────────────────────────────────
#  MAIN HEADER
# ─────────────────────────────────────────────
st.markdown(make_ticker(), unsafe_allow_html=True)

col_title, _ = st.columns([3, 1])
with col_title:
    st.markdown('<h1 class="hero-title">🧬 GANTris-Impute</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">// GENOMIC TENSOR VISUALIZER — AI-POWERED DNA IMPUTATION</p>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  INPUT PANEL
# ─────────────────────────────────────────────
col_in, col_out = st.columns([1, 2], gap="large")

with col_in:
    st.markdown('<div class="section-label">▶ INPUT SEKUENS DNA</div>', unsafe_allow_html=True)

    
    raw_input = st.text_input(
        label="Ketik sekuens DNA (gunakan N untuk basa yang hilang)",
        max_chars=SEQ_LENGTH,
        key="dna_input",          # ← INI KUNCINYA
        placeholder="cth: ATGCNNNNATCG",
        help=f"Masukkan kombinasi A, T, C, G, dan N. Maksimal {SEQ_LENGTH} karakter.",
    )

    clean_seq = "".join(c for c in raw_input.upper() if c in "ATCGN")
    clean_seq = clean_seq[:SEQ_LENGTH].ljust(SEQ_LENGTH, 'N')

    n_known   = sum(1 for c in clean_seq if c != 'N')
    n_missing = SEQ_LENGTH - n_known

    # Preview sekuens berwarna (update real-time setiap ketikan)
    st.markdown(f'<div class="dna-input-box">{colorize_dna(clean_seq)}</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    c1.metric("🔬 Diketahui", n_known)
    c2.metric("❓ Hilang (N)", n_missing)

    # Validasi: tidak boleh semua N
    all_n = n_known == 0
    if all_n:
        st.warning("⚠️ Sekuens tidak boleh sepenuhnya kosong (semua N).")

    st.markdown("<br>", unsafe_allow_html=True)

    run_btn = st.button(
        "🚀 JALANKAN IMPUTASI",
        use_container_width=True,
        type="primary",
        disabled=all_n,    # nonaktifkan jika semua N
    )

    # Saat tombol diklik: jalankan model dan simpan hasil ke session_state
    if run_btn:
        generator = load_model()
        with st.spinner("⚡ Memproses tensor BRCA1..."):
            time.sleep(0.4)
            inp  = encode_input(clean_seq)
            pred = generator(inp).detach().numpy()[0]

            # --- PENAMBAHAN LOGIKA MASKING ---
            # 1. Matikan probabilitas untuk indeks 4 (N), 5 (R), 6 (Y)
            pred[:, 4:] = 0.0
            
            # 2. Re-normalisasi probabilitas A, T, C, G agar totalnya kembali 1.0 (100%)
            row_sums = pred.sum(axis=1, keepdims=True)
            pred = np.divide(pred, row_sums, out=np.zeros_like(pred), where=row_sums!=0)
            # ---------------------------------
        # Simpan hasil agar tidak hilang saat widget lain berinteraksi
        st.session_state["pred_result"]      = pred
        st.session_state["clean_seq_result"] = clean_seq
        st.session_state["has_run"]          = True

#output

with col_out:
    st.markdown('<div class="section-label">◀ HASIL IMPUTASI GAN</div>', unsafe_allow_html=True)

    # Ambil hasil dari session_state (bukan dari run_btn langsung)
    pred       = st.session_state["pred_result"]
    clean_used = st.session_state["clean_seq_result"]
    has_run    = st.session_state["has_run"]

    if not has_run or pred is None:
        st.markdown("""
        <div style="background:#0d1117; border:1px dashed #1f2937; border-radius:12px;
                    padding:60px 20px; text-align:center; color:#334155;">
            <div style="font-family:'Orbitron',monospace; font-size:32px; margin-bottom:12px;">◈</div>
            <div style="font-family:'Share Tech Mono',monospace; font-size:13px; letter-spacing:2px;">
                MENUNGGU INPUT<br>← Masukkan sekuens dan klik tombol
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        best_idxs   = np.argmax(pred, axis=1)
        conf_scores = np.max(pred, axis=1) * 100

        # Tetris Blocks
        blocks_html = '<div class="block-grid">'
        for i in range(SEQ_LENGTH):
            is_imputed = clean_used[i] == 'N'
            
            if not is_imputed:
                continue
            
            # Jika lolos filter (artinya ini adalah 'N'), jalankan render:
            base       = BASES_MAP[best_idxs[i]]
            conf       = conf_scores[i]
            delay      = i * 0.04
            badge      = '★ BARU'      # Statis, karena pasti hasil imputasi
            badge_col  = '#fbbf24'     # Statis, warna kuning peringatan
            opacity    = 1.0 if conf >= conf_filter else 0.2


            blocks_html += f"""
            <div class="tblock tblock-{base}" style="animation-delay:{delay}s; opacity:{opacity};">
                <div style="font-family:'Share Tech Mono',monospace;font-size:9px;color:rgba(255,255,255,0.4);margin-bottom:4px;">POS {i+1:02d}</div>
                <div class="tblock-letter">{base}</div>
                <div class="tblock-conf">{conf:.0f}%</div>
                <div class="conf-bar"><div class="conf-fill" style="width:{conf:.0f}%"></div></div>
                <div style="font-family:'Share Tech Mono',monospace;font-size:9px;color:{badge_col};margin-top:6px;letter-spacing:1px;">{badge}</div>
            </div>
            """
        blocks_html += '</div>'

        with open("style.css") as f:
            css_content = f.read()

        full_html = f"<style>{css_content}</style>{blocks_html}"
        st.components.v1.html(full_html, height=500, scrolling=True)

        #  Statistik 
        if show_stats:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-label">▷ STATISTIK</div>', unsafe_allow_html=True)
            s1, s2, s3, s4 = st.columns(4)

            result_seq  = "".join(BASES_MAP[i] for i in best_idxs)
            avg_conf    = np.mean(conf_scores)
            min_conf    = np.min(conf_scores)
            imputed_c   = conf_scores[np.array(list(clean_used)) == 'N']
            avg_imp     = np.mean(imputed_c) if len(imputed_c) > 0 else 0
            base_counts = {b: result_seq.count(b) for b in "ATCG"}
            dominant    = max(base_counts, key=base_counts.get)

            s1.markdown(f'<div class="stat-card"><div class="stat-value">{avg_conf:.0f}%</div><div class="stat-label">Avg Keyakinan</div></div>', unsafe_allow_html=True)
            s2.markdown(f'<div class="stat-card"><div class="stat-value">{min_conf:.0f}%</div><div class="stat-label">Min Keyakinan</div></div>', unsafe_allow_html=True)
            s3.markdown(f'<div class="stat-card"><div class="stat-value">{avg_imp:.0f}%</div><div class="stat-label">Conf Imputasi</div></div>', unsafe_allow_html=True)
            s4.markdown(f'<div class="stat-card" style="border-color:{COLOR_MAP[dominant]}33;"><div class="stat-value" style="color:{COLOR_MAP[dominant]};">{dominant}</div><div class="stat-label">Basa Dominan</div></div>', unsafe_allow_html=True)

        # ── Matriks Probabilitas ──
        if show_matrix:
            st.markdown("<br>", unsafe_allow_html=True)
            
            n_positions = [i for i in range(SEQ_LENGTH) if clean_used[i] == 'N']
            st.markdown(
                f'<div class="section-label">▷ MATRIKS PROBABILITAS — {len(n_positions)} POSISI DIIMPUTASI</div>',
                unsafe_allow_html=True
            )
        
        rows = ""
        for i in range(SEQ_LENGTH):
            orig = clean_used[i]
            if orig != 'N':
                continue
            
            probs = pred[i]
            best  = int(best_idxs[i])
            cells = ""
            for j, b in enumerate("ATCG"):
                pct = probs[j] * 100
                if j == best:
                    cells += f'<td><span class="prob-best prob-{b}-best">{pct:.1f}%</span></td>'
                else:
                    cells += f'<td style="color:#475569">{pct:.1f}%</td>'
            rows += f"<tr><td style='color:#64748b;font-size:10px'>POS {i+1:02d} ★</td>{cells}</tr>"

        if not rows:
            st.info("Tidak ada posisi N — semua basa sudah diketahui.")
        else:
            table_html = f"""
            <div style="background:#0a0f1a;border:1px solid #1f2937;border-radius:10px;padding:14px;overflow-x:auto;">
                <table class="prob-table">
                    <thead>
                        <tr>
                            <th>POSISI</th>
                            <th style="color:#f43f5e">A</th>
                            <th style="color:#38bdf8">T</th>
                            <th style="color:#34d399">C</th>
                            <th style="color:#fbbf24">G</th>
                        </tr>
                    </thead>
                    <tbody>{rows}</tbody>
                </table>
            </div>
            """
            st.markdown(table_html, unsafe_allow_html=True)
            st.caption("★ = posisi yang diimputasi dari N")
            st.success(f"✅ Imputasi selesai untuk sekuens: {clean_used}")