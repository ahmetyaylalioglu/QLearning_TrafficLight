"""
Dinamik Trafik Isigi Kontrolu - Q-Learning
Acil arac (ambulans/itfaiye) destegi ile.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import io, os, random

random.seed(42)
np.random.seed(42)

# =========================================================
#  KAVSAK ORTAMI
# =========================================================
class TrafikKavsagi:
    """
    Durum: (K-G yogunluk, D-B yogunluk, acil_durum) -> 12 durum
      yogunluk: Az(0)/Cok(1), acil: Yok(0)/KG(1)/DB(2)
    Aksiyon: 0=K-G yesil, 1=D-B yesil
    Odul: -(bekleyen arac) + acil arac cezasi/odul
    """
    DURUM_SAYISI = 12      # 2x2x3
    AKSIYON_SAYISI = 2
    YOGUNLUK_ESIGI = 4
    MAX_ARAC = 12
    ACIL_OLASILIK = 0.08   # her adimda acil arac gelme olasiligi

    def __init__(self):
        self.kg_araclar = 0
        self.db_araclar = 0
        self.yesil_yon = 0
        self.acil_durum = 0    # 0=yok, 1=K-G'de acil, 2=D-B'de acil
        self.acil_tip = None   # "ambulans" veya "itfaiye"

    def _durum(self):
        kg = 1 if self.kg_araclar >= self.YOGUNLUK_ESIGI else 0
        db = 1 if self.db_araclar >= self.YOGUNLUK_ESIGI else 0
        return self.acil_durum * 4 + kg * 2 + db

    def reset(self):
        self.kg_araclar = random.randint(1, 8)
        self.db_araclar = random.randint(1, 8)
        self.yesil_yon = random.choice([0, 1])
        self.acil_durum = 0
        self.acil_tip = None
        return self._durum()

    def step(self, aksiyon):
        self.yesil_yon = aksiyon

        # Acil arac varsa ve dogru yon secildiyse → acil arac gecer
        acil_odul = 0
        if self.acil_durum == 1 and aksiyon == 0:
            acil_odul = 10     # acil araca yesil verdik, bonus
            self.acil_durum = 0
            self.acil_tip = None
        elif self.acil_durum == 2 and aksiyon == 1:
            acil_odul = 10
            self.acil_durum = 0
            self.acil_tip = None
        elif self.acil_durum > 0:
            acil_odul = -20    # acil araca kirmizi verdik, buyuk ceza

        # Yesil yonden 2-4 arac gecir
        if aksiyon == 0:
            self.kg_araclar -= min(self.kg_araclar, random.randint(2, 4))
        else:
            self.db_araclar -= min(self.db_araclar, random.randint(2, 4))

        # Yeni araclar (0-3 arasi, daha uzun kuyruklar icin)
        self.kg_araclar = min(self.kg_araclar + random.randint(0, 3), self.MAX_ARAC)
        self.db_araclar = min(self.db_araclar + random.randint(0, 3), self.MAX_ARAC)

        # Rastgele acil arac gelebilir
        if self.acil_durum == 0 and random.random() < self.ACIL_OLASILIK:
            self.acil_durum = random.choice([1, 2])
            self.acil_tip = random.choice(["ambulans", "itfaiye"])

        odul = -(self.kg_araclar + self.db_araclar) + acil_odul
        return self._durum(), odul

# =========================================================
#  Q-LEARNING AJANI
# =========================================================
class QLearningAjan:
    def __init__(self):
        self.q_tablo = np.zeros((12, 2))   # 12 durum x 2 aksiyon
        self.alfa = 0.2
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_azalma = 0.997
        self.epsilon_min = 0.01

    def aksiyon_sec(self, durum):
        if random.random() < self.epsilon:
            return random.randint(0, 1)
        return int(np.argmax(self.q_tablo[durum]))

    def ogren(self, durum, aksiyon, odul, yeni_durum):
        eski = self.q_tablo[durum, aksiyon]
        hedef = odul + self.gamma * np.max(self.q_tablo[yeni_durum])
        self.q_tablo[durum, aksiyon] = eski + self.alfa * (hedef - eski)

    def epsilon_azalt(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_azalma)

# =========================================================
#  GORSELLESITIRME
# =========================================================
DURUM_ADLARI = {0:"Az/Az", 1:"Az/Cok", 2:"Cok/Az", 3:"Cok/Cok"}
ACIL_ADLARI = {0:"Yok", 1:"K-G", 2:"D-B"}

def kare_ciz(ortam, episode, adim, odul, ajan=None, mod="EGITIM"):
    fig, ax = plt.subplots(figsize=(9, 10), facecolor="#0d1117")
    ax.set_xlim(-14, 14)
    ax.set_ylim(-15, 15)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("#0d1117")

    # Zemin
    ax.add_patch(patches.Rectangle((-14,-15), 28, 30, fc="#1b4332"))
    W = 5
    ax.add_patch(patches.Rectangle((-W/2,-15), W, 30, fc="#3a3a4a"))
    ax.add_patch(patches.Rectangle((-14,-W/2), 28, W, fc="#3a3a4a"))
    ax.add_patch(patches.Rectangle((-W/2,-W/2), W, W, fc="#4a4a5a"))

    # Kesikli cizgiler
    for i in range(-14, 15):
        if abs(i) > W/2+0.3:
            ax.plot([0,0],[i,i+0.5], color="#ffd60a", lw=1.2, alpha=0.5)
            ax.plot([i,i+0.5],[0,0], color="#ffd60a", lw=1.2, alpha=0.5)

    # Trafik isiklari (dur cizgileri)
    kg_renk = "#00e676" if ortam.yesil_yon == 0 else "#ff1744"
    db_renk = "#00e676" if ortam.yesil_yon == 1 else "#ff1744"
    bar = 0.5
    ax.add_patch(patches.Rectangle((-W/2, W/2), W, bar, fc=kg_renk, alpha=0.9))
    ax.add_patch(patches.Rectangle((-W/2, -W/2-bar), W, bar, fc=kg_renk, alpha=0.9))
    ax.add_patch(patches.Rectangle((W/2, -W/2), bar, W, fc=db_renk, alpha=0.9))
    ax.add_patch(patches.Rectangle((-W/2-bar, -W/2), bar, W, fc=db_renk, alpha=0.9))
    # Parlama
    for args in [
        ((-W/2-0.2, W/2-0.2), W+0.4, bar+0.4, kg_renk),
        ((-W/2-0.2, -W/2-bar-0.2), W+0.4, bar+0.4, kg_renk),
        ((W/2-0.2, -W/2-0.2), bar+0.4, W+0.4, db_renk),
        ((-W/2-bar-0.2, -W/2-0.2), bar+0.4, W+0.4, db_renk)]:
        ax.add_patch(patches.Rectangle(args[0], args[1], args[2], fc=args[3], alpha=0.15))

    # Isik aciklamasi
    bx, by = 6.5, 11
    ax.add_patch(patches.FancyBboxPatch((bx-0.5, by-2), 7, 3.8,
                 boxstyle="round,pad=0.3", fc="#111", ec="#555", lw=1.5, alpha=0.9))
    ax.add_patch(patches.Circle((bx+0.3, by+1.1), 0.45, fc=kg_renk))
    ax.text(bx+1.1, by+1.1, f"K-G: {'YESIL' if ortam.yesil_yon==0 else 'KIRMIZI'}",
            va="center", fontsize=10, color=kg_renk, fontweight="bold")
    ax.add_patch(patches.Circle((bx+0.3, by-0.2), 0.45, fc=db_renk))
    ax.text(bx+1.1, by-0.2, f"D-B: {'YESIL' if ortam.yesil_yon==1 else 'KIRMIZI'}",
            va="center", fontsize=10, color=db_renk, fontweight="bold")

    # Arac cizim fonksiyonu
    def arac(x, y, w, h, renk, tip="normal"):
        ax.add_patch(patches.FancyBboxPatch((x,y), w, h, boxstyle="round,pad=0.08",
                     fc=renk, ec="white", lw=0.8))
        if tip == "ambulans":
            # Beyaz arac, kirmizi haç
            cx, cy = x+w/2, y+h/2
            s = min(w,h)*0.25
            ax.plot([cx-s,cx+s],[cy,cy], color="#ff1744", lw=2.5)
            ax.plot([cx,cx],[cy-s,cy+s], color="#ff1744", lw=2.5)
        elif tip == "itfaiye":
            # Kirmizi arac, sari serit
            if w > h:
                ax.add_patch(patches.Rectangle((x+0.1,y+h*0.4), w-0.2, h*0.2,
                             fc="#ffd60a", alpha=0.8))
            else:
                ax.add_patch(patches.Rectangle((x+w*0.4,y+0.1), w*0.2, h-0.2,
                             fc="#ffd60a", alpha=0.8))
        else:
            # Normal arac cam efekti
            if w > h:
                ax.add_patch(patches.Rectangle((x+w*0.15,y+h*0.15), w*0.25, h*0.7,
                             fc="white", alpha=0.2))
            else:
                ax.add_patch(patches.Rectangle((x+w*0.15,y+h*0.6), w*0.7, h*0.25,
                             fc="white", alpha=0.2))

    # K-G araclari (mavi) - kuzeyden ve guneyden
    kg_k = (ortam.kg_araclar + 1) // 2
    kg_g = ortam.kg_araclar - kg_k
    for i in range(min(kg_k, 7)):
        arac(-W/2+0.4, W/2+bar+0.5+i*1.8, 1.6, 1.2, "#42a5f5")
    for i in range(min(kg_g, 7)):
        arac(W/2-2.0, -W/2-bar-1.7-i*1.8, 1.6, 1.2, "#1e88e5")

    # D-B araclari (kirmizi) - dogudan ve batidan
    db_d = (ortam.db_araclar + 1) // 2
    db_b = ortam.db_araclar - db_d
    for i in range(min(db_d, 7)):
        arac(W/2+bar+0.5+i*1.8, -W/2+0.4, 1.2, 1.6, "#ef5350")
    for i in range(min(db_b, 7)):
        arac(-W/2-bar-1.7-i*1.8, W/2-2.0, 1.2, 1.6, "#e53935")

    # Acil arac cizimi
    if ortam.acil_durum == 1:  # K-G'de acil arac
        tip = ortam.acil_tip or "ambulans"
        renk = "white" if tip == "ambulans" else "#ff6f00"
        # Kuzeyden gelen acil arac (en onde)
        arac(-W/2+0.4, W/2+bar+0.5+min(kg_k,7)*1.8, 1.6, 1.2, renk, tip)
        ax.text(0, W/2+bar+0.5+min(kg_k,7)*1.8+1.8, f"ACIL: {tip.upper()}!",
                ha="center", fontsize=11, color="#ff6f00", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="#111", ec="#ff6f00", lw=2))
    elif ortam.acil_durum == 2:  # D-B'de acil arac
        tip = ortam.acil_tip or "ambulans"
        renk = "white" if tip == "ambulans" else "#ff6f00"
        arac(W/2+bar+0.5+min(db_d,7)*1.8, -W/2+0.4, 1.2, 1.6, renk, tip)
        ax.text(W/2+bar+0.5+min(db_d,7)*1.8+1.5, 0, f"ACIL:\n{tip.upper()}!",
                ha="center", fontsize=10, color="#ff6f00", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="#111", ec="#ff6f00", lw=2))

    # Yon etiketleri
    ax.text(0,14,"KUZEY",ha="center",fontsize=9,color="white",alpha=0.5,fontweight="bold")
    ax.text(0,-14.5,"GUNEY",ha="center",fontsize=9,color="white",alpha=0.5,fontweight="bold")
    ax.text(13.5,0,"DOGU",ha="center",fontsize=9,color="white",alpha=0.5,rotation=90,fontweight="bold")
    ax.text(-13.5,0,"BATI",ha="center",fontsize=9,color="white",alpha=0.5,rotation=90,fontweight="bold")

    # Arac sayilari
    ax.text(-W/2-4, 8, f"K-G\n{ortam.kg_araclar} arac", ha="center", fontsize=13,
            color="#42a5f5", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="#111", alpha=0.85))
    ax.text(W/2+5, -8, f"D-B\n{ortam.db_araclar} arac", ha="center", fontsize=13,
            color="#ef5350", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="#111", alpha=0.85))

    # Baslik
    ax.text(0, 14.5, f"Dinamik Trafik Isigi - {mod}", ha="center",
            fontsize=14, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.4", fc="#1a1a2e", ec="#555", lw=1.5))

    # Alt bilgi
    base = ortam._durum() % 4
    acil_t = ACIL_ADLARI[ortam.acil_durum]
    ax.text(0, -14.3,
            f"Bolum:{episode} | Adim:{adim} | Odul:{odul:.0f} | Durum:[{DURUM_ADLARI[base]}] | Acil:{acil_t}",
            ha="center", fontsize=9, color="white",
            bbox=dict(boxstyle="round,pad=0.3", fc="#1a1a2e", ec="#555", lw=1))

    # Q-degerleri
    if ajan is not None:
        s = ortam._durum()
        q0, q1 = ajan.q_tablo[s,0], ajan.q_tablo[s,1]
        best = "K-G" if q0 >= q1 else "D-B"
        qt = f"Q(K-G)={q0:.1f}  Q(D-B)={q1:.1f} => {best}"
        eps = f"eps={ajan.epsilon:.3f}" if ajan.epsilon > 0.001 else ""
        ax.text(-13.5, -14.8, qt, ha="left", fontsize=8, color="#aaa",
                bbox=dict(boxstyle="round,pad=0.2", fc="#111", alpha=0.7))
        if eps:
            ax.text(13.5, -14.8, eps, ha="right", fontsize=9, color="#ffd60a",
                    bbox=dict(boxstyle="round,pad=0.2", fc="#111", alpha=0.7))

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=80, bbox_inches="tight",
                facecolor=fig.get_facecolor(), pad_inches=0.1)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()
    return img

# =========================================================
#  YARDIMCI
# =========================================================
def odul_grafigi_ciz(oduller, yol):
    fig, ax = plt.subplots(figsize=(10,4), facecolor="#0f0f1a")
    ax.set_facecolor("#0f0f1a")
    p = 50
    if len(oduller) >= p:
        ax.plot(np.convolve(oduller, np.ones(p)/p, "valid"),
                color="#00e676", lw=2, label=f"Ort({p})")
    ax.plot(oduller, color="#42a5f5", alpha=0.2, lw=0.5, label="Ham")
    ax.set_xlabel("Bolum", color="white"); ax.set_ylabel("Odul", color="white")
    ax.set_title("Egitim - Odul Degisimi", color="white", fontsize=13)
    ax.legend(facecolor="#222", labelcolor="white")
    ax.tick_params(colors="white")
    for s in ax.spines.values(): s.set_color("#555")
    fig.tight_layout()
    fig.savefig(yol, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Grafik: {yol}")

def q_tablo_yazdir(ajan):
    print("\n  Q-TABLOSU (12 durum x 2 aksiyon)")
    print("  +----------------------+-----------+-----------+")
    print("  | Durum                | K-G Yesil | D-B Yesil |")
    print("  +----------------------+-----------+-----------+")
    for acil in range(3):
        acil_ad = ["Acil:Yok","Acil:K-G","Acil:D-B"][acil]
        for base in range(4):
            s = acil * 4 + base
            v0, v1 = ajan.q_tablo[s,0], ajan.q_tablo[s,1]
            m0 = " *" if v0 >= v1 else "  "
            m1 = " *" if v1 > v0 else "  "
            ad = f"{DURUM_ADLARI[base]:5s} {acil_ad}"
            print(f"  | {ad:<20s} |{v0:>8.2f}{m0} |{v1:>8.2f}{m1} |")
        print("  +----------------------+-----------+-----------+")
    print("  (* = en iyi aksiyon)\n")

# =========================================================
#  EGITIM
# =========================================================
def egitim(episode_sayisi=3000, adim_sayisi=25, gif_kaydet=True):
    ortam = TrafikKavsagi()
    ajan = QLearningAjan()
    odul_gecmisi = []
    gif_kareler = []
    gif_bolumleri = [0, 100, 500, 1000, 1500, 2000, 2500, 2999]

    print("=" * 55)
    print("  EGITIM BASLIYOR (3000 bolum)")
    print("=" * 55)

    for ep in range(episode_sayisi):
        durum = ortam.reset()
        toplam_odul = 0
        kayit = gif_kaydet and (ep in gif_bolumleri)

        for adim in range(adim_sayisi):
            aksiyon = ajan.aksiyon_sec(durum)
            yeni_durum, odul = ortam.step(aksiyon)
            ajan.ogren(durum, aksiyon, odul, yeni_durum)
            toplam_odul += odul
            durum = yeni_durum
            if kayit:
                gif_kareler.append(
                    kare_ciz(ortam, ep+1, adim+1, toplam_odul, ajan=ajan))

        ajan.epsilon_azalt()
        odul_gecmisi.append(toplam_odul)
        if (ep+1) % 500 == 0 or ep == 0:
            ort = np.mean(odul_gecmisi[-100:])
            print(f"  Bolum {ep+1:>5d}/{episode_sayisi} | Ort:{ort:>7.1f} | eps:{ajan.epsilon:.4f}")

    q_tablo_yazdir(ajan)
    return ajan, odul_gecmisi, gif_kareler

# =========================================================
#  TEST
# =========================================================
def test(ajan, episode_sayisi=3, adim_sayisi=25):
    ortam = TrafikKavsagi()
    ajan.epsilon = 0.0
    gif_kareler = []
    oduller = []
    print("=" * 55)
    print("  TEST (epsilon=0)")
    print("=" * 55)
    for ep in range(episode_sayisi):
        durum = ortam.reset()
        toplam_odul = 0
        for adim in range(adim_sayisi):
            aksiyon = ajan.aksiyon_sec(durum)
            yeni_durum, odul = ortam.step(aksiyon)
            toplam_odul += odul
            durum = yeni_durum
            gif_kareler.append(
                kare_ciz(ortam, ep+1, adim+1, toplam_odul, ajan=ajan, mod="TEST"))
        oduller.append(toplam_odul)
        print(f"  Bolum {ep+1}: Odul = {toplam_odul:.0f}")
    print(f"\n  Ortalama: {np.mean(oduller):.1f}")
    return gif_kareler

# =========================================================
#  GIF & MAIN
# =========================================================
def gif_kaydet(kareler, yol, sure_ms=600):
    if not kareler: return
    os.makedirs(os.path.dirname(yol), exist_ok=True)
    kareler[0].save(yol, save_all=True, append_images=kareler[1:],
                    duration=sure_ms, loop=0)
    print(f"  GIF: {yol} ({len(kareler)} kare, {sure_ms}ms)")

if __name__ == "__main__":
    PROJE = os.path.dirname(os.path.abspath(__file__))
    GIF_DIR = os.path.join(PROJE, "gifs")
    os.makedirs(GIF_DIR, exist_ok=True)

    ajan, oduller, ek = egitim(3000, 25)
    gif_kaydet(ek, os.path.join(GIF_DIR, "egitim.gif"), 600)
    odul_grafigi_ciz(oduller, os.path.join(GIF_DIR, "odul_grafigi.png"))

    tk = test(ajan, 3, 25)
    gif_kaydet(tk, os.path.join(GIF_DIR, "test.gif"), 700)

    print(f"\n  TAMAMLANDI! -> {GIF_DIR}")
