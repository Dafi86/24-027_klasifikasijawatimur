
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================================================
# KONFIGURASI APLIKASI
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="GeoLahan Jatim",
    page_icon="🌏",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAMA_APLIKASI = "GeoLahan Jatim"
SUBJUDUL = "Dashboard Klasifikasi Tutupan Lahan Jawa Timur"

KELAS = {
    1: "Sawah",
    2: "Bangunan",
    3: "Mangrove",
    4: "Lahan Hijau",
    5: "Laut",
    6: "Danau/Ranu",
}

FITUR = [
    "B02", "B03", "B04", "B08", "B11",
    "NDVI", "NDWI", "MNDWI", "NDBI",
]

WARNA_KELAS = {
    "Sawah": "#84CC16",
    "Bangunan": "#F97316",
    "Mangrove": "#047857",
    "Lahan Hijau": "#22C55E",
    "Laut": "#0EA5E9",
    "Danau/Ranu": "#6366F1",
}


# =========================================================
# STYLE
# =========================================================

st.markdown("""
<style>
    .stApp {
        background-color: #F5F7FB;
    }

    [data-testid="stSidebar"] {
        background-color: #102A43;
    }

    [data-testid="stSidebar"] * {
        color: #F0F7FF;
    }

    .hero {
        padding: 28px;
        border-radius: 20px;
        color: white;
        background: linear-gradient(120deg, #102A43, #087E8B);
        margin-bottom: 24px;
    }

    .hero h1 {
        font-size: 34px;
        margin: 0 0 8px 0;
    }

    .hero p {
        font-size: 15px;
        margin-bottom: 0;
        color: #E0F2FE;
    }

    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 16px;
        border: 1px solid #E2E8F0;
        min-height: 118px;
    }

    .metric-label {
        color: #64748B;
        font-size: 13px;
    }

    .metric-value {
        color: #102A43;
        font-size: 27px;
        font-weight: 750;
        margin-top: 8px;
    }

    .section-title {
        color: #102A43;
        font-size: 22px;
        font-weight: 700;
        margin: 12px 0;
    }

    .info-card {
        background: white;
        border: 1px solid #E2E8F0;
        padding: 18px;
        border-radius: 14px;
        margin-bottom: 12px;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #E2E8F0;
        padding: 16px;
        border-radius: 14px;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# FUNGSI PEMBANTU
# =========================================================

def baca_csv(nama_file):
    """Membaca CSV dari folder yang sama dengan app.py."""
    path = BASE_DIR / nama_file

    if not path.exists():
        return None

    try:
        return pd.read_csv(path)
    except Exception as e:
        st.warning(f"File {nama_file} gagal dibaca: {e}")
        return None


def tampilkan_csv(nama_file, judul):
    """Menampilkan tabel jika CSV tersedia."""
    st.markdown(f"### {judul}")
    df = baca_csv(nama_file)

    if df is None:
        st.info(
            f"File `{nama_file}` belum tersedia di folder utama. "
            "Unggah file tersebut ke GitHub jika ingin menampilkan tabel ini."
        )
        return

    st.dataframe(df, use_container_width=True, hide_index=True)


def kartu_metrik(label, nilai, catatan=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{nilai}</div>
            <div class="metric-label">{catatan}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def muat_model():
    """
    Mengunduh model privat dari Hugging Face.
    HF_TOKEN harus disimpan di Streamlit Cloud Secrets.
    """
    from huggingface_hub import hf_hub_download

    token = st.secrets.get("HF_TOKEN")

    if not token:
        raise RuntimeError(
            "HF_TOKEN belum diatur di Streamlit Cloud → Settings → Secrets."
        )

    lokasi_model = hf_hub_download(
        repo_id="Dafi86/model-klasifikasi-jatim",
        filename="model_klasifikasi.joblib",
        token=token,
    )

    return joblib.load(lokasi_model)


def hitung_indeks(b02, b03, b04, b08, b11):
    """Menghitung indeks spektral dari nilai band masukan."""

    def aman(pembilang, penyebut):
        if penyebut == 0:
            return 0.0
        return float(pembilang / penyebut)

    return {
        "NDVI": aman(b08 - b04, b08 + b04),
        "NDWI": aman(b03 - b08, b03 + b08),
        "MNDWI": aman(b03 - b11, b03 + b11),
        "NDBI": aman(b11 - b08, b11 + b08),
    }


def format_persen(nilai):
    if nilai is None:
        return "—"
    return f"{nilai:.2f}%"


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown("# 🌏 GeoLahan")
    st.caption("JAWA TIMUR • REMOTE SENSING")
    st.divider()

    halaman = st.radio(
        "NAVIGASI",
        [
            "Dashboard",
            "Peta Klasifikasi",
            "Evaluasi Model",
            "Prediksi Tutupan Lahan",
            "Informasi",
        ],
    )

    st.divider()
    st.caption("KLASIFIKASI TUTUPAN LAHAN")
    st.caption("Analisis berbasis fitur spektral")
    st.caption("Model Machine Learning")


# =========================================================
# DASHBOARD
# =========================================================

if halaman == "Dashboard":

    st.markdown(
        f"""
        <div class="hero">
            <h1>🌏 {NAMA_APLIKASI}</h1>
            <p>{SUBJUDUL}</p>
            <p>Visualisasi, evaluasi, dan prediksi kelas tutupan lahan
            berbasis fitur spektral citra.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    evaluasi = baca_csv("evaluasi_model.csv")
    laporan = baca_csv("classification_report.csv")

    # Nilai default adalah hasil evaluasi yang pernah dilaporkan.
    # Angka tersebut ditampilkan sebagai referensi, bukan hasil
    # perhitungan ulang dari CSV yang belum tersedia.
    akurasi = 82.34
    f1_macro = 78.82
    jumlah_data = 6404

    if evaluasi is not None:
        st.caption(
            "File evaluasi_model.csv ditemukan. "
            "Periksa formatnya sebelum menggunakan nilainya sebagai metrik."
        )

    st.markdown("### 📊 Ringkasan Model")

    c1, c2, c3 = st.columns(3)

    with c1:
        kartu_metrik("Akurasi referensi", format_persen(akurasi),
                     "Hasil evaluasi sebelumnya")

    with c2:
        kartu_metrik("Macro F1 referensi", format_persen(f1_macro),
                     "Hasil evaluasi sebelumnya")

    with c3:
        kartu_metrik("Jumlah sampel evaluasi", f"{jumlah_data:,}",
                     "Jumlah yang dilaporkan sebelumnya")

    st.caption(
        "Metrik di atas merupakan angka referensi dari evaluasi sebelumnya, "
        "bukan hasil pembacaan otomatis dari CSV. Perbarui angka di kode "
        "jika hasil evaluasi terbaru berbeda."
    )

    st.markdown("### 🗺️ Kelas Tutupan Lahan")

    kolom = st.columns(3)

    for i, (nomor, nama) in enumerate(KELAS.items()):
        with kolom[i % 3]:
            st.markdown(
                f"""
                <div class="info-card">
                    <div style="font-size:25px">{["🌾","🏙️","🌿","🌳","🌊","💧"][i]}</div>
                    <b>{nama}</b>
                    <div style="color:#64748B;font-size:13px">
                        Kode kelas: {nomor}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("### 📈 Laporan Evaluasi")

    if laporan is not None:
        st.dataframe(laporan, use_container_width=True, hide_index=True)
    else:
        st.info(
            "Laporan klasifikasi belum tersedia. "
            "Letakkan `classification_report.csv` di folder utama."
        )

    st.markdown("### 🚀 Mulai Analisis")

    a, b, c = st.columns(3)

    with a:
        st.markdown("**🗺️ Peta Klasifikasi**")
        st.write("Lihat peta HTML jika sudah tersedia.")

    with b:
        st.markdown("**📊 Evaluasi Model**")
        st.write("Periksa laporan klasifikasi dan confusion matrix.")

    with c:
        st.markdown("**🧪 Prediksi**")
        st.write("Masukkan nilai band untuk mencoba model.")


# =========================================================
# PETA KLASIFIKASI
# =========================================================

elif halaman == "Peta Klasifikasi":

    st.title("🗺️ Peta Klasifikasi")
    st.write(
        "Halaman untuk melihat peta tutupan lahan yang telah dihasilkan "
        "dari proses klasifikasi."
    )

    # Mendukung beberapa nama file di folder utama.
    kandidat_peta = [
        "peta_klasifikasi.html",
        "map.html",
        "peta.html",
    ]

    path_peta = next(
        (BASE_DIR / nama for nama in kandidat_peta
         if (BASE_DIR / nama).exists()),
        None,
    )

    if path_peta:
        try:
            html_peta = path_peta.read_text(encoding="utf-8")
            st.components.v1.html(
                html_peta,
                height=650,
                scrolling=True,
            )
        except Exception as e:
            st.error(f"Peta gagal ditampilkan: {e}")
    else:
        st.info(
            "File peta belum tersedia. Unggah `peta_klasifikasi.html` "
            "ke folder yang sama dengan `app.py`."
        )

    st.caption(
        "Cakupan peta mengikuti raster atau data yang digunakan. "
        "Jangan menganggap seluruh wilayah Jawa Timur telah tercakup "
        "jika data sumber hanya mencakup sebagian wilayah."
    )


# =========================================================
# EVALUASI MODEL
# =========================================================

elif halaman == "Evaluasi Model":

    st.title("📊 Evaluasi Model")
    st.write(
        "Lihat tabel evaluasi yang dihasilkan dari proses pengujian model."
    )

    evaluasi = baca_csv("evaluasi_model.csv")
    laporan = baca_csv("classification_report.csv")
    confusion = baca_csv("confusion_matrix.csv")

    tab1, tab2, tab3 = st.tabs(
        ["Ringkasan", "Classification Report", "Confusion Matrix"]
    )

    with tab1:
        if evaluasi is not None:
            st.dataframe(
                evaluasi,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info(
                "`evaluasi_model.csv` belum tersedia di folder utama."
            )

    with tab2:
        if laporan is not None:
            st.dataframe(
                laporan,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info(
                "`classification_report.csv` belum tersedia. "
                "Unggah file ini ke folder utama repository."
            )

    with tab3:
        if confusion is not None:
            st.dataframe(
                confusion,
                use_container_width=True,
                hide_index=True,
            )

            # Grafik heatmap sederhana jika CSV berisi matriks angka.
            try:
                angka = confusion.select_dtypes(include="number")

                if not angka.empty:
                    st.markdown("#### Visualisasi Confusion Matrix")
                    st.dataframe(
                        angka.style.background_gradient(cmap="Blues"),
                        use_container_width=True,
                    )
            except Exception:
                pass
        else:
            st.info(
                "`confusion_matrix.csv` belum tersedia di folder utama."
            )


# =========================================================
# PREDIKSI TUTUPAN LAHAN
# =========================================================

elif halaman == "Prediksi Tutupan Lahan":

    st.title("🧪 Prediksi Tutupan Lahan")
    st.write(
        "Masukkan nilai band spektral untuk memperoleh prediksi dari model."
    )

    st.warning(
        "Gunakan nilai band dengan skala dan preprocessing yang sama "
        "seperti saat model dilatih. Nilai contoh bukan data pengamatan "
        "nyata. Form ini memprediksi satu sampel, bukan membuat peta raster."
    )

    with st.form("form_prediksi"):
        st.markdown("#### Nilai band spektral")

        col1, col2, col3 = st.columns(3)

        with col1:
            b02 = st.number_input("B02 (Blue)", value=0.10, format="%.6f")
            b03 = st.number_input("B03 (Green)", value=0.12, format="%.6f")

        with col2:
            b04 = st.number_input("B04 (Red)", value=0.10, format="%.6f")
            b08 = st.number_input("B08 (NIR)", value=0.30, format="%.6f")

        with col3:
            b11 = st.number_input("B11 (SWIR)", value=0.15, format="%.6f")

        tombol_prediksi = st.form_submit_button(
            "🔍 Prediksi Sekarang",
            use_container_width=True,
        )

    if tombol_prediksi:
        indeks = hitung_indeks(b02, b03, b04, b08, b11)

        data_prediksi = {
            "B02": b02,
            "B03": b03,
            "B04": b04,
            "B08": b08,
            "B11": b11,
            **indeks,
        }

        input_model = pd.DataFrame([data_prediksi], columns=FITUR)

        with st.expander("Lihat fitur yang dihitung"):
            st.dataframe(input_model, use_container_width=True)

        try:
            with st.spinner("Memuat model dan melakukan prediksi..."):
                model = muat_model()
                hasil = model.predict(input_model)[0]

            nama_kelas = KELAS.get(
                int(hasil),
                f"Kelas {hasil}",
            )

            st.success("Prediksi selesai")

            st.markdown(
                f"""
                <div class="hero">
                    <p>HASIL KLASIFIKASI</p>
                    <h1>{nama_kelas}</h1>
                    <p>Kode kelas hasil model: {hasil}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if hasattr(model, "predict_proba"):
                try:
                    probabilitas = model.predict_proba(input_model)[0]
                    nama_kelas_model = getattr(
                        model, "classes_", range(len(probabilitas))
                    )

                    df_prob = pd.DataFrame({
                        "Kelas": [
                            KELAS.get(int(k), f"Kelas {k}")
                            for k in nama_kelas_model
                        ],
                        "Probabilitas": probabilitas,
                    }).sort_values(
                        "Probabilitas",
                        ascending=False,
                    )

                    st.markdown("#### Probabilitas prediksi")
                    st.bar_chart(
                        df_prob.set_index("Kelas")["Probabilitas"]
                    )
                    st.dataframe(
                        df_prob,
                        use_container_width=True,
                        hide_index=True,
                    )
                except Exception as e:
                    st.caption(
                        f"Probabilitas tidak dapat ditampilkan: {e}"
                    )

        except Exception as e:
            st.error(
                "Model gagal dimuat atau prediksi gagal dilakukan."
            )
            st.code(str(e))
            st.info(
                "Periksa HF_TOKEN pada Secrets, nama file model di "
                "Hugging Face, dependensi, serta kompatibilitas model "
                "dengan versi scikit-learn/joblib."
            )


# =========================================================
# INFORMASI
# =========================================================

elif halaman == "Informasi":

    st.title("ℹ️ Informasi Aplikasi")

    st.markdown("""
    ### Tentang GeoLahan Jatim

    Aplikasi ini merupakan antarmuka untuk menampilkan informasi
    klasifikasi tutupan lahan Jawa Timur menggunakan fitur spektral
    dan model Machine Learning.

    ### Fitur aplikasi

    - Dashboard ringkasan.
    - Tampilan peta HTML yang sudah tersedia.
    - Tabel evaluasi model.
    - Prediksi satu sampel berdasarkan input band.
    - Informasi kelas dan indeks spektral.

    ### Kelas tutupan lahan

    """)

    st.table(pd.DataFrame({
        "Kode": list(KELAS.keys()),
        "Kelas": list(KELAS.values()),
    }))

    st.markdown("""
    ### Fitur input model

    - B02: Blue
    - B03: Green
    - B04: Red
    - B08: Near Infrared (NIR)
    - B11: Shortwave Infrared (SWIR)
    - NDVI, NDWI, MNDWI, dan NDBI: indeks spektral turunan

    ### Rumus indeks

    - NDVI = (B08 − B04) / (B08 + B04)
    - NDWI = (B03 − B08) / (B03 + B08)
    - MNDWI = (B03 − B11) / (B03 + B11)
    - NDBI = (B11 − B08) / (B11 + B08)

    ### Catatan penting

    Hasil prediksi bergantung pada kualitas input, skala band,
    preprocessing, urutan fitur, dan model yang digunakan ketika
    pelatihan. Hasil satu sampel tidak sama dengan peta klasifikasi
    seluruh wilayah.
    """)

    st.caption(
        "GeoLahan Jatim • Dashboard Klasifikasi Tutupan Lahan"
    )
