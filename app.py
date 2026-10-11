
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from huggingface_hub import hf_hub_download

# =====================================================
# KONFIGURASI
# =====================================================
st.set_page_config(
    page_title="Klasifikasi Tutupan Lahan Jawa Timur",
    page_icon="🌏",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
STATIC_DIR = BASE_DIR / "static"

HF_REPO_ID = "Dafi86/model-klasifikasi-jatim"
HF_MODEL_FILENAME = "model_klasifikasi.joblib"

FEATURES = [
    "B02", "B03", "B04", "B08", "B11",
    "NDVI", "NDWI", "MNDWI", "NDBI"
]

CLASS_NAMES = {
    1: "Sawah",
    2: "Bangunan",
    3: "Mangrove",
    4: "Lahan Hijau",
    5: "Laut",
    6: "Danau/Ranu",
}

# =====================================================
# MEMUAT MODEL DARI HUGGING FACE
# =====================================================
@st.cache_resource
def load_model():
    token = st.secrets.get("HF_TOKEN")

    if not token:
        raise RuntimeError(
            "HF_TOKEN belum diatur. Buka Settings > Secrets "
            "di Streamlit Cloud."
        )

    model_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_MODEL_FILENAME,
        token=token,
    )

    return joblib.load(model_path)


def calculate_indices(b02, b03, b04, b08, b11):
    def safe_ratio(a, b):
        return (a / b) if b != 0 else 0.0

    ndvi = safe_ratio(b08 - b04, b08 + b04)
    ndwi = safe_ratio(b03 - b08, b03 + b08)
    mndwi = safe_ratio(b03 - b11, b03 + b11)
    ndbi = safe_ratio(b11 - b08, b11 + b08)

    return ndvi, ndwi, mndwi, ndbi


def read_csv_file(filename):
    path = DATA_DIR / filename
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception as error:
            st.error(f"Gagal membaca {filename}: {error}")
    return None


# =====================================================
# HEADER DAN NAVIGASI
# =====================================================
st.title("🌏 Dashboard Klasifikasi Tutupan Lahan Jawa Timur")
st.caption(
    "Klasifikasi tutupan lahan menggunakan fitur Sentinel-2 "
    "dan model machine learning."
)

with st.sidebar:
    st.header("Navigasi")
    page = st.radio(
        "Pilih halaman",
        [
            "Dashboard",
            "Peta Klasifikasi",
            "Evaluasi Model",
            "Prediksi Tutupan Lahan",
            "Informasi",
        ],
    )

    st.divider()
    st.caption("Model: Hugging Face")
    st.caption(HF_REPO_ID)


# =====================================================
# HALAMAN DASHBOARD
# =====================================================
if page == "Dashboard":
    st.subheader("Ringkasan Proyek")

    col1, col2, col3 = st.columns(3)
    col1.metric("Jumlah kelas", len(CLASS_NAMES))
    col2.metric("Jumlah fitur", len(FEATURES))
    col3.metric("Sumber data", "Sentinel-2")

    st.markdown("### Kelas Tutupan Lahan")

    cols = st.columns(3)
    for i, (class_id, class_name) in enumerate(CLASS_NAMES.items()):
        cols[i % 3].info(f"{class_id}. {class_name}")

    st.markdown("### Tentang Dashboard")
    st.write(
        "Dashboard ini digunakan untuk melihat peta klasifikasi, "
        "hasil evaluasi model, serta mencoba prediksi tutupan lahan "
        "berdasarkan lima band Sentinel-2 dan indeks spektral."
    )

    st.warning(
        "Cakupan peta bergantung pada data raster yang tersedia. "
        "Jangan menganggap seluruh Jawa Timur tercakup jika data "
        "hanya meliputi sebagian wilayah."
    )


# =====================================================
# HALAMAN PETA KLASIFIKASI
# =====================================================
elif page == "Peta Klasifikasi":
    st.subheader("Peta Klasifikasi Tutupan Lahan")

    map_path = STATIC_DIR / "peta_klasifikasi.html"

    if map_path.exists():
        try:
            html_content = map_path.read_text(encoding="utf-8")
            st.components.v1.html(
                html_content,
                height=650,
                scrolling=True,
            )
        except Exception as error:
            st.error(f"Peta gagal ditampilkan: {error}")
    else:
        st.warning(
            "File peta tidak ditemukan. Pastikan file berikut "
            "sudah diunggah ke GitHub:"
        )
        st.code("static/peta_klasifikasi.html")


# =====================================================
# HALAMAN EVALUASI MODEL
# =====================================================
elif page == "Evaluasi Model":
    st.subheader("Evaluasi Model")

    evaluation = read_csv_file("evaluasi_model.csv")
    report = read_csv_file("classification_report.csv")
    matrix = read_csv_file("confusion_matrix.csv")

    st.markdown("### Ringkasan Evaluasi")
    if evaluation is not None:
        st.dataframe(evaluation, use_container_width=True)
    else:
        st.info("File data/evaluasi_model.csv belum tersedia.")

    st.markdown("### Classification Report")
    if report is not None:
        st.dataframe(report, use_container_width=True)
    else:
        st.info("File data/classification_report.csv belum tersedia.")

    st.markdown("### Confusion Matrix")
    if matrix is not None:
        st.dataframe(matrix, use_container_width=True)
    else:
        st.info("File data/confusion_matrix.csv belum tersedia.")


# =====================================================
# HALAMAN PREDIKSI
# =====================================================
elif page == "Prediksi Tutupan Lahan":
    st.subheader("Prediksi Tutupan Lahan")

    st.write(
        "Masukkan nilai band Sentinel-2. Pastikan skala band dan "
        "preprocessing sesuai dengan proses pelatihan model."
    )

    with st.form("form_prediksi"):
        st.markdown("#### Nilai Band Sentinel-2")

        col1, col2, col3 = st.columns(3)

        b02 = col1.number_input(
            "B02 (Blue)", min_value=0.0, value=0.1, format="%.6f"
        )
        b03 = col2.number_input(
            "B03 (Green)", min_value=0.0, value=0.1, format="%.6f"
        )
        b04 = col3.number_input(
            "B04 (Red)", min_value=0.0, value=0.1, format="%.6f"
        )
        b08 = col1.number_input(
            "B08 (NIR)", min_value=0.0, value=0.2, format="%.6f"
        )
        b11 = col2.number_input(
            "B11 (SWIR)", min_value=0.0, value=0.1, format="%.6f"
        )

        ndvi, ndwi, mndwi, ndbi = calculate_indices(
            b02, b03, b04, b08, b11
        )

        st.markdown("#### Indeks Spektral")
        i1, i2, i3, i4 = st.columns(4)
        i1.metric("NDVI", f"{ndvi:.4f}")
        i2.metric("NDWI", f"{ndwi:.4f}")
        i3.metric("MNDWI", f"{mndwi:.4f}")
        i4.metric("NDBI", f"{ndbi:.4f}")

        submit_prediction = st.form_submit_button(
            "Prediksi",
            type="primary",
        )

    if submit_prediction:
        input_data = pd.DataFrame(
            [[b02, b03, b04, b08, b11,
              ndvi, ndwi, mndwi, ndbi]],
            columns=FEATURES,
        )

        try:
            with st.spinner("Memuat model dan menjalankan prediksi..."):
                model = load_model()
                prediction = model.predict(input_data)

            predicted_class = prediction[0]

            try:
                class_id = int(predicted_class)
            except (TypeError, ValueError):
                class_id = None

            if class_id in CLASS_NAMES:
                st.success(
                    f"Hasil prediksi: **{CLASS_NAMES[class_id]}**"
                )
                st.write(f"ID kelas: {class_id}")
            else:
                st.success(f"Hasil prediksi: {predicted_class}")

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_data)[0]
                classes = getattr(
                    model,
                    "classes_",
                    range(len(probabilities)),
                )

                probability_rows = []
                for class_value, probability in zip(
                    classes, probabilities
                ):
                    try:
                        class_key = int(class_value)
                        class_label = CLASS_NAMES.get(
                            class_key, str(class_value)
                        )
                    except (TypeError, ValueError):
                        class_label = str(class_value)

                    probability_rows.append({
                        "Kelas": class_label,
                        "Probabilitas": float(probability),
                    })

                probability_df = pd.DataFrame(probability_rows)
                probability_df = probability_df.sort_values(
                    "Probabilitas",
                    ascending=False,
                )

                st.markdown("### Probabilitas Kelas")
                st.dataframe(
                    probability_df,
                    use_container_width=True,
                    hide_index=True,
                )

        except Exception as error:
            st.error(f"Prediksi gagal: {error}")
            st.info(
                "Periksa HF_TOKEN di Streamlit Secrets, nama file model "
                "di Hugging Face, serta kesesuaian urutan fitur dan "
                "preprocessing dengan proses pelatihan."
            )


# =====================================================
# HALAMAN INFORMASI
# =====================================================
elif page == "Informasi":
    st.subheader("Informasi Proyek")

    st.markdown("### Fitur Model")
    st.write(", ".join(FEATURES))

    st.markdown("### Kode Kelas")
    class_df = pd.DataFrame([
        {"ID Kelas": class_id, "Nama Kelas": class_name}
        for class_id, class_name in CLASS_NAMES.items()
    ])
    st.dataframe(
        class_df,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Rumus Indeks Spektral")
    st.latex(r"NDVI = \frac{B08-B04}{B08+B04}")
    st.latex(r"NDWI = \frac{B03-B08}{B03+B08}")
    st.latex(r"MNDWI = \frac{B03-B11}{B03+B11}")
    st.latex(r"NDBI = \frac{B11-B08}{B11+B08}")

    st.warning(
        "Rumus, skala band, urutan fitur, dan preprocessing harus sama "
        "dengan yang digunakan saat model dilatih."
    )

st.divider()
st.caption("Dashboard Klasifikasi Tutupan Lahan Jawa Timur")
