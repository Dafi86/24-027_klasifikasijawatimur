
from pathlib import Path
import json
import pickle

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import streamlit.components.v1 as components


# =========================================================
# KONFIGURASI
# =========================================================

st.set_page_config(
    page_title="Dashboard Klasifikasi Tutupan Lahan Jawa Timur",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
STATIC_DIR = BASE_DIR / "static"

MODEL_JOBLIB = MODEL_DIR / "model_klasifikasi.joblib"
MODEL_PICKLE = MODEL_DIR / "model_xgboost.pkl"
METADATA_PATH = MODEL_DIR / "metadata_model.json"

EVALUATION_PATH = DATA_DIR / "evaluasi_model.csv"
REPORT_PATH = DATA_DIR / "classification_report.csv"
CONFUSION_PATH = DATA_DIR / "confusion_matrix.csv"
MAP_PATH = STATIC_DIR / "peta_klasifikasi.html"

CLASS_NAMES = {
    1: "Sawah",
    2: "Bangunan",
    3: "Mangrove",
    4: "Lahan Hijau",
    5: "Laut",
    6: "Danau/Ranu",
}

FEATURES = [
    "B02", "B03", "B04", "B08", "B11",
    "NDVI", "NDWI", "MNDWI", "NDBI",
]


# =========================================================
# TAMPILAN
# =========================================================

st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    .hero {
        padding: 1.5rem;
        border-radius: 14px;
        background: linear-gradient(120deg, #123b35, #26745c);
        color: white;
        margin-bottom: 1.2rem;
    }
    .hero h1 {
        color: white;
        margin-bottom: .4rem;
    }
    .hero p {
        color: #e6f5ee;
        margin-bottom: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>🌍 Dashboard Tutupan Lahan Jawa Timur</h1>
      <p>
        Visualisasi peta, evaluasi model machine learning,
        dan prediksi kelas tutupan lahan.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# FUNGSI PEMBACAAN DATA
# =========================================================

def read_csv_safe(path):
    """Membaca CSV jika tersedia tanpa menghentikan aplikasi."""
    if not path.exists():
        return None

    try:
        return pd.read_csv(path)
    except Exception as exc:
        st.warning(f"Gagal membaca {path.name}: {exc}")
        return None


def read_json_safe(path):
    """Membaca metadata JSON jika tersedia."""
    if not path.exists():
        return None

    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as exc:
        st.warning(f"Gagal membaca {path.name}: {exc}")
        return None


@st.cache_resource
def load_model(model_path, modified_time):
    """
    Memuat model hanya ketika fungsi dipanggil.
    modified_time membantu cache diperbarui jika file berubah.
    """
    path = Path(model_path)

    if not path.exists():
        raise FileNotFoundError(f"File model tidak ditemukan: {path}")

    suffix = path.suffix.lower()

    if suffix == ".joblib":
        return joblib.load(path)

    if suffix in (".pkl", ".pickle"):
        # Hanya untuk file pickle yang berasal dari sumber tepercaya.
        with open(path, "rb") as file:
            return pickle.load(file)

    raise ValueError(f"Format model tidak didukung: {suffix}")


def show_status(label, path):
    if path.exists():
        st.success(f"{label}: tersedia")
    else:
        st.warning(f"{label}: tidak ditemukan")


def normalize_class_report(df):
    """
    Mengubah laporan klasifikasi ke format tabel yang mudah dibaca
    jika CSV memiliki kolom nama kelas sebagai indeks.
    """
    if df is None:
        return None

    result = df.copy()

    if result.columns[0].lower().startswith("unnamed"):
        result = result.rename(columns={result.columns[0]: "Kelas"})

    return result


# =========================================================
# SIDEBAR / NAVIGASI
# =========================================================

st.sidebar.title("Navigasi")

page = st.sidebar.radio(
    "Pilih halaman",
    [
        "Dashboard",
        "Peta Klasifikasi",
        "Evaluasi Model",
        "Prediksi",
        "Informasi",
    ],
)

st.sidebar.divider()
st.sidebar.caption("Klasifikasi tutupan lahan berbasis citra Sentinel-2")


# =========================================================
# HALAMAN DASHBOARD
# =========================================================

if page == "Dashboard":
    st.subheader("Ringkasan Proyek")

    st.write(
        """
        Dashboard ini digunakan untuk melihat hasil klasifikasi
        tutupan lahan, mengevaluasi performa model, dan mencoba
        prediksi berdasarkan nilai band Sentinel-2 beserta indeks
        spektralnya.
        """
    )

    st.subheader("Status berkas proyek")

    col1, col2 = st.columns(2)

    with col1:
        show_status("Model Joblib", MODEL_JOBLIB)
        show_status("Model Pickle lama", MODEL_PICKLE)
        show_status("Metadata model", METADATA_PATH)

    with col2:
        show_status("Data evaluasi", EVALUATION_PATH)
        show_status("Laporan klasifikasi", REPORT_PATH)
        show_status("Confusion matrix", CONFUSION_PATH)
        show_status("Peta HTML", MAP_PATH)

    evaluation = read_csv_safe(EVALUATION_PATH)

    if evaluation is not None:
        st.subheader("Data evaluasi")

        st.metric("Jumlah baris evaluasi", f"{len(evaluation):,}")

        if {"y_true", "y_pred"}.issubset(evaluation.columns):
            accuracy = (
                evaluation["y_true"] == evaluation["y_pred"]
            ).mean()

            st.metric("Akurasi dari data evaluasi", f"{accuracy:.2%}")

    st.info(
        "Model tidak dimuat pada halaman Dashboard. "
        "Buka halaman Prediksi untuk mencoba memuat model."
    )


# =========================================================
# HALAMAN PETA
# =========================================================

elif page == "Peta Klasifikasi":
    st.subheader("Peta Klasifikasi Tutupan Lahan")

    if MAP_PATH.exists():
        try:
            html = MAP_PATH.read_text(encoding="utf-8")
            components.html(
                html,
                height=700,
                scrolling=True,
            )
        except Exception as exc:
            st.error(f"Peta gagal ditampilkan: {exc}")
    else:
        st.warning(
            "File peta tidak ditemukan di "
            f"`{MAP_PATH}`."
        )
        st.write(
            "Letakkan file peta HTML di folder `static` dengan nama "
            "`peta_klasifikasi.html`."
        )

    st.caption(
        "Pastikan peta dan raster yang digunakan memiliki cakupan "
        "wilayah yang sesuai. Area tanpa data tidak boleh dianggap "
        "sebagai hasil klasifikasi."
    )


# =========================================================
# HALAMAN EVALUASI
# =========================================================

elif page == "Evaluasi Model":
    st.subheader("Evaluasi Model")

    evaluation = read_csv_safe(EVALUATION_PATH)
    report = normalize_class_report(read_csv_safe(REPORT_PATH))
    confusion = read_csv_safe(CONFUSION_PATH)

    if evaluation is not None:
        st.markdown("### Ringkasan evaluasi")

        if {"y_true", "y_pred"}.issubset(evaluation.columns):
            y_true = evaluation["y_true"]
            y_pred = evaluation["y_pred"]
            accuracy = (y_true == y_pred).mean()

            c1, c2 = st.columns(2)
            c1.metric("Akurasi", f"{accuracy:.2%}")
            c2.metric("Jumlah sampel", f"{len(evaluation):,}")

            st.markdown("### Distribusi hasil prediksi")

            distribution = (
                evaluation["y_pred"]
                .map(CLASS_NAMES)
                .fillna(evaluation["y_pred"].astype(str))
                .value_counts()
            )

            st.bar_chart(distribution)

        else:
            st.warning(
                "CSV evaluasi harus memiliki kolom `y_true` dan `y_pred` "
                "untuk menghitung akurasi."
            )

        with st.expander("Lihat data evaluasi"):
            st.dataframe(evaluation, use_container_width=True)

    else:
        st.warning("Data evaluasi tidak ditemukan.")

    st.divider()
    st.markdown("### Laporan klasifikasi")

    if report is not None:
        st.dataframe(report, use_container_width=True)
    else:
        st.warning("Laporan klasifikasi tidak ditemukan.")

    st.divider()
    st.markdown("### Confusion matrix")

    if confusion is not None:
        st.dataframe(confusion, use_container_width=True)

        try:
            numeric = confusion.select_dtypes(include="number")

            if not numeric.empty:
                fig, ax = plt.subplots(figsize=(8, 6))
                image = ax.imshow(numeric.to_numpy(), cmap="Greens")

                ax.set_title("Confusion Matrix")
                ax.set_xlabel("Prediksi")
                ax.set_ylabel("Aktual")

                ax.set_xticks(range(numeric.shape[1]))
                ax.set_xticklabels(numeric.columns, rotation=45, ha="right")

                row_labels = (
                    confusion.iloc[:, 0].astype(str).tolist()
                    if confusion.shape[1] > numeric.shape[1]
                    else [str(i) for i in range(numeric.shape[0])]
                )

                ax.set_yticks(range(numeric.shape[0]))
                ax.set_yticklabels(row_labels)

                for i in range(numeric.shape[0]):
                    for j in range(numeric.shape[1]):
                        ax.text(
                            j, i, f"{numeric.iloc[i, j]:g}",
                            ha="center", va="center",
                        )

                fig.colorbar(image, ax=ax)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        except Exception as exc:
            st.warning(f"Grafik confusion matrix gagal dibuat: {exc}")

    else:
        st.warning("File confusion matrix tidak ditemukan.")


# =========================================================
# HALAMAN PREDIKSI
# =========================================================

elif page == "Prediksi":
    st.subheader("Prediksi Tutupan Lahan")

    st.warning(
        "Masukkan nilai band dan indeks dengan skala serta "
        "preprocessing yang sama seperti saat model dilatih. "
        "Nilai contoh di bawah bukan data Sentinel-2 aktual."
    )

    model_path = (
        MODEL_JOBLIB if MODEL_JOBLIB.exists()
        else MODEL_PICKLE
    )

    if not model_path.exists():
        st.error(
            "File model tidak ditemukan. Letakkan model yang sudah "
            "disimpan ulang di `models/model_klasifikasi.joblib`."
        )
        st.stop()

    metadata = read_json_safe(METADATA_PATH)

    if metadata is not None:
        with st.expander("Metadata model"):
            st.json(metadata)

    st.markdown("### Masukkan nilai fitur")

    defaults = {
        "B02": 1000.0,
        "B03": 1200.0,
        "B04": 1000.0,
        "B08": 2500.0,
        "B11": 1500.0,
    }

    with st.form("prediction_form"):
        band_values = {}

        col1, col2, col3 = st.columns(3)

        columns = [col1, col2, col3]

        for i, feature in enumerate(["B02", "B03", "B04", "B08", "B11"]):
            with columns[i % 3]:
                band_values[feature] = st.number_input(
                    feature,
                    value=defaults[feature],
                    step=100.0,
                    format="%.4f",
                    help="Masukkan nilai dengan skala yang sama seperti data training.",
                )

        # Indeks dihitung otomatis berdasarkan nilai band yang dimasukkan.
        # Pastikan rumus ini identik dengan preprocessing saat training.
        b03 = band_values["B03"]
        b04 = band_values["B04"]
        b08 = band_values["B08"]
        b11 = band_values["B11"]

        def safe_index(numerator, denominator):
            if abs(denominator) < 1e-12:
                return 0.0
            return numerator / denominator

        indices = {
            "NDVI": safe_index(b08 - b04, b08 + b04),
            "NDWI": safe_index(b03 - b08, b03 + b08),
            "MNDWI": safe_index(b03 - b11, b03 + b11),
            "NDBI": safe_index(b11 - b08, b11 + b08),
        }

        st.markdown("### Indeks spektral hasil perhitungan")

        index_cols = st.columns(4)

        for i, (name, value) in enumerate(indices.items()):
            index_cols[i].metric(name, f"{value:.4f}")

        submitted = st.form_submit_button(
            "Jalankan Prediksi",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        try:
            model = load_model(
                str(model_path),
                model_path.stat().st_mtime,
            )

            row = {
                **band_values,
                **indices,
            }

            input_df = pd.DataFrame(
                [[row[feature] for feature in FEATURES]],
                columns=FEATURES,
            )

            # Periksa jumlah fitur jika model menyediakan informasi ini.
            expected_features = getattr(model, "n_features_in_", None)

            if (
                expected_features is not None
                and expected_features != input_df.shape[1]
            ):
                raise ValueError(
                    f"Model mengharapkan {expected_features} fitur, "
                    f"tetapi aplikasi mengirim {input_df.shape[1]}. "
                    "Periksa urutan dan jumlah fitur saat training."
                )

            prediction = model.predict(input_df)[0]

            try:
                predicted_id = int(prediction)
            except (TypeError, ValueError):
                predicted_id = prediction

            predicted_name = CLASS_NAMES.get(
                predicted_id,
                str(predicted_id),
            )

            st.success("Prediksi berhasil dilakukan.")
            st.markdown("### Hasil prediksi")

            st.metric("Kelas prediksi", predicted_name)
            st.write(f"ID kelas: `{predicted_id}`")

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_df)[0]
                classes = getattr(model, "classes_", range(len(probabilities)))

                probability_data = []

                for class_id, probability in zip(classes, probabilities):
                    try:
                        key = int(class_id)
                    except (TypeError, ValueError):
                        key = class_id

                    probability_data.append({
                        "Kelas": CLASS_NAMES.get(key, str(key)),
                        "Probabilitas": float(probability),
                    })

                probability_df = pd.DataFrame(probability_data)
                probability_df = probability_df.sort_values(
                    "Probabilitas",
                    ascending=False,
                )

                st.markdown("### Probabilitas per kelas")
                st.dataframe(
                    probability_df.style.format({
                        "Probabilitas": "{:.2%}"
                    }),
                    use_container_width=True,
                )

                chart_data = probability_df.set_index("Kelas")
                st.bar_chart(chart_data)

            with st.expander("Lihat fitur yang dikirim ke model"):
                st.dataframe(input_df, use_container_width=True)

        except Exception as exc:
            st.error("Model gagal dimuat atau gagal melakukan prediksi.")
            st.exception(exc)

            st.markdown(
                """
                Periksa hal berikut:

                - File model benar-benar tersimpan dan tidak rusak.
                - Model berasal dari proses training yang benar.
                - Versi Python dan pustaka sesuai dengan lingkungan training.
                - Jumlah, urutan, skala, dan preprocessing fitur sama seperti saat training.
                - Jika error `STACK_GLOBAL requires str` muncul saat memuat
                  file `.pkl`, simpan ulang model dari lingkungan training
                  menggunakan format yang benar.
                """
            )


# =========================================================
# HALAMAN INFORMASI
# =========================================================

elif page == "Informasi":
    st.subheader("Informasi Proyek")

    st.markdown(
        """
        ### Kelas tutupan lahan

        | ID | Kelas |
        |---:|---|
        | 1 | Sawah |
        | 2 | Bangunan |
        | 3 | Mangrove |
        | 4 | Lahan Hijau |
        | 5 | Laut |
        | 6 | Danau/Ranu |

        ### Fitur model

        - Band Sentinel-2: B02, B03, B04, B08, B11
        - Indeks spektral: NDVI, NDWI, MNDWI, NDBI

        ### Catatan

        Hasil prediksi pada halaman ini merupakan prediksi untuk satu
        baris input fitur. Ini bukan klasifikasi raster penuh.

        Cakupan peta bergantung pada ketersediaan raster dan area yang
        memiliki data valid. Area tanpa data tidak boleh dianggap sebagai
        kelas tutupan lahan.
        """
    )

st.sidebar.divider()
st.sidebar.caption("Dashboard berbasis Streamlit")
