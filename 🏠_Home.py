import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image
import os
from tensorflow import keras
import base64  # untuk ubah gambar jadi format yang bisa dibaca HTML
import json

# —————— CONFIG ——————
st.set_page_config(page_title="Flood Prediction App (Prototype)", layout="wide")


# —————— FUNGSI-FUNGSI OPTIMASI ——————

# --- FUNGSI BARU DENGAN CACHE UNTUK GAMBAR ---
# Fungsi ini hanya akan berjalan sekali untuk setiap gambar.
# Hasilnya disimpan di memori (cache) untuk pemanggilan berikutnya.
@st.cache_data
def get_image_as_base64(file_path):
    """Fungsi ini membaca file gambar dan meng-encode ke base64."""
    try:
        with open(file_path, "rb") as f:
            data = f.read()
            return base64.b64encode(data).decode()
    except FileNotFoundError:
        st.error(f"Gambar tidak ditemukan di: {file_path}")
        return None

# --- FUNGSI KARTU YANG SUDAH DIOPTIMALKAN ---
# Fungsi ini sekarang hanya bertugas membuat HTML, tidak lagi membaca file.
def create_info_card(encoded_img, title, text):
    """Fungsi ini membuat HTML untuk kartu dari data yang sudah siap."""
    if encoded_img is None:
        return ""  # Jangan tampilkan kartu jika gambar gagal di-load

    card_html = f"""
    <div class="card">
        <img src="data:image/png;base64,{encoded_img}" alt="card image">
        <div class="card-body">
            <h4>{title}</h4>
            <p>{text}</p>
        </div>
    </div>
    """
    return card_html


# —————— LOAD MODEL & SCALER (DENGAN CACHE) ——————
@st.cache_resource
def load_my_model(model_path):
    try:
        model = keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

@st.cache_resource
def load_scaler(scaler_path):
    try:
        scaler = joblib.load(scaler_path)
        return scaler
    except Exception as e:
        st.error(f"Gagal memuat scaler: {e}")
        return None

# Path ke file-file
MODEL_PATH = "flood_model_f.h5"
SCALER_PATH = "flood_scaler.pkl"
model = load_my_model(MODEL_PATH)
scaler = load_scaler(SCALER_PATH)


# —————— HEADER ——————
st.markdown(
    "<h1 style='margin-bottom:0;'>Flood Prediction</h1>"
    "<div>Ketahui potensi risiko banjir di sekitar Anda. Mari bersama ciptakan masyarakat yang lebih siap menghadapi bencana dengan teknologi prediktif</div>",
    unsafe_allow_html=True,
)
st.markdown("---")


# —————— TAMPILAN UTAMA ——————
st.sidebar.markdown("<h2 style='text-align: center;'>Menu</h2>", unsafe_allow_html=True)

with st.container():
    # Menampilkan Home image dengan custom size
    try:
        home_image = Image.open("flood.jpg")
        home_image_resized = home_image.resize((1540, 640))
        st.image(home_image_resized, use_container_width=True)
    except FileNotFoundError:
        st.warning("Gambar utama 'flood.jpg' tidak ditemukan.")

    # —————— FORM INPUT DATA ——————
    with st.form("flood_input_form", clear_on_submit=False):
        st.subheader("📝 Silahkan Masukkan Data Untuk Prediksi Dibawah ini")

        col1, col2 = st.columns([1, 1])
        with col1:
            MonsoonIntensity = st.number_input("Moon Soon Intensity", step=1, min_value=0)
            CoastalVulnerability = st.number_input("Coastal Vulnerability", step=1, min_value=0)
            WetlandLoss = st.number_input("WetlandLoss", step=1, min_value=0)
            Encroachments = st.number_input("Encroachments", min_value=0, step=1)
        with col2:
            Urbanization = st.number_input("Urbanization", step=1, min_value=0)
            Siltation = st.number_input("Siltation", step=1, min_value=0)
            Deforestation = st.number_input("Deforestation", step=1, min_value=0)

        submitted = st.form_submit_button("Predict", type="primary")

    st.markdown("---")

    # —————— BLOK PREDIKSI & HASIL ——————
    if submitted:
        if not all([model, scaler]):
             st.error("Model atau Scaler gagal dimuat. Tidak dapat melakukan prediksi.")
        else:
            # Bangun DataFrame dari input
            df_input = pd.DataFrame({
                "MonsoonIntensity": [MonsoonIntensity],
                "CoastalVulnerability": [CoastalVulnerability],
                "WetlandLoss": [WetlandLoss],
                "Urbanization": [Urbanization],
                "Encroachments": [Encroachments],
                "Siltation": [Siltation],
                "Deforestation": [Deforestation],
            })

            # Feature Engineering
            df_input['EnvironmentalDegradationScore'] = df_input[['Deforestation', 'WetlandLoss', 'Urbanization']].mean(axis=1)
            df_input['RiverObstructionRisk'] = df_input[['Encroachments', 'Siltation']].mean(axis=1)
            
            df_predict = df_input.drop(columns=['WetlandLoss', 'Encroachments', 'Urbanization'])
            
            # Mengurutkan kolom sesuai kebutuhan model
            urutan_fitur_model = [
                'MonsoonIntensity', 'CoastalVulnerability', 'EnvironmentalDegradationScore',
                'RiverObstructionRisk', 'Siltation', 'Deforestation'
            ]
            df_predict = df_predict[urutan_fitur_model]

            # SCALING DATA & PREDIKSI
            scaled_input = scaler.transform(df_predict)
            prediction = model.predict(scaled_input)
            flood_probability_percentage = float(prediction.flatten()[0]) * 100

            # Tampilkan Hasil Prediksi
            st.markdown(f"""
                <h3 style='text-align: center; color: #2B7A78;'>
                    🧾 Probabilitas Banjir: <span style="font-size: 28px;">{flood_probability_percentage:.2f}%</span>
                </h3>
                """, unsafe_allow_html=True)
            
            # Tampilkan Alert Sesuai Hasil
            if prediction >= 0.8:
                st.markdown("""
                    <div style="background-color: #fff3cd; color: #856404; padding: 20px; margin: 20px auto; border: 1px solid #ffeeba; border-radius: 5px; max-width: 700px; text-align: center;">
                        <h4 style="margin-top: 0;">⚠️ WARNING ALERT!!!</h4>
                        <h5 style="margin-top: 0;">Tingkat Risiko Banjir Tinggi</h5>
                        <p style="text-align: justify;">Mohon tetap waspada dan segera amankan barang-barang penting. Pantau terus perkembangan cuaca dan ikuti arahan resmi dari pemerintah daerah apabila kondisi memburuk.</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div style="background-color: #d4edda; color: #155724; padding: 20px; margin: 20px auto; border: 1px solid #c3e6cb; border-radius: 5px; max-width: 700px; text-align: center;">
                        <h4 style="margin-top: 0;">✅ STATUS AMAN</h4>
                        <h5 style="margin-top: 0;">Tingkat risiko banjir rendah</h5>
                        <p style="text-align: justify;">Cuaca dan kondisi lingkungan saat ini tergolong aman. Tetap waspada dan ikuti informasi resmi jika ada perubahan situasi.</p>
                    </div>
                    """, unsafe_allow_html=True)

            # --- BLOK SOSIALISASI YANG SUDAH DIOPTIMALKAN ---
            st.markdown("---")
            st.subheader("Sosialisasi")

            try:
                with open('sosialisasi.json', 'r', encoding='utf-8') as f:
                    data_sosialisasi = json.load(f)

                # Masukkan CSS untuk styling kartu
                st.markdown("""
                <style>
                    .card {
                        border: 1px solid #ddd; border-radius: 10px;
                        box-shadow: 0 4px 8px rgba(0,0,0,0.1); transition: 0.3s;
                        width: 100%; margin-bottom: 20px; overflow: hidden;
                        display: flex; flex-direction: column; height: 100%;
                    }
                    .card:hover { box-shadow: 0 8px 16px rgba(0,0,0,0.2); }
                    .card img { width: 100%; object-fit: cover; height: 150px; }
                    .card-body { padding: 15px; flex-grow: 1; }
                    .card-body h4 { margin-top: 0; margin-bottom: 10px; font-weight: bold; }
                    .card-body p { font-size: 14px; color: #555; }
                </style>
                """, unsafe_allow_html=True)

                # Membuat layout grid dan menampilkan kartu secara dinamis
                cols = st.columns(3)
                for i, item in enumerate(data_sosialisasi):
                    with cols[i % 3]:
                        # Panggil fungsi yang di-cache untuk mendapatkan base64
                        encoded_image = get_image_as_base64(item['img_path'])
                        # Buat HTML card dengan data yang sudah siap
                        card_html = create_info_card(encoded_image, item['title'], item['text'])
                        st.markdown(card_html, unsafe_allow_html=True)
            
            except FileNotFoundError:
                st.error("File 'sosialisasi.json' tidak ditemukan.")
            except Exception as e:
                st.error(f"Terjadi kesalahan saat menampilkan bagian sosialisasi: {e}")

    else:
        st.info("🔎 Masukkan nilai fitur di atas, lalu klik **Predict** untuk melihat hasil.")

# —————— FOOTER ——————
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>© 2025 Flood Prediction Capstone</p>",
    unsafe_allow_html=True,
)