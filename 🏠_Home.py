import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import joblib
import os
from tensorflow import keras

# —————— CONFIG ——————
st.set_page_config(page_title="Flood Prediction App (Prototype)", layout="wide")


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

MODEL_PATH = "flood_model_f.h5"
SCALER_PATH = "flood_scaler.pkl"
model = load_my_model(MODEL_PATH)
scaler = load_scaler(SCALER_PATH)

# —————— CSS STYLING ——————
st.markdown("""
    <style>
    .custom-card {
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 8px #999;
        margin-bottom: 15px;
        background-color: var(--secondary-background-color);
        color: var(--text-color);
        transition: transform 0.2s;
    }
    </style>
""", unsafe_allow_html=True)


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
    try:
        # st.image("flood_resize.jpg", use_container_width=True)
        # Menampilkan HOme image dengan custom size
        IMAGE_DIRECTORY = "."
        IMAGE_FILENAME = "flood.jpg"
        image_path = os.path.join(IMAGE_DIRECTORY, IMAGE_FILENAME)
        home_image = Image.open(image_path)
        home_image_resized = home_image.resize((1540, 640))
        landing_image = None
        st.image(home_image_resized, use_container_width=True)
    except FileNotFoundError:
        st.warning("Gambar utama 'flood_resize.jpg' tidak ditemukan.")

    with st.form("flood_input_form", clear_on_submit=False):
        st.subheader("📝 Silahkan Masukkan Data Untuk Prediksi Dibawah ini")
        st.info("**Petunjuk:** Untuk semua fitur di bawah ini, silakan masukkan nilai antara **0** hingga **30**. Semakin tinggi nilai yang diinput, berarti semakin tinggi tingkat keparahannya", icon="ℹ️")
        col1, col2 = st.columns([1, 1])
        with col1:
            MonsoonIntensity = st.number_input("Intensitas Curah Hujan", step=1, min_value=0, max_value=30, help="Menjelaskan seberapa tinggi curah hujan saat musim muson. Nilai 0 berarti hujan ringan, nilai 25 berarti hujan sangat lebat dan ekstrem.")
            CoastalVulnerability = st.number_input("Kerentanan Wilayah Pesisir", step=1, min_value=0, max_value=30, help="Menggambarkan risiko banjir dari laut (banjir rob). Nilai 0 berarti wilayah pesisir aman, nilai 25 berarti sangat rentan terhadap pasang laut ekstrem.")
            WetlandLoss = st.number_input("Kehilangan Lahan Basah", step=1, min_value=0, max_value=30, help="Mengukur seberapa banyak area resapan air alami (rawa, danau) yang hilang. Lahan basah berfungsi seperti spons untuk menyerap air hujan.")
            Encroachments = st.number_input("Perambahan Lahan (Pembangunan Liar)", min_value=0, step=1, max_value=30, help="Tingkat pembangunan di area terlarang (misal: bantaran sungai) yang dapat menyumbat aliran air. Nilai 0 berarti tidak ada, 25 berarti sangat banyak.")
        with col2:
            Urbanization = st.number_input("Kepadatan Penduduk (Urbanisasi)", step=1, min_value=0, max_value=30, help="Seberapa padat wilayah dengan bangunan dan jalan beton/aspal. Semakin tinggi nilainya, semakin sedikit tanah yang bisa menyerap air.")
            Siltation = st.number_input("Pendangkalan Sungai (Sedimentasi)", step=1, min_value=0, max_value=30, help="Tingkat pendangkalan sungai akibat endapan lumpur/pasir. Sungai yang dangkal (nilai tinggi) tidak dapat menampung banyak air dan mudah meluap.")
            Deforestation = st.number_input("Deforestasi (Penggundulan Hutan)", step=1, min_value=0, max_value=30, help="Mengukur tingkat penggundulan hutan. Hutan yang gundul (nilai tinggi) kehilangan kemampuan untuk menyerap dan menahan air hujan.")

        submitted = st.form_submit_button("Prediksi", type="primary")
    st.markdown("---")

    if submitted:
        if not all([model, scaler]):
            st.error("Model atau Scaler gagal dimuat. Tidak dapat melakukan prediksi.")
        else:
            df_input = pd.DataFrame({
                "MonsoonIntensity": [MonsoonIntensity], "CoastalVulnerability": [CoastalVulnerability],
                "WetlandLoss": [WetlandLoss], "Urbanization": [Urbanization],
                "Encroachments": [Encroachments], "Siltation": [Siltation], "Deforestation": [Deforestation],
            })
            df_input['EnvironmentalDegradationScore'] = df_input[['Deforestation', 'WetlandLoss', 'Urbanization']].mean(axis=1)
            df_input['RiverObstructionRisk'] = df_input[['Encroachments', 'Siltation']].mean(axis=1)
            df_predict = df_input.drop(columns=['WetlandLoss', 'Encroachments', 'Urbanization'])
            urutan_fitur_model = ['MonsoonIntensity', 'CoastalVulnerability', 'EnvironmentalDegradationScore', 'RiverObstructionRisk', 'Siltation', 'Deforestation']
            df_predict = df_predict[urutan_fitur_model]
            scaled_input = scaler.transform(df_predict)
            prediction = model.predict(scaled_input)
            flood_probability_percentage = float(prediction.flatten()[0]) * 100

            st.markdown(f"""
                <h3 style='text-align: center; color: #2B7A78;'>
                    🧾 Probabilitas Banjir: <span style="font-size: 28px;">{flood_probability_percentage:.2f}%</span>
                </h3>
                """, unsafe_allow_html=True)
            
            if prediction >= 0.8:
                st.markdown("""
                    <div style="background-color: #fff3cd; color: #856404; padding: 20px; margin: 20px auto; border: 1px solid #ffeeba; border-radius: 5px; max-width: 700px; text-align: center;">
                        <h4 style="margin-top: 0;">⚠️ PERINGATAN !!!</h4> <h5 style="margin-top: 0;">Tingkat Risiko Banjir Tinggi</h5>
                        <p style="text-align: justify;">Mohon tetap waspada dan segera amankan barang-barang penting. Pantau terus perkembangan cuaca dan ikuti arahan resmi dari pemerintah daerah apabila kondisi memburuk.</p>
                    </div> """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div style="background-color: #d4edda; color: #155724; padding: 20px; margin: 20px auto; border: 1px solid #c3e6cb; border-radius: 5px; max-width: 700px; text-align: center;">
                        <h4 style="margin-top: 0;">✅ STATUS AMAN</h4> <h5 style="margin-top: 0;">Tingkat risiko banjir rendah</h5>
                        <p style="text-align: justify;">Cuaca dan kondisi lingkungan saat ini tergolong aman. Tetap waspada dan ikuti informasi resmi jika ada perubahan situasi.</p>
                    </div>""", unsafe_allow_html=True)

            # —————— KODE CARD SOSIALISASI ——————
            st.markdown("---")
            st.subheader("Sosialisasi")

            cols = st.columns(3)

            # ——— CARD 1
            with cols[0]:
                st.markdown("""
                    <div class="custom-card">
                        <img src="https://raw.githubusercontent.com/teguhh18/image-streamlit/main/monsoon.jpg" width="100%" height="200px" style="border-radius:10px; object-fit:cover;">
                        <h4>Intensitas Muson</h4>
                        <p>Musim muson membawa curah hujan tinggi yang bisa menyebabkan banjir besar, terutama di daerah dataran rendah. Masyarakat perlu memahami pola cuaca musiman dan meningkatkan kesiapan saat intensitas muson meningkat, seperti membersihkan saluran air dan tidak membuang sampah sembarangan.</p>
                    </div>
                """, unsafe_allow_html=True)

                st.markdown("""
                    <div class="custom-card">
                        <img src="https://raw.githubusercontent.com/teguhh18/image-streamlit/main/Siltation.jpg" width="100%" height="200px" style="border-radius:10px; object-fit:cover;">
                        <h4>Pendangkalan Sungai</h4>
                        <p>Endapan lumpur atau pasir yang berlebihan di sungai memperkecil kapasitas aliran air. Sosialisasi bertujuan untuk mengedukasi pentingnya reboisasi dan pengelolaan lahan agar sedimentasi dapat diminimalisir dan tidak memicu banjir.</p>
                    </div>
                """, unsafe_allow_html=True)
                


            # ——— CARD 2
            with cols[1]:
                st.markdown("""
                    <div class="custom-card">
                        <img src="https://raw.githubusercontent.com/teguhh18/image-streamlit/main/RiverObstructionRisk.jpg" width="100%" height="200px" style="border-radius:10px; object-fit:cover;">
                        <h4>Risiko Sumbatan Sungai</h4>
                        <p>Sampah, endapan lumpur, dan bangunan liar di sepanjang aliran sungai dapat menghambat aliran air, memicu banjir secara tiba-tiba. Edukasi diperlukan agar masyarakat ikut menjaga kebersihan sungai dan tidak membangun di bantaran sungai tanpa izin</p>
                    </div>
                """, unsafe_allow_html=True)

                st.markdown("""
                    <div class="custom-card">
                        <img src="https://raw.githubusercontent.com/teguhh18/image-streamlit/main/EnvironmentalDegradation.jpg" width="100%" height="200px" style="border-radius:10px; object-fit:cover;">
                        <h4>Kerusakan Lingkungan</h4>
                        <p>Aktivitas seperti illegal logging, pembukaan lahan berlebihan, dan pencemaran memperburuk daya serap tanah. Sosialisasi harus menekankan pentingnya konservasi lingkungan dan pengawasan terhadap aktivitas yang merusak ekosistem.</p>
                    </div>
                """, unsafe_allow_html=True)


            # ——— CARD 3
            with cols[2]:
                st.markdown("""
                    <div class="custom-card">
                        <img src="https://raw.githubusercontent.com/teguhh18/image-streamlit/main/Coastal.jpg" width="100%" height="200px" style="border-radius:10px; object-fit:cover;">
                        <h4>Kerentanan Wilayah Pesisir</h4>
                        <p>Wilayah pesisir sangat rentan terhadap banjir akibat pasang laut dan badai tropis. Sosialisasi penting dilakukan kepada warga pesisir terkait jalur evakuasi, sistem peringatan dini, serta pentingnya penanaman vegetasi penahan abrasi seperti mangrove.</p>
                    </div>
                """, unsafe_allow_html=True)

                st.markdown("""
                    <div class="custom-card">
                        <img src="https://raw.githubusercontent.com/teguhh18/image-streamlit/main/Deforestation.jpg" width="100%" height="200px" style="border-radius:10px; object-fit:cover;">
                        <h4>Penggundulan Hutan</h4>
                        <p>Hutan yang gundul kehilangan kemampuan menahan air hujan, menyebabkan limpasan permukaan yang lebih besar. Edukasi penting agar masyarakat memahami fungsi hutan sebagai pelindung banjir alami dan mendukung kegiatan penghijauan.</p>
                    </div>
                """, unsafe_allow_html=True)

    else:
        st.info("🔎 Masukkan nilai fitur di atas, lalu klik **Predict** untuk melihat hasil.")

# —————— FOOTER ——————
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>© 2025 Flood Prediction Capstone</p>",
    unsafe_allow_html=True,
)