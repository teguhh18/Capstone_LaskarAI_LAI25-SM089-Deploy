import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image
import os
from tensorflow import keras
import base64 # untuk ubah gambar jadi format yang bisa dibaca HTML
import json

# —————— CONFIG ——————
st.set_page_config(page_title="Flood Prediction App (Prototype)", layout="wide")

# —————— HEADER ——————
col_title = st.columns(1)[0]
with col_title:
    st.markdown(
        "<h1 style='margin-bottom:0;'>Flood Prediction</h1>"
        "<div>Ketahui potensi risiko banjir di sekitar Anda. Mari bersama ciptakan masyarakat yang lebih siap menghadapi bencana dengan teknologi prediktif</div>",
        unsafe_allow_html=True,
    )
st.markdown("---")


# —————— LOAD MODEL ——————
@st.cache_resource
def load_my_model(model_path):
    try:
        model = keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None


# Path ke file model .h5
MODEL_PATH = "flood_model_f.h5"
model = load_my_model(MODEL_PATH)

# —————— END LOAD MODEL ——————


# —————— LOAD SCALER ——————
@st.cache_resource
def load_scaler(scaler_path):
    try:
        scaler = joblib.load(scaler_path)
        return scaler
    except Exception as e:
        st.error(f"Gagal memuat scaler: {e}")
        return None


SCALER_PATH = "flood_scaler.pkl"
scaler = load_scaler(SCALER_PATH)
# —————— END LOAD SCALER ——————

st.sidebar.markdown("<h2 style='text-align: center;'>Menu</h2>", unsafe_allow_html=True)

with st.container():
    # Menampilkan home image
    # st.image("flood.jpg")

    # Menampilkan HOme image dengan custom size
    IMAGE_DIRECTORY = "."
    IMAGE_FILENAME = "flood.jpg"
    image_path = os.path.join(IMAGE_DIRECTORY, IMAGE_FILENAME)
    home_image = Image.open(image_path)
    home_image_resized = home_image.resize((1540, 640))
    landing_image = None
    st.image(home_image_resized, use_container_width=True)

    # —————— FORM INPUT DATA——————
    with st.form("flood_input_form", clear_on_submit=False):
        st.subheader("📝Silahkan Masukkan Data Untuk Prediksi Dibawah ini")

        col1, col2 = st.columns([1, 1])
        with col1:
            MonsoonIntensity = st.number_input(
                "Moon Soon Intensity", step=1, min_value=0
            )

            CoastalVulnerability = st.number_input(
                "Coastal Vulnerability", step=1, min_value=0
            )

            WetlandLoss = st.number_input(
                "WetlandLoss", step=1, min_value=0
            )

            Encroachments = st.number_input(
                "Encroachments", min_value=0, step=1
            )

        with col2:

            Urbanization = st.number_input("Urbanization", step=1, min_value=0)
            Siltation = st.number_input("Siltation", step=1, min_value=0)
            Deforestation = st.number_input("Deforestation", step=1, min_value=0)

        submitted = st.form_submit_button("Predict", type="primary")

    st.markdown("---")

    # —————— PREDIKSI ——————
    if submitted:
        # Bangun DataFrame
        df_input = pd.DataFrame(
            {
                "MonsoonIntensity": [MonsoonIntensity],
                "CoastalVulnerability": [CoastalVulnerability],
                "WetlandLoss": [WetlandLoss],
                "Urbanization": [Urbanization],
                "Encroachments": [Encroachments],
                "Siltation": [Siltation],
                "Deforestation": [Deforestation],
            }
        )
        # Environmental Degradation
        df_input['EnvironmentalDegradationScore'] = (df_input[['Deforestation', 'WetlandLoss', 'Urbanization']].mean(axis=1))
        df_input['RiverObstructionRisk'] = (df_input[['Encroachments', 'Siltation']].mean(axis=1))

        df_predict = df_input.drop(columns=['WetlandLoss', 'Encroachments', 'Urbanization'])

        # --- BAGIAN PERBAIKAN UNTUK MENGURUTKAN KOLOM ---
        # Definisikan urutan kolom yang benar sesuai kebutuhan model
        urutan_fitur_model = [
            'MonsoonIntensity',
            'CoastalVulnerability',
            'EnvironmentalDegradationScore',
            'RiverObstructionRisk',
            'Siltation',
            'Deforestation'
        ]

        # Terapkan urutan baru ke DataFrame
        df_predict = df_predict[urutan_fitur_model]

        # --- AKHIR DARI BAGIAN PERBAIKAN ---
        # st.write('data', df_predict)  #Test hasil pengurutan fitur/kolom

        # SCALING DATA
        scaled_input = scaler.transform(df_predict)

        # PREDIKSI
        prediction = model.predict(scaled_input)

        # Ambil nilai prediksi, ubah jadi float
        flood_probability = float(prediction.flatten()[0])
        # UBAH JADI BENTUK PERSENTASE
        flood_probability_percentage = flood_probability * 100

        # Kode untuk hasil prediksi
        st.markdown(f"""
            <h3 style='text-align: center; color: #2B7A78;'>
                🧾 Probabilitas Banjir: <span style="font-size: 28px;">{flood_probability_percentage:.2f}%</span>
            </h3>
            """, unsafe_allow_html=True)


        if prediction >= 0.8:
            # ALERT WARNING
            st.markdown(
                """
                    <div style="
                        background-color: #fff3cd;
                        color: #856404;
                        padding: 20px;
                        margin: 20px auto;
                        border: 1px solid #ffeeba;
                        border-radius: 5px;
                        max-width: 700px;
                        text-align: center;
                    ">
                        <h4 style="margin-top: 0;">⚠️ WARNING ALERT!!!</h4>
                        <h5 style="margin-top: 0;">Tingkat Risiko Banjir Tinggi</h5>
                        <p style="text-align: justify;">
                            Mohon tetap waspada dan segera amankan barang-barang penting. Pantau terus perkembangan cuaca dan ikuti arahan resmi dari pemerintah daerah apabila kondisi memburuk.
                        </p>
                    </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            # ALERT AMAN
            st.markdown(
                """
                    <div style="
                        background-color: #d4edda;
                        color: #155724;
                        padding: 20px;
                        margin: 20px auto;
                        border: 1px solid #c3e6cb;
                        border-radius: 5px;
                        max-width: 700px;
                        text-align: center;
                    ">
                        <h4 style="margin-top: 0;">✅ STATUS AMAN</h4>
                        <h5 style="margin-top: 0;">Tingkat risiko banjir rendah</h5>
                        <p style="text-align: justify;">
                            Cuaca dan kondisi lingkungan saat ini tergolong aman. Tetap waspada dan ikuti informasi resmi jika ada perubahan situasi.
                    </div>
                """,
                unsafe_allow_html=True,
            )


        # --- Mulai Blok Kode untuk Sosialisasi ---

        st.markdown("---")
        st.subheader("Sosialisasi")

        try:
            with open('sosialisasi.json', 'r', encoding='utf-8') as f:
                data_sosialisasi = json.load(f)
        except FileNotFoundError:
            st.error("File 'sosialisasi.json' tidak ditemukan")
            st.stop()

        # 2. FUNGSI UNTUK MEMBUAT KARTU (CARD) SECARA OTOMATIS
        def create_info_card(img_path, title, text):
            # Mengubah gambar menjadi format Base64 agar bisa ditampilkan di HTML
            with open(img_path, "rb") as f:
                data = f.read()
                encoded = base64.b64encode(data).decode()

            # HTML dan CSS untuk satu kartu
            card_html = f"""
            <div class="card">
                <img src="data:image/png;base64,{encoded}" alt="card image">
                <div class="card-body">
                    <h4>{title}</h4>
                    <p>{text}</p>
                </div>
            </div>
            """
            return card_html
        
        # 3. CSS UNTUK STYLING CARD
        #    Kode ini mendefinisikan tampilan dari kartu (border, bayangan, padding, dll)
        st.markdown("""
        <style>
            .card {
                border: 1px solid #ddd;      /* Garis tepi tipis */
                border-radius: 10px;         /* Sudut yang melengkung */
                box-shadow: 0 4px 8px rgba(0,0,0,0.1); /* Efek bayangan */
                transition: 0.3s;            /* Efek transisi saat hover */
                width: 100%;                 /* Lebar kartu memenuhi kolom */
                margin-bottom: 20px;         /* Jarak antar kartu di baris bawahnya */
                overflow: hidden;            /* Memastikan gambar tidak keluar dari sudut melengkung */
                display: flex;               /* Menggunakan flexbox untuk layout */
                flex-direction: column;      /* Arah item dari atas ke bawah */
                height: 100%;                /* Membuat semua kartu di baris yang sama memiliki tinggi yang sama */
            }
            .card:hover {
                box-shadow: 0 8px 16px rgba(0,0,0,0.2); /* Bayangan lebih jelas saat disentuh mouse */
            }
            .card img {
                width: 100%;                 /* Gambar memenuhi lebar kartu */
                object-fit: cover;           /* Gambar terpotong rapi jika rasio tidak pas */
                height: 150px;               /* Tinggi gambar tetap */
            }
            .card-body {
                padding: 15px;               /* Jarak konten dari tepi kartu */
                flex-grow: 1;                /* Memastikan body kartu mengisi ruang yang tersedia */
            }
            .card-body h4 {
                margin-top: 0;
                margin-bottom: 10px;
                font-weight: bold;
            }
            .card-body p {
                font-size: 14px;
                color: #555;
            }
        </style>
        """, unsafe_allow_html=True)

        # 4. MEMBUAT LAYOUT GRID DAN MENAMPILKAN KARTU
        row1_col1, row1_col2, row1_col3 = st.columns(3)
        with row1_col1:
            card_html = create_info_card(data_sosialisasi[0]['img_path'], data_sosialisasi[0]['title'], data_sosialisasi[0]['text'])
            st.markdown(card_html, unsafe_allow_html=True)

        with row1_col2:
            card_html = create_info_card(data_sosialisasi[1]['img_path'], data_sosialisasi[1]['title'], data_sosialisasi[1]['text'])
            st.markdown(card_html, unsafe_allow_html=True)

        with row1_col3:
            card_html = create_info_card(data_sosialisasi[2]['img_path'], data_sosialisasi[2]['title'], data_sosialisasi[2]['text'])
            st.markdown(card_html, unsafe_allow_html=True)

        # Baris Kedua
        row2_col1, row2_col2, row2_col3 = st.columns(3)
        with row2_col1:
            card_html = create_info_card(data_sosialisasi[3]['img_path'], data_sosialisasi[3]['title'], data_sosialisasi[3]['text'])
            st.markdown(card_html, unsafe_allow_html=True)

        with row2_col2:
            card_html = create_info_card(data_sosialisasi[4]['img_path'], data_sosialisasi[4]['title'], data_sosialisasi[4]['text'])
            st.markdown(card_html, unsafe_allow_html=True)

        with row2_col3:
            card_html = create_info_card(data_sosialisasi[5]['img_path'], data_sosialisasi[5]['title'], data_sosialisasi[5]['text'])
            st.markdown(card_html, unsafe_allow_html=True)

        # --- Akhir Blok Kode ---
            
    else:
        st.info(
            "🔎 Masukkan nilai fitur di atas, lalu klik **Predict** untuk melihat hasil."
        )

# —————— FOOTER ——————
st.markdown("---")

st.markdown(
    "<p style='text-align: center; color: gray;'>"
    "© 2025 Flood Prediction Capstone"
    "</p>",
    unsafe_allow_html=True,
)
