import streamlit as st

st.title("About")
st.write("Flood Prediction App adalah aplikasi prototipe yang dirancang untuk memprediksi kemungkinan terjadinya banjir berdasarkan data lingkungan dan infrastruktur. Aplikasi ini dibangun menggunakan teknologi Machine Learning dan dikembangkan sebagai bagian dari proyek capstone yang bertujuan untuk membantu mitigasi bencana banjir dengan menyediakan sistem pendukung keputusan yang cepat dan mudah digunakan.")

st.subheader("🎯 Tujuan")
st.write("Aplikasi ini bertujuan untuk memberikan informasi yang akurat dan cepat mengenai risiko banjir di suatu wilayah, sehingga masyarakat dapat mengambil tindakan pencegahan yang diperlukan. Dengan menggunakan data lingkungan dan algoritma Machine Learning, aplikasi ini dapat memprediksi kemungkinan terjadinya banjir berdasarkan berbagai faktor seperti curah hujan, kondisi lingkungan yang ada.")

st.subheader("🧠 Teknologi yang Digunakan")
st.markdown("""
- Model ML: Keras + TensorFlow, model yang dibangun berbasis data lingkungan.
- Frontend: Streamlit (Python Web App Framework).
- Tools: Google Colaboratory, Streamlit, Git.
- Library Pendukung: OS, Shutil, Pandas, Numpy, Matplotlib, Seaborn, Tensorflow, Google.colab.drive, Scikit-learn, dan Scikeras.
""")

st.subheader("📁 Dataset")
st.write("Model dilatih menggunakan dataset dari : **Kaggle Flood Prediction Dataset oleh Naiya Khalid** https://www.kaggle.com/datasets/naiyakhalid/flood-prediction-dataset Dataset ini berisi berbagai fitur terkait curah hujan, kelembapan, dan parameter geografis yang relevan dengan risiko banjir.")

st.subheader("👨‍💻 Pengembang")
st.markdown("""
- Nama Proyek: Flood Prediction Capstone
- Tim Pengembang:
  - Ahmad Noval (Universitas Mercu Buana)
  - Khafka Fadillah Wibawa Nurdiansyah (UIN Sunan Gunung Djati Bandung)
  - Nafa Khairunnisa (UIN Sunan Gunung Djati Bandung)
  - Teguh Budiono (Universitas Teknokrat Indonesia)
- GitHub: https://github.com/teguhh18/Capstone_LaskarAI_LAI25-SM089
""")