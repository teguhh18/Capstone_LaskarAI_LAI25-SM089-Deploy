import streamlit as st

st.title("📬 Contact Us")
st.markdown(
    """
    Jika Anda memiliki pertanyaan, saran, atau ingin berkolaborasi, silakan hubungi anggota tim kami melalui email atau LinkedIn berikut:
    """
)

# Menambahkan sedikit CSS untuk membuat gambar menjadi bulat
st.markdown("""
<style>
img {
    border-radius: 50%;
}
</style>
""", unsafe_allow_html=True)


# Data anggota
team_members = [
    {
        "name": "Ahmad Noval",
        "role": "Web Developer",
        "email": "ahmdnvl11@gmail.com",
        "linkedin": "http://www.linkedin.com/in/ahmadnoval",
        "foto": "https://raw.githubusercontent.com/teguhh18/image-streamlit/main/Ahmad-Noval.jpg"
    },
    {
        "name": "Khafka Fadillah Wibawa Nurdiansyah",
        "role": "Machine Learning Developer",
        "email": "khafka.fadillahww@gmail.com",
        "linkedin": "http://www.linkedin.com/in/khafkafadillah",
        "foto": "https://raw.githubusercontent.com/teguhh18/image-streamlit/main/khafka.jpg"
    },
    {
        "name": "Nafa Khairunnisa",
        "role": "Machine Learning Developer",
        "email": "nkhairunn2412@gmail.com",
        "linkedin": "http://www.linkedin.com/in/nafa-khairunnisa",
        "foto": "https://raw.githubusercontent.com/teguhh18/image-streamlit/main/nafa.JPG"
    },
    {
        "name": "Teguh Budiono",
        "role": "Web Developer",
        "email": "teguhbudiono147@gmail.com",
        "linkedin": "https://www.linkedin.com/in/teguhh18/",
        "foto": "https://raw.githubusercontent.com/teguhh18/image-streamlit/main/teguh_resize.jpg"
    },
]

# URL untuk gambar placeholder jika foto tidak tersedia
placeholder_image = "https://www.w3schools.com/howto/img_avatar.png"

for member in team_members:
    st.markdown("---")
    col1, col2 = st.columns([1, 4]) # Mengubah rasio agar lebih seimbang

    with col1:
        # --- BAGIAN YANG DIPERBAIKI ---
        # 1. Gunakan .get() untuk mengambil URL foto dengan aman.
        #    Jika key 'foto' tidak ada, ia akan mengembalikan None.
        foto_url = member.get("foto")

        # 2. Cek apakah URL foto ada. Jika tidak, gunakan placeholder.
        if foto_url:
            # 3. Hapus kurung kurawal {} dan naikkan width agar lebih jelas
            st.image(foto_url, width=100)
        else:
            st.image(placeholder_image, width=100)
        # --- AKHIR BAGIAN PERBAIKAN ---

    with col2:
        st.markdown(f"""
        **{member['name']}**  
        _{member['role']}_  
        📧 Email: [{member['email']}](mailto:{member['email']})  
        🔗 LinkedIn: [Lihat Profil]({member['linkedin']})
        """)

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>"
    "Terima kasih telah menggunakan aplikasi kami. 💙"
    "</p>", unsafe_allow_html=True)