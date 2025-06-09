import streamlit as st

st.title("📬 Contact Us")
st.markdown(
    """
    Jika Anda memiliki pertanyaan, saran, atau ingin berkolaborasi, silakan hubungi anggota tim kami melalui email atau LinkedIn berikut:
    """
)

# Data anggota
team_members = [
    {
        "name": "Ahmad Noval",
        "role": "Web Development",
        "email": "ahmdnvl11@gmail.com",
        "linkedin": "http://www.linkedin.com/in/ahmadnoval"
    },
    {
        "name": "Khafka Fadillah Wibawa Nurdiansyah",
        "role": "Machine Learning",
        "email": "khafka.fadillahww@gmail.com",
        "linkedin": "http://www.linkedin.com/in/khafkafadillah"
    },
    {
        "name": "Nafa Khairunnisah",
        "role": "Machine Learning",
        "email": "nkhairunn2412@gmail.com",
        "linkedin": "http://www.linkedin.com/in/nafa-khairunnisa"
    },
    {
        "name": "Teguh Budiono",
        "role": "Web Development",
        "email": "teguhbudiono147@gmail.com",
        "linkedin": "https://www.linkedin.com/in/teguhh18/"
    },
]

for member in team_members:
    st.markdown("---")
    col1, col2 = st.columns([1, 5])
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=50)
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
