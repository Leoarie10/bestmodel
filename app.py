import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(
    page_title="Layoff Prediction App",
    page_icon="💼",
    layout="centered"
)

# --- 1. Fungsi Load Model ---
@st.cache_resource
def load_assets():
    try:
        # Memuat model
        model = joblib.load('rf_model_compressed.pkl')
        
        # Memuat label encoder (untuk target variable)
        # Sesuai file yang Anda upload, isinya: ['Large', 'Medium', 'Small']
        encoder = joblib.load('label_encoder.pkl')
        
        return model, encoder
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None

model, label_encoder = load_assets()

# --- 2. Judul dan Deskripsi ---
st.title("💼 Layoff Prediction App")
st.write("""
Aplikasi ini memprediksi kategori/skala layoff berdasarkan data perusahaan 
(Industri, Lokasi, Funding, dll).
""")
st.markdown("---")

# --- 3. Form Input User ---
# Menggunakan kolom agar tampilan lebih rapi
col1, col2 = st.columns(2)

with col1:
    industry = st.text_input("Industry", value="Retail", help="Contoh: Retail, Tech, Finance")
    country = st.text_input("Country", value="United States")
    location = st.text_input("Location", value="SF Bay Area")
    stage = st.selectbox("Stage", ["Series A", "Series B", "IPO", "Acquired", "Unknown", "Seed"])

with col2:
    source = st.text_input("Source", value="TechCrunch", help="Sumber berita")
    funds_raised = st.number_input("Funds Raised (in Millions)", min_value=0.0, value=50.0, step=1.0)
    year = st.number_input("Year", min_value=2020, max_value=2030, value=2023)

# --- 4. Tombol Prediksi ---
if st.button("🔍 Prediksi Sekarang", type="primary"):
    if model is not None:
        # Membuat DataFrame dari input user
        # Nama kolom HARUS sama persis dengan yang digunakan saat training di Notebook
        input_data = pd.DataFrame({
            'industry': [industry],
            'country': [country],
            'stage': [stage],
            'location': [location],
            'source': [source],
            'funds_raised': [funds_raised],
            'year': [year]
        })

        st.write("Data Input:")
        st.dataframe(input_data)

        try:
            # Melakukan prediksi
            prediction_index = model.predict(input_data)
            
            # Mengubah hasil angka kembali ke teks (Decode)
            # Jika model langsung mengeluarkan string, baris ini mungkin tidak perlu, 
            # tapi karena ada label_encoder.pkl, kemungkinan output model adalah angka (0, 1, 2)
            if hasattr(label_encoder, 'inverse_transform'):
                prediction_label = label_encoder.inverse_transform(prediction_index)
                final_result = prediction_label[0]
            else:
                final_result = prediction_index[0]

            # Menampilkan Hasil
            st.success(f"### Hasil Prediksi: {final_result}")
            
            # Menampilkan Probability/Confidence jika model mendukung
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(input_data)[0]
                st.write("Confidence Score:")
                
                # Mapping probabilitas ke nama kelas
                classes = label_encoder.classes_ if hasattr(label_encoder, 'classes_') else model.classes_
                prob_df = pd.DataFrame(proba, index=classes, columns=["Probability"])
                st.bar_chart(prob_df)

        except Exception as e:
            st.error("Terjadi kesalahan saat memprediksi.")
            st.warning(f"Detail Error: {e}")
            st.info("Tips: Jika error menyebutkan masalah konversi string ke float, berarti model Anda belum berupa Pipeline yang menangani OneHotEncoding.")
    else:
        st.error("Model belum dimuat. Pastikan file .pkl ada di folder yang sama.")

# Footer
st.markdown("---")
st.caption("Group 3 Final Project - Deployment")
