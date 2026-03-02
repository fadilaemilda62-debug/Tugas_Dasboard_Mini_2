import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# =====================================================
# KONFIGURASI HALAMAN
# =====================================================
st.set_page_config(
    page_title="Mini Dashboard Analisis",
    layout="wide"
)

st.title("📊 Mini Dashboard Analisis Data Siswa")

# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data
def load_data():
    df = pd.read_excel("data_simulasi_50_siswa_20_soal.xlsx")
    return df

df = load_data()

# =====================================================
# PREPROCESSING
# =====================================================
# Hapus kolom responden jika ada
if "Responden" in df.columns:
    data = df.drop(columns=["Responden"])
else:
    data = df.copy()

# Pastikan numerik
data = data.select_dtypes(include=np.number)

# Tambah total skor
data["Total_Skor"] = data.sum(axis=1)

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.header("⚙️ Pengaturan Analisis")

target = st.sidebar.selectbox(
    "Pilih Variabel Target (Y)",
    data.columns
)

# OPSI FITUR (FIX ERROR STREAMLIT)
opsi_fitur = [col for col in data.columns if col != target]

default_fitur = opsi_fitur[:min(3, len(opsi_fitur))]

fitur = st.sidebar.multiselect(
    "Pilih Variabel Prediktor (X)",
    opsi_fitur,
    default=default_fitur
)

# =====================================================
# TAB DASHBOARD
# =====================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Statistik",
    "🔥 Korelasi",
    "📈 Regresi",
    "📊 Visualisasi"
])

# =====================================================
# TAB 1 — STATISTIK
# =====================================================
with tab1:

    st.subheader("Statistik Deskriptif")

    st.dataframe(data.describe(), use_container_width=True)

    st.subheader("Preview Data")
    st.dataframe(data.head(), use_container_width=True)

# =====================================================
# TAB 2 — KORELASI
# =====================================================
with tab2:

    st.subheader("Heatmap Korelasi")

    corr = data.corr()

    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(corr, cmap="coolwarm", ax=ax)

    st.pyplot(fig)

# =====================================================
# TAB 3 — REGRESI LINEAR
# =====================================================
with tab3:

    st.subheader("Analisis Regresi Linear")

    if len(fitur) > 0:

        X = data[fitur]
        y = data[target]

        model = LinearRegression()
        model.fit(X, y)

        prediksi = model.predict(X)

        # ======================
        # KOEFISIEN
        # ======================
        coef_df = pd.DataFrame({
            "Variabel": fitur,
            "Koefisien": model.coef_
        })

        st.write("### Koefisien Regresi")
        st.dataframe(coef_df, use_container_width=True)

        st.write("### Intercept")
        st.success(round(model.intercept_, 4))

        # ======================
        # R2 SCORE
        # ======================
        r2 = model.score(X, y)

        st.write("### R² Score")
        st.info(round(r2, 4))

        # ======================
        # PLOT PREDIKSI
        # ======================
        fig2, ax2 = plt.subplots()
        ax2.scatter(y, prediksi)
        ax2.set_xlabel("Nilai Aktual")
        ax2.set_ylabel("Nilai Prediksi")
        ax2.set_title("Aktual vs Prediksi")

        st.pyplot(fig2)

    else:
        st.warning("⚠️ Pilih minimal satu variabel prediktor.")

# =====================================================
# TAB 4 — VISUALISASI
# =====================================================
with tab4:

    st.subheader("Distribusi Total Skor")

    fig3, ax3 = plt.subplots()
    ax3.hist(data["Total_Skor"], bins=10)
    ax3.set_xlabel("Total Skor")
    ax3.set_ylabel("Frekuensi")

    st.pyplot(fig3)

    st.subheader("Scatter Plot Interaktif")

    col1, col2 = st.columns(2)

    with col1:
        x_axis = st.selectbox("Sumbu X", data.columns)

    with col2:
        y_axis = st.selectbox("Sumbu Y", data.columns, index=1)

    fig4, ax4 = plt.subplots()
    ax4.scatter(data[x_axis], data[y_axis])
    ax4.set_xlabel(x_axis)
    ax4.set_ylabel(y_axis)

    st.pyplot(fig4)

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.caption("Mini Dashboard Analisis • Streamlit + Python")
