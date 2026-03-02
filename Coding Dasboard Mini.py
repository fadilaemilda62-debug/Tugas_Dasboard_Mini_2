import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# =============================
# CONFIG
# =============================
st.set_page_config(page_title="Dashboard Analisis Siswa", layout="wide")

st.title("📊 Mini Dashboard Analisis Data Siswa")

# =============================
# LOAD DATA
# =============================
@st.cache_data
def load_data():
    df = pd.read_excel("data_simulasi_50_siswa_20_soal.xlsx")
    return df

df = load_data()

# Hilangkan kolom responden
data = df.drop(columns=["Responden"])

# =============================
# TOTAL SCORE
# =============================
data["Total_Skor"] = data.sum(axis=1)

# =============================
# SIDEBAR
# =============================
st.sidebar.header("Pengaturan Analisis")

target = st.sidebar.selectbox(
    "Pilih Variabel Target (Y)",
    data.columns
)

fitur = st.sidebar.multiselect(
    "Pilih Variabel Prediktor (X)",
    [col for col in data.columns if col != target],
    default=data.columns[:3]
)

# =============================
# TAB MENU
# =============================
tab1, tab2, tab3, tab4 = st.tabs([
    "Statistik",
    "Korelasi",
    "Regresi",
    "Visualisasi"
])

# =============================
# TAB 1 — STATISTIK
# =============================
with tab1:
    st.subheader("Statistik Deskriptif")

    st.dataframe(data.describe())

# =============================
# TAB 2 — KORELASI
# =============================
with tab2:
    st.subheader("Heatmap Korelasi")

    corr = data.corr()

    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax)

    st.pyplot(fig)

# =============================
# TAB 3 — REGRESI
# =============================
with tab3:

    if len(fitur) > 0:

        X = data[fitur]
        y = data[target]

        model = LinearRegression()
        model.fit(X, y)

        prediksi = model.predict(X)

        st.subheader("Hasil Regresi Linear")

        # Koefisien
        coef_df = pd.DataFrame({
            "Variabel": fitur,
            "Koefisien": model.coef_
        })

        st.write("### Koefisien Regresi")
        st.dataframe(coef_df)

        st.write("Intercept:", model.intercept_)

        # R2 Score
        r2 = model.score(X, y)
        st.write("### R² Score:", round(r2, 4))

    else:
        st.warning("Pilih minimal satu variabel prediktor.")

# =============================
# TAB 4 — VISUALISASI
# =============================
with tab4:

    st.subheader("Distribusi Skor Total")

    fig, ax = plt.subplots()
    ax.hist(data["Total_Skor"], bins=10)
    ax.set_xlabel("Total Skor")
    ax.set_ylabel("Frekuensi")

    st.pyplot(fig)

    st.subheader("Scatter Plot")

    x_axis = st.selectbox("Pilih sumbu X", data.columns)
    y_axis = st.selectbox("Pilih sumbu Y", data.columns, index=1)

    fig2, ax2 = plt.subplots()
    ax2.scatter(data[x_axis], data[y_axis])
    ax2.set_xlabel(x_axis)
    ax2.set_ylabel(y_axis)

    st.pyplot(fig2)
