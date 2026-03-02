import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

st.set_page_config(layout="wide")

# =====================================================
# HEADER
# =====================================================
st.title("📊 Mini Dashboard Analisis Data Siswa")

# =====================================================
# LOAD DATA
# =====================================================
file = "data_simulasi_50_siswa_20_soal.xlsx"
df = pd.read_excel(file)

soal_cols = [c for c in df.columns if "Soal" in c]

# =====================================================
# DATA OTOMATIS
# =====================================================
df["Total_Nilai"] = df[soal_cols].sum(axis=1)
df["Rata_siswa"] = df[soal_cols].mean(axis=1)
rata_soal = df[soal_cols].mean()

# =====================================================
# TAB MENU (SEPERTI GAMBAR)
# =====================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Statistik",
    "🔥 Korelasi",
    "📉 Regresi",
    "📊 Visualisasi",
    "🎓 Analisis Penelitian"
])

# =====================================================
# TAB 1 — STATISTIK
# =====================================================
with tab1:

    st.header("Statistik Deskriptif")

    stat = df[soal_cols].agg(
        ["mean","median","std","min","max"]
    ).T

    stat["modus"] = df[soal_cols].mode().iloc[0]

    st.dataframe(stat)

    st.subheader("Tabel Nilai Siswa")
    st.dataframe(df[["Total_Nilai","Rata_siswa"]])

# =====================================================
# TAB 2 — KORELASI
# =====================================================
with tab2:

    st.header("Heatmap Korelasi")

    corr = df[soal_cols].corr()

    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(corr, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# =====================================================
# TAB 3 — REGRESI
# =====================================================
with tab3:

    st.header("Regresi Linear")

    X = df[["Rata_siswa"]]
    y = df["Total_Nilai"]

    model = LinearRegression().fit(X,y)
    pred = model.predict(X)

    fig, ax = plt.subplots()
    ax.scatter(X,y)
    ax.plot(X,pred)
    ax.set_xlabel("Rata-rata")
    ax.set_ylabel("Total Nilai")

    st.pyplot(fig)

    st.write("Koefisien:", model.coef_[0])
    st.write("R²:", model.score(X,y))

# =====================================================
# TAB 4 — VISUALISASI
# =====================================================
with tab4:

    col1, col2 = st.columns(2)

    # Bar chart
    with col1:
        st.subheader("Bar Chart Rata-rata Soal")
        fig1, ax1 = plt.subplots()
        rata_soal.plot(kind="bar", ax=ax1)
        st.pyplot(fig1)

    # Line chart
    with col2:
        st.subheader("Line Chart Skor Siswa")
        fig2, ax2 = plt.subplots()
        ax2.plot(df["Total_Nilai"])
        st.pyplot(fig2)

    st.subheader("Histogram Distribusi")
    fig3, ax3 = plt.subplots()
    ax3.hist(df["Total_Nilai"], bins=10)
    st.pyplot(fig3)

    st.subheader("Diagram Lingkaran")

    kategori = pd.cut(
        df["Total_Nilai"],
        bins=3,
        labels=["Rendah","Sedang","Tinggi"]
    )

    fig4, ax4 = plt.subplots()
    kategori.value_counts().plot.pie(autopct="%1.1f%%", ax=ax4)
    st.pyplot(fig4)

# =====================================================
# TAB 5 — ANALISIS PENELITIAN
# =====================================================
with tab5:

    st.subheader("Validitas Butir")

    total = df["Total_Nilai"]
    validitas = []

    for col in soal_cols:
        r,_ = pearsonr(df[col], total)
        validitas.append(r)

    valid_df = pd.DataFrame({
        "Soal":soal_cols,
        "r_hitung":validitas
    })

    st.dataframe(valid_df)

    st.subheader("Reliabilitas (Cronbach Alpha)")

    k = len(soal_cols)
    var_item = df[soal_cols].var(axis=0, ddof=1)
    var_total = df[soal_cols].sum(axis=1).var(ddof=1)

    alpha = (k/(k-1))*(1-(var_item.sum()/var_total))

    st.success(f"Cronbach Alpha = {alpha:.3f}")

    st.subheader("Indeks Kesukaran")

    indeks = df[soal_cols].mean()
    st.dataframe(indeks)

    st.subheader("Daya Pembeda")

    df_sorted = df.sort_values("Total_Nilai")
    n = int(len(df)*0.27)

    bawah = df_sorted.head(n)
    atas = df_sorted.tail(n)

    dp = atas[soal_cols].mean() - bawah[soal_cols].mean()
    st.dataframe(dp)
