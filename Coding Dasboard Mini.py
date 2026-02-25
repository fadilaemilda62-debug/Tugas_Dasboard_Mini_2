import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="Dashboard Analisis Siswa",
    layout="wide"
)

st.title("📊 Dashboard Analisis Hasil Siswa")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    df = pd.read_excel("data_simulasi_50_siswa_20_soal.xlsx")
    return df

df = load_data()

# =========================
# PREPROCESSING
# =========================
df["Total_Nilai"] = df.sum(axis=1)
df["Rata_Rata"] = df.mean(axis=1)

# =========================
# METRIK UTAMA
# =========================
col1, col2, col3 = st.columns(3)

col1.metric("Jumlah Siswa", len(df))
col2.metric("Rata-rata Nilai", round(df["Rata_Rata"].mean(), 2))
col3.metric("Nilai Maksimum", df["Total_Nilai"].max())

st.divider()

# =========================
# RATA-RATA PER SOAL
# =========================
st.subheader("📘 Rata-rata Nilai Tiap Soal")

mean_per_soal = df.iloc[:, :20].mean()

fig1, ax1 = plt.subplots()
mean_per_soal.plot(kind="bar", ax=ax1)
ax1.set_ylabel("Rata-rata Skor")
ax1.set_xlabel("Soal")

st.pyplot(fig1)

# =========================
# DISTRIBUSI NILAI SISWA
# =========================
st.subheader("📈 Distribusi Total Nilai Siswa")

fig2, ax2 = plt.subplots()
ax2.hist(df["Total_Nilai"], bins=10)
ax2.set_xlabel("Total Nilai")
ax2.set_ylabel("Jumlah Siswa")

st.pyplot(fig2)

# =========================
# NILAI PER SISWA
# =========================
st.subheader("👩‍🎓 Data Nilai Siswa")

st.dataframe(df)

# =========================
# FILTER INTERAKTIF
# =========================
st.subheader("🔎 Filter Nilai")

nilai_min = st.slider(
    "Pilih minimum total nilai",
    int(df["Total_Nilai"].min()),
    int(df["Total_Nilai"].max()),
    int(df["Total_Nilai"].min())
)

filtered = df[df["Total_Nilai"] >= nilai_min]

st.write(f"Jumlah siswa terfilter: {len(filtered)}")
st.dataframe(filtered)