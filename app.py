import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.fft import fft, fftfreq
import kagglehub
import os

# Descargar el dataset desde KaggleHub
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
csv_path = os.path.join(path, "creditcard.csv")

# Cargar CSV
df = pd.read_csv(csv_path, encoding="latin1")


# Transformar columna Time a formato datetime
df["FechaHora"] = pd.to_datetime(df["Time"], unit="s", origin="2023-01-01")

# Verificar valores nulos
print(df.isnull().sum())

df['Class'] = df['Class'].map({0: 'No Fraude', 1: 'Fraude'})

print(df['Class'].value_counts())

# Estadísticas básicas del monto
print(df['Amount'].describe())

# Tiempo en horas desde la primera transacción
df['Hora'] = (df['Time'] // 3600).astype(int)

#numero de transacciones hechas por hora 
df['Hora'].value_counts().sort_index()


# Configuración general
st.set_page_config(page_title="Dashboard de Fraude Bancario", layout="wide")
st.title(" Dashboard Transacciones")


#  Aquí haces la transformación
df["FechaHora"] = pd.to_datetime(df["Time"], unit="s", origin="2023-01-01")


# ========================
#  Filtros en Sidebar
# ========================
st.sidebar.header("Filtros")

clase = st.sidebar.radio("Tipo de transacción", ["Todas", "Fraude", "No Fraude"])
if clase != "Todas":
    df = df[df["Class"] == clase]

monto_max = int(df["Amount"].max())
rango_monto = st.sidebar.slider("Rango de monto (€)", 0, monto_max, (0, 2000))
df = df[(df["Amount"] >= rango_monto[0]) & (df["Amount"] <= rango_monto[1])]

rango_hora = st.sidebar.slider("Hora del día", 0, 23, (0, 23))
df = df[(df["Hora"] >= rango_hora[0]) & (df["Hora"] <= rango_hora[1])]

# ========================
#  Métricas principales
# ========================
col1, col2, col3, col4 = st.columns(4)
col1.metric("Transacciones", f"{len(df):,}")
col2.metric("Fraudes", f"{(df['Class'] == 'Fraude').sum():,}")
col3.metric("Porcentaje Fraude", f"{(df['Class'] == 'Fraude').mean()*100:.2f}%")
col4.metric("Monto Total (€)", f"{df['Amount'].sum():,.2f}")

# ========================
#  Gráficos: pastel y monto
# ========================
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Distribución de Fraude")
    fig1, ax1 = plt.subplots()
    df["Class"].value_counts().plot.pie(
        autopct="%1.1f%%", startangle=90, ax=ax1, colors=["#5cb85c", "#d9534f"]
    )
    ax1.axis("equal")
    ax1.set_ylabel("")
    st.pyplot(fig1)

with col_b:
    st.subheader("Distribución de Montos (€)")
    fig2, ax2 = plt.subplots()
    df[df["Amount"] < 2000]["Amount"].hist(bins=40, ax=ax2, color="#0275d8")
    ax2.set_xlabel("Monto")
    ax2.set_ylabel("Frecuencia")
    st.pyplot(fig2)

# ========================
#  Frecuencia de Fraudes por Hora (corregido)
# ========================
st.subheader(" Frecuencia de Fraudes por Hora")

if "Fraude" in df["Class"].values:
    fraudes_hora = df[df["Class"] == "Fraude"]["Hora"].value_counts().sort_index()
    fig3, ax3 = plt.subplots()
    fraudes_hora.plot(kind="bar", color="#f0ad4e", ax=ax3)
    ax3.set_xlabel("Hora del Día")
    ax3.set_ylabel("Cantidad de Fraudes")
    st.pyplot(fig3)
else:
    st.info("No hay fraudes en el conjunto filtrado. Ajusta los filtros para visualizar esta gráfica.")

# ========================
#  Tabla de top fraudes
# ========================
st.subheader(" Top 10 Fraudes por Monto")
fraudes_top = df[df["Class"] == "Fraude"].sort_values("Amount", ascending=False).head(10)
st.dataframe(fraudes_top)

# ========================
#  Mapa de calor de correlación PCA
# ========================
st.subheader("Mapa de Calor - Variables PCA vs Fraude")

df_corr = df.copy()
df_corr["Class"] = df_corr["Class"].map({"No Fraude": 0, "Fraude": 1})
pca_cols = [f"V{i}" for i in range(1, 29)] + ["Class"]
corr = df_corr[pca_cols].corr()

fig4, ax4 = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax4)
st.pyplot(fig4)


st.subheader(" Evolución de Transacciones por Hora")

series = df.set_index("FechaHora").resample("H")["Amount"].count()

fig_ts, ax_ts = plt.subplots()
series.plot(ax=ax_ts, color="#0275d8")
ax_ts.set_title("Cantidad de transacciones por hora")
ax_ts.set_ylabel("Número de transacciones")
st.pyplot(fig_ts)




# ========================
#  Análisis Espectral (FFT)
# ========================
st.subheader(" Análisis Espectral de la Actividad Transaccional")

# Serie temporal de número de transacciones por hora
serie_ts = df.set_index("FechaHora").resample("H")["Amount"].count().dropna()
N = len(serie_ts)
T = 1.0  # intervalo entre puntos (1 hora)

# Aplicar FFT
yf = fft(serie_ts.values)
xf = fftfreq(N, T)[:N // 2]
espectro = 2.0 / N * np.abs(yf[:N // 2])

# Gráfico del espectro
fig_fft, ax_fft = plt.subplots()
ax_fft.plot(xf, espectro, color="#5bc0de")
ax_fft.set_title("Espectro de frecuencia (FFT) - Transacciones por hora")
ax_fft.set_xlabel("Frecuencia (ciclos por hora)")
ax_fft.set_ylabel("Magnitud")
st.pyplot(fig_fft)



