import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Regresión Lineal: Horas vs Puntaje",
    layout="wide"
)

# ==========================================
# 1. ENUNCIADO Y CONSIGNAS
# ==========================================
st.title("4.1 Ejercicio 5. Horas de estudio y puntaje de evaluación: modelo lineal")

st.info("""
**Problema:**  
Se desea describir la relación entre la cantidad de horas de estudio de un grupo de estudiantes y su puntaje en una evaluación. Considere las horas como variable explicativa $x$ y el puntaje como respuesta $y$. Ajuste una recta mediante mínimos cuadrados y evalúe su utilidad descriptiva.

**Consignas:**
1. Represente los pares $(x, y)$ en un gráfico de dispersión. Describa la dirección y la forma aparente de la asociación.
2. Obtenga por mínimos cuadrados $\\hat{y} = ax + b$, donde $a$ es la pendiente y $b$ la ordenada al origen. Muestre cómo se obtienen o cómo se calcula cada parámetro.
3. Calcule $R^2$, obtenga los valores ajustados $\\hat{y}_i$ y los residuos $e_i = y_i - \\hat{y}_i$. Incluya la suma de cuadrados de los residuos o el error estándar de la estimación.
4. Interprete la pendiente en el contexto del ejercicio. Comente si el intercepto tiene interpretación práctica y evite extrapolar fuera del rango observado sin justificación.
""")

# ==========================================
# 2. CONTROLES INTERACTIVOS (SIDEBAR)
# ==========================================
st.sidebar.header("Panel de Datos Interactivos")
st.sidebar.markdown("Modifica las observaciones o agrega/elimina filas en la tabla para recalcular todo el modelo:")

datos_iniciales = pd.DataFrame({
    "Horas (x)": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    "Puntaje (y)": [52.0, 55.0, 61.0, 65.0, 68.0, 72.0, 75.0, 81.0]
})

df_editado = st.sidebar.data_editor(
    datos_iniciales,
    num_rows="dynamic",
    use_container_width=True
)

# Limpiar posibles filas vacías
df_clean = df_editado.dropna()

if len(df_clean) < 3:
    st.error("Se necesitan al menos 3 pares de datos válidos $(x, y)$ para ajustar el modelo y calcular el error estándar.")
    st.stop()

x = df_clean["Horas (x)"].to_numpy(dtype=float)
y = df_clean["Puntaje (y)"].to_numpy(dtype=float)
n = len(x)

if np.std(x) == 0:
    st.error("Los valores de Horas (x) no pueden ser todos iguales (varianza cero).")
    st.stop()

# ==========================================
# 3. CÁLCULOS DE MÍNIMOS CUADRADOS
# ==========================================
x_mean = np.mean(x)
y_mean = np.mean(y)

S_xx = np.sum((x - x_mean)**2)
S_xy = np.sum((x - x_mean) * (y - y_mean))
S_yy = np.sum((y - y_mean)**2)

# Pendiente (a) y Ordenada al origen (b)
a = S_xy / S_xx
b = y_mean - a * x_mean

# Valores ajustados, residuos y métricas de ajuste
y_hat = a * x + b
residuos = y - y_hat

SCR = np.sum(residuos**2)          # Suma de Cuadrados de los Residuos (SSE)
SCT = S_yy                         # Suma de Cuadrados Total (SST)
SCE = SCT - SCR                    # Suma de Cuadrados Explicada (SSR)
R2 = SCE / SCT if SCT != 0 else 0  # Coeficiente de determinación
s_e = np.sqrt(SCR / (n - 2))       # Error estándar de la estimación

# ==========================================
# 4. RESOLUCIÓN DE CONSIGNAS
# ==========================================
st.header("Datos Cargados")
st.dataframe(df_clean.T, use_container_width=True)

st.markdown("---")
st.subheader("1. Gráfico de Dispersión y Recta Ajustada")

fig, ax = plt.subplots(figsize=(8, 4.2))
ax.scatter(x, y, color="#2980b9", s=70, label="Datos observados $(x_i, y_i)$", zorder=3)

# Línea de regresión
x_linea = np.linspace(x.min(), x.max(), 100)
y_linea = a * x_linea + b
ax.plot(x_linea, y_linea, color="#e74c3c", lw=2, label=f"Recta ajustada: $\\hat{{y}} = {a:.4f}x + {b:.4f}$", zorder=2)

# Dibujar segmentos de residuos
for xi, yi, yhi in zip(x, y, y_hat):
    ax.plot([xi, xi], [yi, yhi], color="gray", linestyle=":", alpha=0.7, zorder=1)

ax.set_xlabel("Horas de estudio ($x$)", fontsize=11)
ax.set_ylabel("Puntaje obtenido ($y$)", fontsize=11)
ax.set_title("Dispersión de Puntajes vs. Horas de Estudio", fontsize=12)
ax.legend()
ax.grid(True, linestyle="--", alpha=0.5)
fig.tight_layout()
st.pyplot(fig)

direccion = "positiva (creciente)" if a > 0 else "negativa (decreciente)"
st.write(f"""
- **Descripción:** Los puntos muestran una asociación lineal **{direccion}** clara y sostenida: a medida que aumentan las horas de estudio ($x$), el puntaje en la evaluación ($y$) tiende a incrementarse de manera proporcional alrededor de una línea recta.
""")

st.markdown("---")
st.subheader("2. Cálculo de Parámetros por Mínimos Cuadrados ($\\hat{y} = ax + b$)")

col_form, col_res = st.columns([1.2, 1])

with col_form:
    st.latex(r"a = \frac{S_{xy}}{S_{xx}} = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2}")
    st.latex(r"b = \bar{y} - a\bar{x}")

with col_res:
    st.write(f"- **Promedios:** $\\bar{{x}} = {x_mean:.2f}$ horas | $\\bar{{y}} = {y_mean:.2f}$ puntos")
    st.write(f"- **Sumas de cuadrados y productos:** $S_{{xx}} = {S_xx:.2f}$ | $S_{{xy}} = {S_xy:.2f}$")
    st.write(f"- **Pendiente ($a$):** ${S_xy:.2f} / {S_xx:.2f} = \\mathbf{{{a:.4f}}}$")
    st.write(f"- **Ordenada al origen ($b$):** ${y_mean:.2f} - ({a:.4f} \\times {x_mean:.2f}) = \\mathbf{{{b:.4f}}}$")

st.success(f"**Ecuación de la recta ajustada:** $\\hat{{y}} = {a:.4f}x + {b:.4f}$")

st.markdown("---")
st.subheader("3. Bondad de Ajuste ($R^2$), Valores Ajustados y Residuos")

col1, col2, col3 = st.columns(3)
col1.metric("Coeficiente de Determinación (R²)", f"{R2:.4f} ({R2*100:.2f}%)")
col2.metric("Suma Cuadrados Residuos (SCR)", f"{SCR:.4f}")
col3.metric("Error Estándar Estimación (sₑ)", f"{s_e:.4f}")

df_resultados = pd.DataFrame({
    "Horas (x_i)": x,
    "Puntaje Obs (y_i)": y,
    "Puntaje Ajustado (ŷ_i)": np.round(y_hat, 4),
    "Residuo (e_i = y_i - ŷ_i)": np.round(residuos, 4),
    "Residuo² (e_i²)": np.round(residuos**2, 4)
})

st.dataframe(df_resultados, use_container_width=True)

st.markdown("---")
st.subheader("4. Interpretación en el Contexto del Ejercicio")

st.markdown(f"""
* **Interpretación de la pendiente ($a = {a:.4f}$):** Por cada **1 hora adicional** de estudio, se estima que el puntaje de la evaluación aumenta en promedio **{a:.2f} puntos**.
* **Interpretación del intercepto ($b = {b:.4f}$):** Representa el puntaje promedio estimado ({b:.2f} puntos) para un estudiante que dedica **0 horas** de estudio. Aunque tiene sentido matemático como punto de partida en el dominio de la evaluación, su interpretación práctica debe tomarse con cautela porque $x = 0$ está fuera del rango observado ($[{x.min():.0f}, {x.max():.0f}]$ horas).
* **Advertencia sobre extrapolación:** Este modelo lineal es válido únicamente dentro del rango observado de **${x.min():.0f}$ a ${x.max():.0f}$ horas** de estudio. Extrapolar a valores mucho mayores (por ejemplo, 20 horas) produciría puntajes irreales (superiores a 100 puntos), ya que la relación deja de comportarse de forma lineal fuera del intervalo estudiado.
""")
