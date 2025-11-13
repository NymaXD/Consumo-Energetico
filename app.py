import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.integrate as spi

# --- Título ---
st.title("🎛️ Dashboard interactivo de energía utilizada en centros de datos")

# --- Sliders para parámetros ---
N = st.sidebar.slider("Número de servidores (N)", 10, 10000, 2000, 10)
P_idle = st.sidebar.slider("Potencia en reposo del servidor (Pidle) [W]", 10, 1000, 100, 10)
P_max = st.sidebar.slider("Potencia máxima del servidor (Pmax) [W]", 100, 10000, 400, 10)
COP_0 = st.sidebar.slider("Coeficiente de rendimiento (COP₀)", 1.0, 5.0, 3.0, 0.1)
Prc = st.sidebar.slider("Potencia de refrigeración (Prc) [W]", 1000, 1000000, 300000, 1000)
lambda_0 = st.sidebar.slider("Tasa media de peticiones por segundo (λ₀)", 1000, 100000, 10000, 1000)
E_S = st.sidebar.slider("Tiempo medio de servicio [s]", 0.001, 1.0, 0.1, 0.001)
A = st.sidebar.slider("Variación de las peticiones (A) - factor 0-1", 0.0, 1.0, 0.2, 0.01)
T = st.sidebar.slider("Horizonte de tiempo (horas)", 1, 24, 24, 1)

# --- Constantes ---
k = 1 / COP_0

# --- Funciones ---
def lambda_t(t):
    """Tasa de llegadas (req/s) con variación diaria senoidal."""
    # A representa la amplitud de variación, aseguramos que lambda no sea negativa
    return np.maximum(0, lambda_0 + A * lambda_0 * np.sin(2 * np.pi * t / 24))

def u_t(t):
    """Utilización promedio por servidor (0–1)."""
    u = (lambda_t(t) * E_S) / N
    return np.clip(u, 0.0, 1.0)  # recorte entre 0 y 1

def P_t(t):
    """Potencia total instantánea (W) según el modelo correcto."""
    u = u_t(t)
    return N * (P_idle + (P_max - P_idle) * u) * (1 + k) + Prc

def integral_u_t(t):
    """Integral de u(τ) desde 0 hasta t - solución analítica."""
    base = (lambda_0 * E_S) / N
    # ∫₀ᵗ u(τ)dτ = base · [t + A·(12/π)·(1 - cos(2πt/24))]
    integral = base * (t + A * (12/np.pi) * (1 - np.cos(2*np.pi*t/24)))
    return np.maximum(0, integral)

def E_analitica(t):
    """Energía total usando la fórmula exacta del modelo (Wh)."""
    return N * (1 + k) * (P_idle * t + (P_max - P_idle) * integral_u_t(t)) + Prc * t

# --- Cálculo de energía total ---
# Método 1: Integración numérica (para verificar)
E_total_Wh, _ = spi.quad(P_t, 0, T)
E_total_kWh = E_total_Wh / 1000

# Método 2: Fórmula analítica exacta (más precisa)
E_total_analitica_Wh = E_analitica(T)
E_total_analitica_kWh = E_total_analitica_Wh / 1000

# --- Generar arrays para gráficas ---
t_points = np.linspace(0, T, 500)

# Vectorizar las funciones para mayor eficiencia
lambda_points = np.maximum(0, lambda_0 + A * lambda_0 * np.sin(2 * np.pi * t_points / 24))
u_points = np.clip((lambda_points * E_S) / N, 0.0, 1.0)
P_points = N * (P_idle + (P_max - P_idle) * u_points) * (1 + k) + Prc

# Calcular energía acumulada usando la fórmula analítica
E_acum_points = np.array([E_analitica(t) / 1000 for t in t_points])  # en kWh

# --- DataFrame de resumen ---
df = pd.DataFrame({
    "Variable": [
        "Número de servidores (N)",
        "Potencia en reposo (Pidle W)",
        "Potencia máxima (Pmax W)",
        "Coeficiente de rendimiento (COP0)",
        "Potencia de refrigeración (Prc W)",
        "Tasa media de peticiones por segundo (lambda_0)",
        "Tiempo medio de servicio (E_S) [s]",
        "Variación de las peticiones (A)",
        "Horizonte de tiempo (T)",
        "Energía total numérica (kWh)"
    ],
    "Valor": [
        N, P_idle, P_max, COP_0, Prc, lambda_0, E_S, A, T,
        round(E_total_kWh, 2)
    ],
    "Unidad": [
        "unidades", "W", "W", "-", "W", "req/s", "s", "-", "h", "kWh"
    ]
})

st.subheader("📊 Resumen de parámetros")
st.dataframe(df, use_container_width=True)

# --- Gráficas ---
st.subheader("📈 Visualizaciones")

# Gráfica 1: Energía total acumulada vs Tiempo
fig1, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(t_points, E_acum_points, 'b-', linewidth=2)
ax1.set_xlabel("Tiempo (h)", fontsize=12)
ax1.set_ylabel("Energía acumulada (kWh)", fontsize=12)
ax1.set_title("Energía Total Acumulada vs Tiempo", fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
st.pyplot(fig1)

# Gráfica 2: Energía acumulada vs Utilización (con dirección temporal)
fig2, ax2 = plt.subplots(figsize=(10, 5))

# Colorear por tiempo para mostrar la dirección
scatter = ax2.scatter(u_points, E_acum_points, c=t_points, cmap='viridis', 
                      alpha=0.6, s=30, edgecolors='black', linewidths=0.5)

ax2.set_xlabel("Utilización u(t)", fontsize=12)
ax2.set_ylabel("Energía acumulada (kWh)", fontsize=12)
ax2.set_title("Energía Total vs Utilización", fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax2)
cbar.set_label('Tiempo (h)', fontsize=10)
st.pyplot(fig2)

# Gráfica 3: Utilización vs Tasa de peticiones
fig3, ax3 = plt.subplots(figsize=(10, 5))
# Generar puntos de lambda desde 0 hasta lambda_0
lambda_range = np.linspace(0, lambda_0, 500)
u_range = np.clip((lambda_range * E_S) / N, 0.0, 1.0)
ax3.plot(lambda_range, u_range, 'g-', linewidth=2)
ax3.set_xlabel("Tasa de peticiones λ(t) (req/s)", fontsize=12)
ax3.set_ylabel("Utilización u(t)", fontsize=12)
ax3.set_title("Utilización vs Tasa de Peticiones", fontsize=14, fontweight='bold')
ax3.set_xlim(0, lambda_0)
ax3.grid(True, alpha=0.3)
st.pyplot(fig3)


# Gráfica 4: Potencia instantánea vs Tiempo
fig4, ax4 = plt.subplots(figsize=(10, 5))
ax4.plot(t_points, P_points/1000, 'r-', linewidth=2)
ax4.set_xlabel("Tiempo (h)", fontsize=12)
ax4.set_ylabel("Potencia instantánea (kW)", fontsize=12)
ax4.set_title("Potencia Instantánea vs Tiempo", fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.fill_between(t_points, P_points/1000, alpha=0.3, color='red')
st.pyplot(fig4)

# Gráfica 5: Potencia vs tiempo
fig5, ax5c = plt.subplots(figsize=(10, 5))

ax5c.plot(t_points, P_points/1000, 'teal', linewidth=2)
ax5c.set_xlabel("Tiempo (h)", fontsize=10)
ax5c.set_ylabel("Potencia (kW)", fontsize=10)
ax5c.grid(True, alpha=0.3)

plt.tight_layout()
st.pyplot(fig5)