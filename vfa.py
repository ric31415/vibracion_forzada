import streamlit as st
import sympy as sp
import numpy as np
import plotly.express as px
import pandas as pd

st.set_page_config(
    page_title="App Web",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Vibracion forzada con carga armonica")

# Datos Generales

st.sidebar.title("INGRESE DATOS")
g = st.sidebar.number_input("Gravedad (g) en cm/s2", min_value=0.0, max_value=2000.0, value=981.0, step=0.1)
t1 = st.sidebar.number_input("Tiempo del sismo en segundos", min_value=0.0, max_value=100.0, value=20.0, step=0.1)
t2 = st.sidebar.number_input("Tiempo a evaluar en segundos", min_value=0.0, max_value=100.0, value=25.0, step=0.1)
t3 = st.sidebar.number_input("Puntos a evaluar", min_value=0, max_value=10000, value=1000, step=1)
# Datos del Material
def_max = st.sidebar.number_input("Deformación Máxima (def_max) en cm", min_value=0.0, max_value=10.0, value=0.65, step=0.01)
fc = st.sidebar.number_input("Resistencia a Compresión (fc) en kgf/cm2", min_value=0.0, max_value=500.0, value=210.0, step=1.0)

# Cálculo de E basado en fc
E = 15100 * sp.sqrt(fc)

# Datos de la Estructura
W = st.sidebar.number_input("Peso de la Estructura (W) en kgf", min_value=0.0, max_value=100000.0, value=60000.0, step=100.0)
altura = st.sidebar.number_input("Altura de la Estructura (altura) en cm", min_value=0.0, max_value=1000.0, value=400.0, step=1.0)
xhi = st.sidebar.number_input("Factor de Amortiguamiento (xhi)", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
P0 = st.sidebar.number_input("Carga Aplicada (P0) en kgf", min_value=0.0, max_value=50000.0, value=10000.0, step=100.0)
u0 = st.sidebar.number_input("Desplazamiento Inicial (u0)", min_value=-10.0, max_value=10.0, value=0.0, step=0.1)
v0 = st.sidebar.number_input("Velocidad Inicial (v0)", min_value=-10.0, max_value=10.0, value=0.0, step=0.1)

# Datos de la Sección
b = st.sidebar.number_input("Ancho de la Sección (b) en cm", min_value=0.0, max_value=100.0, value=42.0, step=0.1)
h = st.sidebar.number_input("Altura de la Sección (h) en cm", min_value=0.0, max_value=100.0, value=42.0, step=0.1)


xhi = sp.nsimplify(0.2)
I = 1/12*b*h**3 # cm3


# Calculo de la rigidez
k = 12*E*I/altura**3
k = 2*k

# Calculo de la frecuencias
wn = round(sp.sqrt(k*g/W), 2)
wd = round(wn*sp.sqrt(1 - xhi**2), 2)
w = round(0.5*wn, 2)



# Ecuacion de respuesta en la Fase 1

A, B, t = sp.symbols('A B t')
c1 = P0/k
c2 = w/wn
C = c1 * (1 - c2**2) / ((1 - c2**2)**2 + (2*xhi*c2)**2)
D = c1 * (-2*xhi*c2) / ((1 - c2**2)**2 + (2*xhi*c2)**2)

u = sp.exp(-xhi*wn*t) * (A*sp.cos(wd*t) + B*sp.sin(wd*t)) + C*sp.sin(w*t) + D*sp.cos(w*t)
v = sp.diff(u, t)

A, B = sp.nsolve([u.subs(t, 0) - u0, v.subs(t, 0) - v0], [A, B], [0.5, 0.5])

u1 = sp.exp(-xhi*wn*t) * (A*sp.cos(wd*t) + B*sp.sin(wd*t)) + C*sp.sin(w*t) + D*sp.cos(w*t)
v1 = sp.diff(u1, t)

# Ecuacion de respuesta en la Fase 2
u0 = u1.subs(t, t1)
v0 = v1.subs(t, t1)

u2 = sp.exp(-xhi*wn*t) * (u0*sp.cos(wd*t) + ((v0 + xhi*wn*u0) / wd)*sp.sin(wd*t))
v2 = sp.diff(u2, t)

u22 = sp.exp(-xhi*wn*(t-t1)) * (u0*sp.cos(wd*(t-t1)) + ((v0 + xhi*wn*u0) / wd)*sp.sin(wd*(t-t1)))
v22 = sp.diff(u22, t)

# Ecuacion de respuesta total

u = sp.Piecewise((u1, t <= t1), (u22, t > t1))
v = sp.Piecewise((v1, t <= t1), (v22, t > t1))



# Mostrar entradas
st.write(f""" Calculos Previos

I = {I:.2f} cm4

k = {k:.2f} kgf/cm2

ωn = {wn:.2f} rad/seg

ωd = {wd:.2f} rad/seg

ω = {w:.2f} rad/seg

A = {A:.2f} cm

B = {B:.2f} cm

C = {C:.2f} cm

D = {D:.2f} cm
""")




# Ecuaciones
st.write("Ecuacion de desplazamiento en la fase 1")
e1 = "u = " + sp.latex(u1.evalf(2))
st.latex(e1)
st.write("Ecuacion de velocidad en la fase 1")
e2 = "\dot{u} = " + sp.latex(sp.factor(v1.evalf(2)))
st.latex(e2)
st.write("Ecuacion de desplazamiento en la fase 2")
e3 = "u = " + sp.latex(u2.evalf(2))
st.latex(e3)
st.write("Ecuacion de velocidad en la fase 2")
e4 = "\dot{u} = " + sp.latex(sp.factor(v2.evalf(2)))
st.latex(e4)

# Gráficas
x = np.linspace(0, t2, t3, endpoint=True)
f1 = sp.lambdify(t, u, "numpy")
f2 = sp.lambdify(t, v, "numpy")

y1 = f1(x)
y2 = f2(x)

dic = {"Tiempo": x, "Desplazamiento": y1, "Velocidad": y2}
df = pd.DataFrame(dic)

fig1 = px.line(df, x="Tiempo", y="Desplazamiento", title="Desplazamiento")
fig2 = px.line(df, x="Tiempo", y="Velocidad", title="Velocidad")

st.write("Las graficas de todo el movimiento son:")

st.plotly_chart(fig1)
st.plotly_chart(fig2)

# Evaluar funciones u y v en un valor específico de tiempo t
t_input = st.number_input("Ingresa un valor de tiempo t para evaluar u y v:", value=0.0, step=0.1)
u_eval = u.evalf(subs={t: t_input}, n=2)
v_eval = v.evalf(subs={t: t_input}, n=2)

st.write(f"""Para t = {t_input:.1f}

Desplazamiento u = {u_eval:.2f}

Velocidad v = {v_eval:.2f}""")
