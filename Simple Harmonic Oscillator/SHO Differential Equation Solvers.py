import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from scipy.fft import fft, fftfreq

import pandas as pd
import matplotlib.animation as animation

def ler_float(mensagem):
    while True:
        entrada = input(mensagem)
        try:
            return float(entrada)
        except ValueError:
            print("Entrada inválida. Digite apenas números.")

def ler_positivo(mensagem):
    while True:
        valor = ler_float(mensagem)
        if valor > 0:
            return valor
        print("O valor deve ser maior que zero.")

def ler_amplitude(mensagem):
    while True:
        valor = ler_float(mensagem)
        if valor >= 0:
            return valor
        print("A amplitude deve ser maior ou igual a zero.")

def ler_x0_valido(mensagem, amplitude):
    while True:
        x0 = ler_float(mensagem)
        if x0 - amplitude > 0:
            return x0
        print("Erro: a massa atravessaria a parede. Digite um x0 maior ou uma amplitude menor.")

m = ler_positivo("Digite a massa (kg): ")
k = ler_positivo("Digite a constante da mola (N/m): ")
phi_0 = ler_float("Digite a fase phi: ")
Amplitude = ler_amplitude("Digite a amplitude: ")
x0 = ler_x0_valido("Digite a posição de equilíbrio x0: ", Amplitude)


omega = math.sqrt(k / m)

# Fórmula do Oscilador Harmônico Simples
# X(t)= x0 + A * np.cos(omega * t + phi_0)

def x_t(t):
    return x0 + Amplitude * np.cos(omega * t + phi_0)

# Gerar um intervalo de tempo para plotar o gráfico
t = np.linspace(0, 50, 1000)
x = x_t(t)  

# Plotar o gráfico
plt.plot(t, x)
plt.xlabel('Tempo (s)')
plt.ylabel('Posição (m)')
plt.title('Oscilador Harmônico Simples')
plt.grid()
plt.show()


# ---------------------------------------------------------------------------------------------------------------------


# =========================
# PARÂMETROS VISUAIS
# =========================
massa_largura = 2.2
massa_altura = 0.8
parede_x = 0
chao_y = -0.6
massa_y = -massa_altura / 2

# =========================
# FUNÇÃO PARA DESENHAR MOLA
# =========================
def desenhar_mola(x_inicio, x_fim, y, n_espiras=12, amplitude=0.15):
    if x_fim <= x_inicio:
        x_fim = x_inicio + 0.01

    xs = np.linspace(x_inicio, x_fim, n_espiras * 2 + 2)
    ys = np.zeros_like(xs) + y

    for i in range(1, len(xs) - 1):
        ys[i] = y + amplitude if i % 2 == 1 else y - amplitude

    ys[0] = y
    ys[-1] = y
    return xs, ys

# =========================
# FIGURA
# =========================
fig, ax = plt.subplots()
ax.set_title("Simulação do Sistema Massa-Mola")

xmax = x0 + Amplitude + 3
ax.set_xlim(-0.5, xmax)
ax.set_ylim(-1.2, 1.2)
ax.set_aspect('equal')
ax.axis('off')

# parede
parede = Rectangle((parede_x - 0.2, -0.8), 0.2, 1.6, color='black')
ax.add_patch(parede)

# chão
ax.plot([0, xmax], [chao_y, chao_y], linewidth=4)

# massa
massa = Rectangle((x[0], massa_y), massa_largura, massa_altura, fill=False, linewidth=2)
ax.add_patch(massa)

# texto da massa
texto_m = ax.text(x[0] + massa_largura/2, 0, "m", ha='center', va='center', fontsize=16)

# mola
linha_mola, = ax.plot([], [], linewidth=2)

# texto k
texto_k = ax.text(0, 0.35, "k", fontsize=16)

# =========================
# INIT
# =========================
def init():
    massa.set_xy((x[0], massa_y))
    texto_m.set_position((x[0] + massa_largura/2, 0))

    x_mola_inicio = 0
    x_mola_fim = x[0]
    xs, ys = desenhar_mola(x_mola_inicio, x_mola_fim, 0)
    linha_mola.set_data(xs, ys)

    texto_k.set_position(((x_mola_inicio + x_mola_fim)/2, 0.35))
    return massa, texto_m, linha_mola, texto_k

# =========================
# UPDATE
# =========================
def atualizar(frame):
    pos = x[frame]

    # atualiza massa
    massa.set_xy((pos, massa_y))
    texto_m.set_position((pos + massa_largura/2, 0))

    # atualiza mola até a face esquerda da massa
    x_mola_inicio = 0
    x_mola_fim = pos
    xs, ys = desenhar_mola(x_mola_inicio, x_mola_fim, 0)
    linha_mola.set_data(xs, ys)

    # atualiza texto k
    texto_k.set_position(((x_mola_inicio + x_mola_fim)/2, 0.35))

    return massa, texto_m, linha_mola, texto_k

ani = animation.FuncAnimation(
    fig,
    atualizar,
    frames=len(t),
    init_func=init,
    interval=30,
    blit=True
)

plt.show()