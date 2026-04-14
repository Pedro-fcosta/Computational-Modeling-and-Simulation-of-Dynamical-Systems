import math
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from scipy.fft import fft, fftfreq

import pandas as pd
import matplotlib.animation as animation

# Estudo das bibliotecas utilizadas:
# math:
# Fornece funções matemáticas básicas, como trigonometria, exponenciais, logaritmos, etc.
#   import math
#   x = math.sin(1.2)
#
# numpy:
# Vetores e matrizes, arrays, operações rápidas, criação de malhas de tempo, etc.
#   import numpy as np
#   t = np.linspace(0, 10, 100)  # Cria um array de 100 pontos entre 0 e 10
#   x = np.cos(t)  # Aplica a função cosseno a cada elemento do array t
#   retorna x = array([1.0, 0.998, 0.992, ..., -0.999, -0.998, -0.984])
#
# matplotlib.pyplot:
# Para criar gráficos e visualizações.
#   import matplotlib.pyplot as plt
#   import numpy as np
#   t = np.linspace(0, 10, 100)
#   x = np.cos(t)
#   plt.plot(t, x)  # Plota x em função de t
#   plt.xlabel('Tempo')
#   plt.ylabel('x(t)')
#   plt.show()  # Exibe o gráfico
#   retorna um gráfico de uma onda cosseno ao longo do tempo
#
# scipy.integrate.solve_ivp:
# Para resolver equações diferenciais ordinárias (EDOs) usando métodos numéricos.
#   from scipy.integrate import solve_ivp
#   import numpy as np
#   import matplotlib.pyplot as plt
#
#   def sistema(t, y):
#       x, v = y
#       dxdt = v
#       dvdt = -x  # Exemplo de um sistema harmônico simples
#       return [dxdt, dvdt]
#       
#   sol = solve_ivp(sistema, [0, 10], [1, 0], t_eval=np.linspace(0, 10, 500))
#
#   plt.plot(sol.t, sol.y[0])  # Plota x(t)
#   plt.show()  # Exibe o gráfico
#
# scipy.optimize.fsolve:
# Para encontrar raízes de funções não lineares.
#   from scipy.optimize import fsolve
#   import numpy as np
#   def f(x):
#       return x**3 - x - 1  # Queremos encontrar
#
# raiz = fsolve(f, 1.0)  # Chute inicial de 1.0
# print(raiz)  # Imprime a raiz encontrada
#
# scipy.fft.fft e scipy.fft.fftfreq:
# Para descobrir frequências dominantes, estudar oscilações, analisar sinais, etc.
#   def sistema(t, y):
#       x, v = y
#       dxdt = v
#       dvdt = -x #equivalente a x'' = -x
#       return [dxdt, dvdt]
#
# intervalo de tempo e pontos onde quero a solução
# t = np.linspace(0, 20, 1000)
#
# resolve a eq com condição inicial x(0)=1, v(0)=0
# sol = solve_ivp(sistema, [0, 20], [1, 0], t_eval=t)
#
# gráfico da posição
# plt.plot(sol.t, sol.y[0])
# plt.xlabel('Tempo')
# plt.ylabel('posição x(t)')
#plt.title('Sistema massa-mola')
# plt.grid()
# plt.show()