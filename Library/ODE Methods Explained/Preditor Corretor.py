# Adams Moulton method (Preditor Corretor):

# Método explícito de ordem 4:
# Yi+1(0) = Yi + (h/24) * [(55*f(Xi, Yi) - 59*f(Xi-1, Yi-1) + 37*f(Xi-2, Yi-2) - 9*f(Xi-3, Yi-3))] i >= 3 

# Método implícito de ordem 4:
# Yi+1(1) = Yi + (h/24) * [9*f(Xi+1, Yi+1(0)) + 19*f(Xi, Yi) - 5*f(Xi-1, Yi-1) + f(Xi-2, Yi-2)]

# Critério de parada do método Implícito:
# (|Yi+1(k) - Yi+1(k-1)|) / (|Yi+1(1)|) < e   k = 1,2, 3, ...

# Initial Value Problem (IVP)
# Differential equation in differential form:
# Y' = [1 / (Y**2)] + x * ln(Y)
#
# Initial condition:
# Y(1) = 0,8

# Aproxime Y(2) usando h = 0,2
# Sabe-se que, pelo MRK4:
# X0 = 1 , Y0 = 0,8 => f(X0, Y0) = 1,3394
# X1 = 1,2 , Y1 = 1,0227 => f(X1, Y1) = 0,9830
# X2 = 1,4 , Y2 = 1,2131 => f(X2, Y2) = 0,9500
# X3 = 1,6 , Y3 = 1,4116 => f(X3, Y3) = 1,0534

# Adote e = 0,5 * 10 **(-2) e erro relativo como critério de parada

# 1 passo: Calcular Y(0)4 usando método explícito
# Y4(0) = Y3 + (h/24) * (55*f(X3, Y3) - 59*f(X2, Y2) + 37*f(X1, Y1) - 9*f(X0, Y0))

import math

def f(x, y):
    return 1 / (y**2) + x * math.log(y, math.e)

def erro_relativo(Yi_k, Yi_k_menos_1):
    return abs(Yi_k - Yi_k_menos_1) / abs(Yi_k)


Y4_0 = 1.4116 + (0.2/24) * (55*1.0534 - 59*0.9500 + 37*0.9830 - 9*1.3394)
print("Y4(0) =", Y4_0)

f_X4_Y4_0 = f(1.8, Y4_0)
print("f(X4, Y4(0)) =", f_X4_Y4_0)
print()


# 2 passo: Calcular Y(0)4 usando método implícito

# 1 iteração: k = 1
# Y4(1) = Y3 + (h/24) * (9*f(X4, Y4(0)) + 19*f(X3, Y3) - 5*f(X2, Y2) + f(X1, Y1))
Y4_1 = 1.4116 + (0.2/24) * (9*f_X4_Y4_0 + 19*1.0534 - 5*0.9500 + 0.9830)
print("Y4(1) =", Y4_1)
erro_relativo_k1 = erro_relativo(Y4_1, Y4_0)
print("Erro relativo k = 1:", erro_relativo_k1)
# continua pois erro_relativo_k1 > 0.5 * 10 **(-2)
print()

f_X4_Y4_1 = f(1.8, Y4_1)
print("f(X4, Y4(1)) =", f_X4_Y4_1)
print()

# 2 iteração: k = 2
# Y4(2) = Y3 + (h/24) * (9*f(X4, Y4(1)) + 19*f(X3, Y3) - 5*f(X2, Y2) + f(X1, Y1))
Y4_2 = 1.4116 + (0.2/24) * (9*f_X4_Y4_1 + 19*1.0534 - 5*0.9500 + 0.9830)
print("Y4(2) =", Y4_2)
erro_relativo_k2 = erro_relativo(Y4_2, Y4_1)
print("Erro relativo k = 2:", erro_relativo_k2)
# para k = 2, o erro relativo é menor que 0.5 * 10 **(-2), então o método implícito pode ser interrompido

f_X4_Y4_2 = f(1.8, Y4_2)
print("f(X4, Y4(2)) =", f_X4_Y4_2)
print()

# 3 passo: Calcular Y5(0) usando método explícito

Y5_0 = Y4_2 + (0.2/24) * (55*f_X4_Y4_2 - 59*f(1.6, 1.4116) + 37*f(1.4, 1.2131) - 9*f(1.2, 1.0227))
print("Y5(0) =", Y5_0)

f_X5_Y5_0 = f(2.0, Y5_0)
print("f(X5, Y5(0)) =", f_X5_Y5_0)
print()

# 1 iteração: calcular Y5(1) usando método implícito
Y5_1 = Y4_2 + (0.2/24) * (9*f_X5_Y5_0 + 19*f_X4_Y4_2 - 5*f(1.6, 1.4116) + f(1.4, 1.2131))
print("Y5(1) =", Y5_1)
erro_relativo_k1_Y5 = erro_relativo(Y5_1, Y5_0)
print("Erro relativo k = 1 para Y5:", erro_relativo_k1_Y5)
# para k = 1, o erro relativo é menor que 0.5 * 10 **(-2), então o método implícito pode ser interrompido
f_X5_Y5_1 = f(2.0, Y5_1)
print("f(X5, Y5(1)) =", f_X5_Y5_1)

# y5(1) é a aproximação de Y(2) usando o método de Adams Moulton (Preditor Corretor) com h = 0.2
print("Aproximação de Y(2) usando Adams Moulton (Preditor Corretor) =", Y5_1)