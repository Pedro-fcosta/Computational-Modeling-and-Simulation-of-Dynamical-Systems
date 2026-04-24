# Adams Moulton method (Preditor Corretor):

# Initial Value Problem (IVP)
# Differential equation in differential form:
# Y' = [1 / (Y**2)] + x * ln(Y)
#
# Initial condition:
# Y(1) = 0,8
#
# Aproxime Y(2) usando h = 0,2
# Sabe-se que, pelo MRK4:
# X0 = 1 , Y0 = 0,8 => f(X0, Y0) = 1,3394
# X1 = 1,2 , Y1 = 1,0227 => f(X1, Y1) = 0,9830
# X2 = 1,4 , Y2 = 1,2131 => f(X2, Y2) = 0,9500
# X3 = 1,6 , Y3 = 1,4116 => f(X3, Y3) = 1,0534
#
# Adote e = 0,5 * 10 **(-2) e erro relativo como critério de parada
#
# 1 passo: Calcular Y(0)4 usando método explícito
# Y4(0) = Y3 + (h/24) * (55*f(X3, Y3) - 59*f(X2, Y2) + 37*f(X1, Y1) - 9*f(X0, Y0))

import math

def f(x, y):
    return 1 / (y**2) + x * math.log(y)


Y4_0 = 1.4116 + (0.2/24) * (55*1.0534 - 59*0.9500 + 37*0.9830 - 9*1.3394)
print("Y4(0) =", Y4_0)

f_X4_Y4_0 = f(1.8, Y4_0)
print("f(X4, Y4(0)) =", f_X4_Y4_0)

# 2 passo: Calcular Y(0)4 usando método implícito
