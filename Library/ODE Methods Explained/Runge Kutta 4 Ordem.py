# Runge Kutta 4th Order Method:

# Initial Value Problem (IVP)
# Differential equation in differential form:
# (y**2 - x**2) dx + x*y dy = 0
#
# Initial condition:
# y(1) = 2
#
# Initial data:
# x0 = 1
# y0 = 2
#
# Step size:
# h = 0.1
#
# Objective:
# Approximate y(1.2) using a numerical method
#
# Grid points:
# x0 = 1.0
# x1 = 1.1
# x2 = 1.2
#
# Explicit form for numerical methods:
# dy/dx = (x / y) - (y / x)
#
# Function used in the code:
# f(x, y) = (x / y) - (y / x)


def f(x, y):
	return ((x/y) - (y/x))

# Yk_plus1 = Yk + 1/6 * (k1 + 2*k2 + 2*k3 + k4)
# k1 = h*f(xk, Yk)
# k2 = h*f(xk + h/2, Yk + k1/2)
# k3 = h*f(xk + h/2, Yk + k2/2)
# k4 = h*f(xk + h, Yk + k3)

# 1 passo
k = 0
x0 = 1
y0 = 2
h = 0.1

k1 = h*f(x0, y0)
k2 = h*f(x0 + h/2, y0 + k1/2)
k3 = h*f(x0 + h/2, y0 + k2/2)
k4 = h*f(x0 + h, y0 + k3)

y1 = y0 + 1/6 * (k1 + 2*k2 + 2*k3 + k4)
print("y1 =", y1)

# 2 passo
k = 1
x1 = 1.1
h = 0.1

k1 = h*f(x1, y1)
k2 = h*f(x1 + h/2, y1 + k1/2)
k3 = h*f(x1 + h/2, y1 + k2/2)
k4 = h*f(x1 + h, y1 + k3)

y2 = y1 + 1/6 * (k1 + 2*k2 + 2*k3 + k4)
print("y2 =", y2)
print('\n')

# Real example for Mechanical Dynamics:
# Simple harmonic oscillator with Runge Kutta 4th order method:

import numpy as np
import matplotlib.pyplot as plt

m = 1.0
k = 4.0
dt = 0.01
t_max = 20

def sistema(t, y):
    x, v = y
    dxdt = v
    dvdt = -(k/m) * x
    return np.array([dxdt, dvdt])

def rk4_step(f, t, y, dt):
    k1 = f(t, y)
    k2 = f(t + dt/2, y + dt*k1/2)
    k3 = f(t + dt/2, y + dt*k2/2)
    k4 = f(t + dt, y + dt*k3)
    return y + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

N = int(t_max / dt)
t = np.zeros(N + 1)
Y = np.zeros((N + 1, 2))

Y[0] = [1.0, 0.0]  # x0, v0

for i in range(N):
    t[i+1] = t[i] + dt
    Y[i+1] = rk4_step(sistema, t[i], Y[i], dt)

x = Y[:, 0]
v = Y[:, 1]

plt.plot(t, x)
plt.xlabel("Time")
plt.ylabel("Position")
plt.title("Simple Harmonic Oscillator - RK4")
plt.grid()
plt.show()