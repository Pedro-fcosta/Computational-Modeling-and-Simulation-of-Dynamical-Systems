# Velocity Verlet Integration
# 
# x_(n+1) = x_n + v_n*dt + 0.5*a_n*dt**2
# a_(n+1) = -(k/m) * x_(n+1)
# v_(n+1) = v_n + 0.5*(a_n + a_(n+1))*dt
#
# Initial Value Problem (IVP)
# Differential equation in differential form:
# m*x'' + kx = 0
#
# Equivalent form:
# x'' = -(k/m) * x
#
# Initial conditions:
# x(0) = 1
# v(0) = x'(0) = 0
#
# Parameters:
# m = 1
# k = 4
#
# Step size:
# dt = 0.01
#
# Objetctive:
# Aproximate x(t) and v(t) using the Velocity Verlet Method
#
# With m = 1 and k = 4, the system is a simple harmonic oscillator with angular frequency ω = sqrt(k/m) = 2. The analytical solution is:
# x(t) = cos(2t)
#
# ==============================================================================================================


import numpy as np
import matplotlib.pyplot as plt
import math


# Parameters
m = 1.0
k = 4.0
dt = 0.01
t_max = 15

# initial conditions
X0 = 1.0
V0 = 0.0

def aceleration(X):
    return -(k/m) * X

A0 = aceleration(X0)


# ==============================================================================================================


t = [0.0]
x = [X0]
v = [V0]
a = [A0]

while t[-1] < t_max:
    x_next = x[-1] + v[-1] * dt + 0.5 * a[-1] * dt**2
    a_next = aceleration(x_next)
    v_next = v[-1] + 0.5 * (a[-1] + a_next) * dt
    t_next = t[-1] + dt

    x.append(x_next)
    v.append(v_next)
    a.append(a_next)
    t.append(t_next)


("""
print("t =", t)
print("x =", x)
print("v =", v)
""")
 
# compare to analytical solution

def x_exact(t):
    return math.cos(2*t)

def v_exact(t):
    return -2 * math.sin(2*t)

x_analitic = [x_exact(ti) for ti in t]
v_analitic = [v_exact(ti) for ti in t]



# plotting
plt.plot(t, x, label="Velocity Verlet")
plt.plot(t, x_analitic, "--", label="Analytical Solution")
plt.xlabel("Time")
plt.ylabel("Position")
plt.title("SHO: Velocity Verlet vs Analytical Solution")
plt.legend()
plt.grid()
plt.show()



# plotting velocity
plt.plot(t, v, label="Velocity Verlet")
plt.plot(t, v_analitic, "--", label="Analytical Solution")
plt.xlabel("Time")
plt.ylabel("Velocity")
plt.title("Velocity Comparison")
plt.legend()
plt.grid()
plt.show()



# plotting absolute error
error_x = np.abs(np.array(x) - np.array(x_analitic))
error_v = np.abs(np.array(v) - np.array(v_analitic))

plt.plot(t, error_x, label="Position Error")
plt.plot(t, error_v, label="Velocity Error")
plt.xlabel("Time")
plt.ylabel("Absolute Error")
plt.title("Error Comparison")
plt.legend()
plt.grid()
plt.show()
