# Adams-Bashforth / Adams-Moulton 4th Order Predictor-Corrector Method

# ============================================================
# PART 1 - Initial Value Problem (IVP)
# ============================================================

# Initial Value Problem (IVP)
# Differential equation:
# y' = 1 / (y^2) + x * ln(y)
#
# Initial condition:
# y(1) = 0.8
#
# Step size:
# h = 0.2
#
# Objective:
# Approximate y(2.0) using the Predictor-Corrector method
#
# Known starting values obtained previously with RK4:
# x0 = 1.0, y0 = 0.8
# x1 = 1.2, y1 = 1.0227
# x2 = 1.4, y2 = 1.2131
# x3 = 1.6, y3 = 1.4116
#
# Tolerance:
# eps = 0.5 * 10^(-2)

import math
import numpy as np
import matplotlib.pyplot as plt


def f_ivp(x, y):
    return 1 / (y**2) + x * math.log(y, math.e)


def relative_error(new_value, old_value):
    return abs(new_value - old_value) / abs(new_value)


# Initial known points
h = 0.2
eps = 0.5 * 10**(-2)

x_values_ivp = [1.0, 1.2, 1.4, 1.6]
y_values_ivp = [0.8, 1.0227, 1.2131, 1.4116]

x_final_ivp = 2.0

while x_values_ivp[-1] < x_final_ivp:
    n = len(y_values_ivp) - 1
    x_next = x_values_ivp[n] + h

    # Predictor: Adams-Bashforth 4
    y_pred = y_values_ivp[n] + (h / 24) * (
        55 * f_ivp(x_values_ivp[n],     y_values_ivp[n])
        - 59 * f_ivp(x_values_ivp[n-1], y_values_ivp[n-1])
        + 37 * f_ivp(x_values_ivp[n-2], y_values_ivp[n-2])
        - 9  * f_ivp(x_values_ivp[n-3], y_values_ivp[n-3])
    )

    # Corrector: Adams-Moulton 4
    y_old = y_pred

    while True:
        y_corr = y_values_ivp[n] + (h / 24) * (
            9  * f_ivp(x_next, y_old)
            + 19 * f_ivp(x_values_ivp[n],     y_values_ivp[n])
            - 5  * f_ivp(x_values_ivp[n-1],   y_values_ivp[n-1])
            +     f_ivp(x_values_ivp[n-2],    y_values_ivp[n-2])
        )

        if relative_error(y_corr, y_old) < eps:
            break

        y_old = y_corr

    x_values_ivp.append(x_next)
    y_values_ivp.append(y_corr)

print("IVP solved with Predictor-Corrector:")
for xi, yi in zip(x_values_ivp, y_values_ivp):
    print(f"x = {xi:.1f}, y = {yi:.6f}")

print("\nApproximation of y(2.0):", f"{y_values_ivp[-1]:.6f}")
print()


# ============================================================
# PART 2 - Simple Harmonic Oscillator (SHO)
# ============================================================

# Simple Harmonic Oscillator:
# m*x'' + k*x = 0
#
# Equivalent first-order system:
# x' = v
# v' = -(k/m) * x
#
# Initial conditions:
# x(0) = 1
# v(0) = 0
#
# Parameters:
# m = 1
# k = 4
#
# Analytical solution:
# x(t) = cos(2t)
# v(t) = -2*sin(2t)


# SHO parameters
m = 1.0
k = 4.0
dt = 0.01
t_max = 20.0


def sho_system(t, y):
    x, v = y
    dxdt = v
    dvdt = -(k / m) * x
    return np.array([dxdt, dvdt])


def rk4_step(f, t, y, dt):
    k1 = f(t, y)
    k2 = f(t + dt / 2, y + dt * k1 / 2)
    k3 = f(t + dt / 2, y + dt * k2 / 2)
    k4 = f(t + dt, y + dt * k3)
    return y + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def x_analytical(t):
    return np.cos(2 * t)


def v_analytical(t):
    return -2 * np.sin(2 * t)


# Initial conditions
t_values = [0.0]
Y_values = [np.array([1.0, 0.0])]   # [x0, v0]

# Generate the first 3 additional points using RK4
for _ in range(3):
    y_next = rk4_step(sho_system, t_values[-1], Y_values[-1], dt)
    t_values.append(t_values[-1] + dt)
    Y_values.append(y_next)

# Predictor-Corrector loop
while t_values[-1] < t_max:
    n = len(Y_values) - 1
    t_next = t_values[n] + dt

    f_n  = sho_system(t_values[n],   Y_values[n])
    f_n1 = sho_system(t_values[n-1], Y_values[n-1])
    f_n2 = sho_system(t_values[n-2], Y_values[n-2])
    f_n3 = sho_system(t_values[n-3], Y_values[n-3])

    # Predictor: Adams-Bashforth 4
    y_pred = Y_values[n] + (dt / 24) * (
        55 * f_n - 59 * f_n1 + 37 * f_n2 - 9 * f_n3
    )

    # Corrector: Adams-Moulton 4
    f_pred = sho_system(t_next, y_pred)
    y_corr = Y_values[n] + (dt / 24) * (
        9 * f_pred + 19 * f_n - 5 * f_n1 + f_n2
    )

    t_values.append(t_next)
    Y_values.append(y_corr)

# Convert to arrays
t_values = np.array(t_values)
Y_values = np.array(Y_values)

# Numerical solution
x_numerical = Y_values[:, 0]
v_numerical = Y_values[:, 1]

# Analytical solution
x_exact = x_analytical(t_values)
v_exact = v_analytical(t_values)

# Absolute errors
x_absolute_error = np.abs(x_numerical - x_exact)
v_absolute_error = np.abs(v_numerical - v_exact)

# ============================================================
# PLOTS
# ============================================================

# Position comparison
plt.figure(figsize=(10, 5))
plt.plot(t_values, x_numerical, label="Predictor-Corrector Numerical Solution")
plt.plot(t_values, x_exact, "--", label="Analytical Solution")
plt.xlabel("Time")
plt.ylabel("Position")
plt.title("Simple Harmonic Oscillator - Position Comparison")
plt.legend()
plt.grid()
plt.show()

# Velocity comparison
plt.figure(figsize=(10, 5))
plt.plot(t_values, v_numerical, label="Predictor-Corrector Numerical Solution")
plt.plot(t_values, v_exact, "--", label="Analytical Solution")
plt.xlabel("Time")
plt.ylabel("Velocity")
plt.title("Simple Harmonic Oscillator - Velocity Comparison")
plt.legend()
plt.grid()
plt.show()

# Absolute error plot
plt.figure(figsize=(10, 5))
plt.plot(t_values, x_absolute_error, label="Position Absolute Error")
plt.plot(t_values, v_absolute_error, label="Velocity Absolute Error")
plt.xlabel("Time")
plt.ylabel("Absolute Error")
plt.title("Absolute Error Relative to the Analytical Solution")
plt.legend()
plt.grid()
plt.show()

# Final error values
print("Final absolute error in position:", x_absolute_error[-1])
print("Final absolute error in velocity:", v_absolute_error[-1])