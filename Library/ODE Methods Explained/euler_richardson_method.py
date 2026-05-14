# Euler-Richardson Method

# ============================================================
# PART 1 - Initial Value Problem (IVP)
# ============================================================

# Initial Value Problem (IVP)
# Differential equation in differential form:
# (y^2 - x^2) dx + x*y dy = 0
#
# Explicit form:
# dy/dx = (x / y) - (y / x)
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
# Approximate y(1.2) using the Euler-Richardson method


def f_ivp(x, y):
    return (x / y) - (y / x)


def euler_richardson_step_scalar(f, x, y, h):
    k1 = f(x, y)
    y_mid = y + (h / 2) * k1
    k2 = f(x + h / 2, y_mid)
    y_next = y + h * k2
    return y_next


# Initial values
x0 = 1.0
y0 = 2.0
h = 0.1
x_final = 1.2

# Lists to store the numerical solution
x_values_ivp = [x0]
y_values_ivp = [y0]

# Number of steps
n_steps_ivp = int((x_final - x0) / h)

# Euler-Richardson loop
x = x0
y = y0

for _ in range(n_steps_ivp):
    y = euler_richardson_step_scalar(f_ivp, x, y, h)
    x = x + h

    x_values_ivp.append(x)
    y_values_ivp.append(y)

# Print IVP results
print("IVP solved with Euler-Richardson:")
for xi, yi in zip(x_values_ivp, y_values_ivp):
    print(f"x = {xi:.1f}, y = {yi:.6f}")

print("\nApproximation of y(1.2):", f"{y_values_ivp[-1]:.6f}")
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

import numpy as np
import matplotlib.pyplot as plt

# Parameters
m = 1.0
k = 4.0
dt = 0.01
t_max = 20.0


def sho_system(t, y):
    x, v = y
    dxdt = v
    dvdt = -(k / m) * x
    return np.array([dxdt, dvdt])


def euler_richardson_step_vector(f, t, y, dt):
    k1 = f(t, y)
    y_mid = y + (dt / 2) * k1
    k2 = f(t + dt / 2, y_mid)
    y_next = y + dt * k2
    return y_next


def x_analytical(t):
    return np.cos(2 * t)


def v_analytical(t):
    return -2 * np.sin(2 * t)


# Time array size
n_steps_sho = int(t_max / dt)

# Arrays to store numerical solution
t_values = np.zeros(n_steps_sho + 1)
Y_values = np.zeros((n_steps_sho + 1, 2))

# Initial conditions
Y_values[0] = [1.0, 0.0]   # [x0, v0]

# Euler-Richardson loop
for i in range(n_steps_sho):
    t_values[i + 1] = t_values[i] + dt
    Y_values[i + 1] = euler_richardson_step_vector(sho_system, t_values[i], Y_values[i], dt)

# Extract numerical position and velocity
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
plt.plot(t_values, x_numerical, label="Euler-Richardson Numerical Solution")
plt.plot(t_values, x_exact, "--", label="Analytical Solution")
plt.xlabel("Time")
plt.ylabel("Position")
plt.title("Simple Harmonic Oscillator - Position Comparison")
plt.legend()
plt.grid()
plt.show()

# Velocity comparison
plt.figure(figsize=(10, 5))
plt.plot(t_values, v_numerical, label="Euler-Richardson Numerical Solution")
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

# Final errors
print("Final absolute error in position:", x_absolute_error[-1])
print("Final absolute error in velocity:", v_absolute_error[-1])