import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# SIMPLE HARMONIC OSCILLATOR (SHO)
# ============================================================
# Equation:
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
# ============================================================

m = 1.0
k = 4.0
dt = 0.1
t_max = 20.0

omega = np.sqrt(k / m)
n_steps = int(t_max / dt)

# ------------------------------------------------------------
# Analytical solution
# ------------------------------------------------------------
def x_analytical(t):
    return np.cos(omega * t)

def v_analytical(t):
    return -omega * np.sin(omega * t)

# ------------------------------------------------------------
# SHO system for first-order methods
# ------------------------------------------------------------
def sho_system(t, y):
    x, v = y
    dxdt = v
    dvdt = -(k / m) * x
    return np.array([dxdt, dvdt])

# ------------------------------------------------------------
# Acceleration for Velocity Verlet
# ------------------------------------------------------------
def acceleration(x):
    return -(k / m) * x

# ============================================================
# NUMERICAL METHODS
# ============================================================

# ------------------------------------------------------------
# Euler-Richardson
# ------------------------------------------------------------
def euler_richardson_step(f, t, y, dt):
    k1 = f(t, y)
    y_mid = y + (dt / 2) * k1
    k2 = f(t + dt / 2, y_mid)
    return y + dt * k2

def solve_euler_richardson():
    t_values = np.zeros(n_steps + 1)
    Y_values = np.zeros((n_steps + 1, 2))
    Y_values[0] = [1.0, 0.0]

    for i in range(n_steps):
        t_values[i + 1] = t_values[i] + dt
        Y_values[i + 1] = euler_richardson_step(sho_system, t_values[i], Y_values[i], dt)

    return t_values, Y_values[:, 0], Y_values[:, 1]

# ------------------------------------------------------------
# RK4
# ------------------------------------------------------------
def rk4_step(f, t, y, dt):
    k1 = f(t, y)
    k2 = f(t + dt / 2, y + dt * k1 / 2)
    k3 = f(t + dt / 2, y + dt * k2 / 2)
    k4 = f(t + dt, y + dt * k3)
    return y + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

def solve_rk4():
    t_values = np.zeros(n_steps + 1)
    Y_values = np.zeros((n_steps + 1, 2))
    Y_values[0] = [1.0, 0.0]

    for i in range(n_steps):
        t_values[i + 1] = t_values[i] + dt
        Y_values[i + 1] = rk4_step(sho_system, t_values[i], Y_values[i], dt)

    return t_values, Y_values[:, 0], Y_values[:, 1]

# ------------------------------------------------------------
# Velocity Verlet
# ------------------------------------------------------------
def solve_velocity_verlet():
    t_values = np.zeros(n_steps + 1)
    x_values = np.zeros(n_steps + 1)
    v_values = np.zeros(n_steps + 1)
    a_values = np.zeros(n_steps + 1)

    x_values[0] = 1.0
    v_values[0] = 0.0
    a_values[0] = acceleration(x_values[0])

    for i in range(n_steps):
        t_values[i + 1] = t_values[i] + dt

        x_values[i + 1] = (
            x_values[i]
            + v_values[i] * dt
            + 0.5 * a_values[i] * dt**2
        )

        a_values[i + 1] = acceleration(x_values[i + 1])

        v_values[i + 1] = (
            v_values[i]
            + 0.5 * (a_values[i] + a_values[i + 1]) * dt
        )

    return t_values, x_values, v_values

# ------------------------------------------------------------
# Predictor-Corrector (AB4-AM4)
# ------------------------------------------------------------
def solve_predictor_corrector():
    t_values = [0.0]
    Y_values = [np.array([1.0, 0.0])]

    # Generate first 3 extra points using RK4
    for _ in range(3):
        y_next = rk4_step(sho_system, t_values[-1], Y_values[-1], dt)
        t_values.append(t_values[-1] + dt)
        Y_values.append(y_next)

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

    t_values = np.array(t_values)
    Y_values = np.array(Y_values)

    return t_values, Y_values[:, 0], Y_values[:, 1]

# ============================================================
# ERROR FUNCTIONS
# ============================================================
def relative_error(numerical, exact):
    denominator = np.maximum(np.abs(exact), 1e-12)
    return np.abs(numerical - exact) / denominator

# ============================================================
# PLOT FUNCTION - 4 GRAPHS IN ONE SCREEN
# ============================================================
def plot_method_dashboard(method_name, t_values, x_num, v_num):
    x_exact = x_analytical(t_values)
    v_exact = v_analytical(t_values)

    x_rel_error = relative_error(x_num, x_exact)
    v_rel_error = relative_error(v_num, v_exact)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f"{method_name} - SHO Dashboard", fontsize=14)

    # Position vs Time
    axes[0, 0].plot(t_values, x_num, label="Numerical")
    axes[0, 0].plot(t_values, x_exact, "--", label="Analytical")
    axes[0, 0].set_title("Position vs Time")
    axes[0, 0].set_xlabel("Time")
    axes[0, 0].set_ylabel("Position")
    axes[0, 0].grid()
    axes[0, 0].legend()

    # Velocity vs Time
    axes[0, 1].plot(t_values, v_num, label="Numerical")
    axes[0, 1].plot(t_values, v_exact, "--", label="Analytical")
    axes[0, 1].set_title("Velocity vs Time")
    axes[0, 1].set_xlabel("Time")
    axes[0, 1].set_ylabel("Velocity")
    axes[0, 1].grid()
    axes[0, 1].legend()

    # Relative Error in Position
    axes[1, 0].plot(t_values, x_rel_error)
    axes[1, 0].set_title("Relative Error - Position")
    axes[1, 0].set_xlabel("Time")
    axes[1, 0].set_ylabel("Relative Error")
    axes[1, 0].grid()

    # Relative Error in Velocity
    axes[1, 1].plot(t_values, v_rel_error)
    axes[1, 1].set_title("Relative Error - Velocity")
    axes[1, 1].set_xlabel("Time")
    axes[1, 1].set_ylabel("Relative Error")
    axes[1, 1].grid()

    plt.tight_layout()
    plt.show()

# ============================================================
# RUN ALL METHODS
# ============================================================
t_er, x_er, v_er = solve_euler_richardson()
t_rk4, x_rk4, v_rk4 = solve_rk4()
t_vv, x_vv, v_vv = solve_velocity_verlet()
t_pc, x_pc, v_pc = solve_predictor_corrector()

# ============================================================
# INDIVIDUAL DASHBOARDS
# ============================================================
plot_method_dashboard("Euler-Richardson", t_er, x_er, v_er)
plot_method_dashboard("RK4", t_rk4, x_rk4, v_rk4)
plot_method_dashboard("Velocity Verlet", t_vv, x_vv, v_vv)
plot_method_dashboard("Predictor-Corrector", t_pc, x_pc, v_pc)

# ============================================================
# GLOBAL COMPARISON PLOTS
# ============================================================

# Analytical solution for the common time grid
x_exact_er = x_analytical(t_er)
v_exact_er = v_analytical(t_er)

x_exact_rk4 = x_analytical(t_rk4)
v_exact_rk4 = v_analytical(t_rk4)

x_exact_vv = x_analytical(t_vv)
v_exact_vv = v_analytical(t_vv)

x_exact_pc = x_analytical(t_pc)
v_exact_pc = v_analytical(t_pc)

# Relative errors
x_error_er = relative_error(x_er, x_exact_er)
v_error_er = relative_error(v_er, v_exact_er)

x_error_rk4 = relative_error(x_rk4, x_exact_rk4)
v_error_rk4 = relative_error(v_rk4, v_exact_rk4)

x_error_vv = relative_error(x_vv, x_exact_vv)
v_error_vv = relative_error(v_vv, v_exact_vv)

x_error_pc = relative_error(x_pc, x_exact_pc)
v_error_pc = relative_error(v_pc, v_exact_pc)

# ------------------------------------------------------------
# 1. Position vs Time - All methods
# ------------------------------------------------------------
plt.figure(figsize=(12, 6))
plt.plot(t_er, x_er, label="Euler-Richardson")
plt.plot(t_rk4, x_rk4, label="RK4")
plt.plot(t_vv, x_vv, label="Velocity Verlet")
plt.plot(t_pc, x_pc, label="Predictor-Corrector")
plt.plot(t_rk4, x_exact_rk4, "--", label="Analytical Solution")
plt.xlabel("Time")
plt.ylabel("Position")
plt.title("SHO - Position vs Time (All Methods)")
plt.legend()
plt.grid()
plt.show()

# ------------------------------------------------------------
# 2. Velocity vs Time - All methods
# ------------------------------------------------------------
plt.figure(figsize=(12, 6))
plt.plot(t_er, v_er, label="Euler-Richardson")
plt.plot(t_rk4, v_rk4, label="RK4")
plt.plot(t_vv, v_vv, label="Velocity Verlet")
plt.plot(t_pc, v_pc, label="Predictor-Corrector")
plt.plot(t_rk4, v_exact_rk4, "--", label="Analytical Solution")
plt.xlabel("Time")
plt.ylabel("Velocity")
plt.title("SHO - Velocity vs Time (All Methods)")
plt.legend()
plt.grid()
plt.show()

# ------------------------------------------------------------



methods = ["Euler-Richardson", "RK4", "Velocity Verlet", "Predictor-Corrector"]

mean_abs_x_errors = [
    np.mean(np.abs(x_er - x_exact_er)),
    np.mean(np.abs(x_rk4 - x_exact_rk4)),
    np.mean(np.abs(x_vv - x_exact_vv)),
    np.mean(np.abs(x_pc - x_exact_pc))
]

max_abs_x_errors = [
    np.max(np.abs(x_er - x_exact_er)),
    np.max(np.abs(x_rk4 - x_exact_rk4)),
    np.max(np.abs(x_vv - x_exact_vv)),
    np.max(np.abs(x_pc - x_exact_pc))
]

plt.figure(figsize=(10, 5))
plt.bar(methods, mean_abs_x_errors, label="Mean Absolute Error")
plt.bar(methods, max_abs_x_errors, label="Max Absolute Error", alpha=0.7)
plt.ylabel("Absolute Error")
plt.title("Mean and Max Absolute Error in Position")
plt.legend()
plt.grid(axis="y")
plt.show()
