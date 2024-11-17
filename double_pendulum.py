import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import RK45
from matplotlib.animation import FuncAnimation

# Constants
g = 9.81  # Gravity, m/s^2
L1 = 1.0  # Length of the first pendulum, m
L2 = 1.0  # Length of the second pendulum, m
m1 = 1.0  # Mass of the first pendulum, kg
m2 = 1.0  # Mass of the second pendulum, kg

# Equations of motion
def double_pendulum_derivatives(t, y):
    theta1, omega1, theta2, omega2 = y

    delta = theta2 - theta1
    denom1 = (m1 + m2) * L1 - m2 * L1 * np.cos(delta) ** 2
    denom2 = (L2 / L1) * denom1

    dtheta1_dt = omega1
    dtheta2_dt = omega2

    domega1_dt = ((m2 * L1 * omega1 ** 2 * np.sin(delta) * np.cos(delta) +
                   m2 * g * np.sin(theta2) * np.cos(delta) +
                   m2 * L2 * omega2 ** 2 * np.sin(delta) -
                   (m1 + m2) * g * np.sin(theta1)) / denom1)
    
    domega2_dt = ((-m2 * L2 * omega2 ** 2 * np.sin(delta) * np.cos(delta) +
                   (m1 + m2) * g * np.sin(theta1) * np.cos(delta) -
                   (m1 + m2) * L1 * omega1 ** 2 * np.sin(delta) -
                   (m1 + m2) * g * np.sin(theta2)) / denom2)

    return [dtheta1_dt, domega1_dt, dtheta2_dt, domega2_dt]

# Initial conditions
theta1_0 = np.pi / 2  # Initial angle for the first pendulum, rad
omega1_0 = 0.0        # Initial angular velocity for the first pendulum, rad/s
theta2_0 = np.pi / 2  # Initial angle for the second pendulum, rad
omega2_0 = 0.0        # Initial angular velocity for the second pendulum, rad/s
y0 = [theta1_0, omega1_0, theta2_0, omega2_0]

# Time span
t0 = 0.0
t_max = 20.0
dt = 0.01
times = np.arange(t0, t_max, dt)

# Run the RK45 simulation
solver = RK45(double_pendulum_derivatives, t0, y0, t_max, max_step=dt)
results = []
time_results = []

while solver.status == 'running':
    solver.step()
    results.append(solver.y)
    time_results.append(solver.t)

results = np.array(results)
time_results = np.array(time_results)

# Extract results
theta1_vals, omega1_vals, theta2_vals, omega2_vals = results.T

# Convert to Cartesian coordinates for visualization
x1 = L1 * np.sin(theta1_vals)
y1 = -L1 * np.cos(theta1_vals)

x2 = x1 + L2 * np.sin(theta2_vals)
y2 = y1 - L2 * np.cos(theta2_vals)

# Animation setup
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-2 * (L1 + L2), 2 * (L1 + L2))
ax.set_ylim(-2 * (L1 + L2), 2 * (L1 + L2))
ax.set_aspect('equal')
ax.grid()

line, = ax.plot([], [], 'o-', lw=2, label="Pendulum Path")
trail, = ax.plot([], [], 'r-', lw=0.5, alpha=0.5, label="Trail")
trail_x, trail_y = [], []

def init():
    line.set_data([], [])
    trail.set_data([], [])
    return line, trail

def update(frame):
    trail_x.append(x2[frame])
    trail_y.append(y2[frame])
    line.set_data([0, x1[frame], x2[frame]], [0, y1[frame], y2[frame]])
    trail.set_data(trail_x, trail_y)
    return line, trail

ani = FuncAnimation(fig, update, frames=len(times), init_func=init, blit=True, interval=dt * 200)
plt.legend()
plt.title("Double Pendulum Animation")
plt.show()
