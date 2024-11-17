Web VPython 3.2
# Helper functions for array operations
# Utility functions
def add(array1, array2):
    """Add two arrays together."""
    if isinstance(array1, list) and isinstance(array2, list):
        result = []
        for x, y in zip(array1, array2):
            result.append(x + y)
        return result
    elif isinstance(array1, list):
        result = []
        for x in array1:
            result.append(x + array2)
        return result
    elif isinstance(array2, list):
        result = []
        for y in array2:
            result.append(array1 + y)
        return result
    return array1 + array2

def scale(array, factor):
    """Scale each element in the input array by the given factor."""
    if isinstance(array, list):
        result = []
        for x in array:
            result.append(x * factor)
        return result
    return array * factor

def odeint(y0, t, g, L1, L2, m1, m2):
    """Solve the first-order ODE using the Runge-Kutta 4(5) method."""
    dt = t[1] - t[0]
    y = [y0]
    
    def double_pendulum_derivatives(y, t):
        """Compute the derivatives for the double pendulum system."""
        theta1, omega1, theta2, omega2 = y
        delta = theta2 - theta1
        denom1 = (m1 + m2) * L1 - m2 * L1 * cos(delta) ** 2
        denom2 = (L2 / L1) * denom1

        dtheta1_dt = omega1
        dtheta2_dt = omega2

        domega1_dt = ((m2 * L1 * omega1 ** 2 * sin(delta) * cos(delta) +
                       m2 * g * sin(theta2) * cos(delta) +
                       m2 * L2 * omega2 ** 2 * sin(delta) -
                       (m1 + m2) * g * sin(theta1)) / denom1)
        
        domega2_dt = ((-m2 * L2 * omega2 ** 2 * sin(delta) * cos(delta) +
                       (m1 + m2) * g * sin(theta1) * cos(delta) -
                       (m1 + m2) * L1 * omega1 ** 2 * sin(delta) -
                       (m1 + m2) * g * sin(theta2)) / denom2)

        return [dtheta1_dt, domega1_dt, dtheta2_dt, domega2_dt]

    # Runge-Kutta 4(5) integration loop
    for i in range(len(t) - 1):
        k1 = double_pendulum_derivatives(y[i], t[i])
        k2 = double_pendulum_derivatives(add(scale(k1, 0.25 * dt), y[i]), t[i] + 0.25 * dt)
        k3 = double_pendulum_derivatives(add(add(scale(k1, 3/32 * dt), scale(k2, 9/32 * dt)), y[i]), t[i] + 3/8 * dt)
        k4 = double_pendulum_derivatives(add(add(add(scale(k1, 1932/2197 * dt), scale(k2, -7200/2197 * dt)), scale(k3, 7296/2197 * dt)), y[i]), t[i] + 12/13 * dt)
        k5 = double_pendulum_derivatives(add(add(add(add(scale(k1, 439/216 * dt), scale(k2, -8 * dt)), scale(k3, 3680/513 * dt)), scale(k4, -845/4104 * dt)), y[i]), t[i] + dt)
        k6 = double_pendulum_derivatives(add(add(add(add(add(scale(k1, -8/27 * dt), scale(k2, 2 * dt)), scale(k3, -3544/2565 * dt)), scale(k4, 1859/4104 * dt)), scale(k5, -11/40 * dt)), y[i]), t[i] + 0.5 * dt)

        y_next = add(add(add(add(add(scale(k1, 16/135 * dt), scale(k3, 6656/12825 * dt)), scale(k4, 28561/56430 * dt)), scale(k5, -9/50 * dt)), scale(k6, 2/55 * dt)), y[i])
        y.append(y_next)

    # Return the results as a transposed list of lists for each variable (theta1, omega1, theta2, omega2)
    if isinstance(y[0], list):
        y_arr = []
        for i in range(len(y[0])):
            y_arr.append([y[j][i] for j in range(len(y))])
        return y_arr
    return [y]
    
# Define initial conditions
y0 = [pi / 2, 0, pi / 2, 0]  # Initial positions and velocities
t = [i * 0.05 for i in range(1000)]  # Time steps

# Physical parameters
g = 9.81  # Gravitational acceleration
L1 = 1.0  # Length of first pendulum
L2 = 1.0  # Length of second pendulum
m1 = 1.0  # Mass of first pendulum
m2 = 1.0  # Mass of second pendulum

# Solve the system
solution = odeint(y0, t, g, L1, L2, m1, m2)

# Extract solved data for animation
theta1_vals = solution[0]
theta2_vals = solution[2]

# VPython setup for animation
pivot = vector(0, 0, 0)
ball1 = sphere(pos=vector(L1 * sin(theta1_vals[0]), -L1 * cos(theta1_vals[0]), 0), radius=0.05, color=vector(1, 0, 0))
ball2 = sphere(pos=vector(L1 * sin(theta1_vals[0]) + L2 * sin(theta2_vals[0]),
                          -L1 * cos(theta1_vals[0]) - L2 * cos(theta2_vals[0]), 0), radius=0.05, color=vector(0, 0, 1))
rod1 = curve(color=vector(1, 1, 1))
rod2 = curve(color=vector(1, 1, 1))
trail = curve(color=vector(0.5, 0.5, 0.5), radius=0.01)

# Animation loop
for i in range(len(solution[0])):
    rate(100)  # Limit the update rate to 100 frames per second

    # Update positions based on solved values
    theta1 = theta1_vals[i]
    theta2 = theta2_vals[i]

    ball1.pos = vector(L1 * sin(theta1), -L1 * cos(theta1), 0)
    ball2.pos = vector(ball1.pos.x + L2 * sin(theta2), ball1.pos.y - L2 * cos(theta2), 0)

    # Update rods
    rod1.clear()
    rod1.append(pivot, ball1.pos)
    rod2.clear()
    rod2.append(ball1.pos, ball2.pos)

    # Update trail
    trail.append(ball2.pos)