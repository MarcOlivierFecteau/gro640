# ~/usr/bin/env python3

# Exercice 2.2 - Statique d'un robot manipulateur

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

l1 = l2 = 3.0  # m
q1 = np.linspace(-np.pi, np.pi, 100)  # rad
q2 = np.linspace(-np.pi, np.pi, 100)  # rad
F = 1  # N


def J(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [l1 * np.sin(q1) + l2 * np.sin(q1 + q2), l2 * np.sin(q1 + q2)],
            [l1 * np.cos(q1) + l2 * np.cos(q1 + q2), l2 * np.cos(q1 + q2)],
        ],
        dtype=np.float64,
    )


def q2torque(J):
    fe = np.array([0, -1])
    tau = J.T @ fe

    return tau


tau1 = np.zeros((np.shape(q1)[0], np.shape(q2)[0]), dtype=np.float64)
tau2 = np.zeros((np.shape(q1)[0], np.shape(q2)[0]), dtype=np.float64)
for i in range(len(q1)):
    for j in range(len(q2)):
        penis = J(q1[i], q2[j])
        tau1[i, j], tau2[i, j] = q2torque(penis)

# Plot the surface
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# 3. Plot the surface
# 'cmap' sets the colormap
# 'rstride' and 'cstride' control the density of the grid lines on the surface
# If you want a solid surface without grid lines, you can remove rstride/cstride or set them to 1.
X, Y = np.meshgrid(q1, q2)
surface = ax.plot_surface(X, Y, tau1, cmap="viridis", rstride=1, cstride=1)

# 4. Add labels and a title
ax.set_xlabel("q1 (rad)")
ax.set_ylabel("q2 (rad)")
ax.set_zlabel("tau (Nm)")
ax.set_title("Interactive 2D Surface Plot")

# 5. Add a color bar to indicate Z values
fig.colorbar(surface, shrink=0.5, aspect=5)

# 6. Show the plot
# For interactive plots, plt.show() might block execution until the window is closed.
# With plt.ion() at the beginning, it might open the window and allow further code execution.
# In a typical script, you'd still call plt.show() at the end.
plt.show()
