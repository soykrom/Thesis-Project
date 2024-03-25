import pickle

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline


def generate_trajectory():
    # Example key points
    key_points = [[0, 0], [450, 50], [450, 450], [-450, 450], [-450, 50]]

    key_points = np.array(key_points)
    num_points = len(key_points)

    # Append the first key point to the end to satisfy periodic boundary condition
    key_points = np.vstack([key_points, key_points[0]])

    # Generate cubic spline interpolation
    t = np.linspace(0, 1, num_points + 1)  # Adjust the number of points for the appended key point
    cs = CubicSpline(t, key_points, bc_type='periodic')

    # Generate smooth trajectory
    num_samples = 1000  # Adjust the number of samples for smoother trajectory
    t_smooth = np.linspace(0, 1, num_samples)
    trajectory = cs(t_smooth)

    return trajectory, key_points


# Generate trajectory
trajectory, key_points = generate_trajectory()

with open("trajectory.pkl", "wb") as file:
    pickle.dump(trajectory, file)

# Plot trajectory
plt.plot(trajectory[:, 0], trajectory[:, 1], label='Trajectory')
plt.scatter(key_points[:, 0], key_points[:, 1], color='red', label='Key Points')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Generated Trajectory')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
