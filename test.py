import matplotlib.pyplot as plt
import numpy as np


# Function to load data from the file
def load_data(filename):
    max_dists = []
    ep_lens = []
    tests = []

    try:
        with open(filename, "r") as file:
            for line in file:
                max_dist, ep_len, test = map(float, line.strip()[1:-1].split(','))
                max_dists.append(max_dist)
                ep_lens.append(ep_len)
                tests.append(test)
        return max_dists, ep_lens, tests
    except FileNotFoundError:
        print("File not found:", filename)
        return [], []


# Load data from the file
max_dists, ep_len, tests = load_data("max_y.txt")

# Plotting
plt.scatter(ep_len, max_dists, color='blue', alpha=0.5)
plt.xlabel('Tests')
plt.ylabel('Max Distance')
plt.title('Max Distance vs Resets')
plt.grid(True)
plt.show()
