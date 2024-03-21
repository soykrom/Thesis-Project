import matplotlib.pyplot as plt
import numpy as np


def get_indexes(lst, value):
    return [index for index, element in enumerate(lst) if element == value]


# Function to load data from the file
def load_data(filename):
    rewards = []
    resets = []

    try:
        with open(filename, "r") as file:
            for line in file:
                reward, reset = map(float, line.strip()[1:-1].split(',')[-2:])
                rewards.append(reward)
                resets.append(reset)
        return rewards, resets
    except FileNotFoundError:
        print("File not found:", filename)
        return [], []


# Load data from the file
rewards, resets = load_data("positions.txt")
indexes = get_indexes(resets, 586)
print(indexes)
print(rewards[indexes[len(indexes) - 1]])

rewards = [rewards[index] for index in indexes]

results = []
reward_sum = 0
for i in range(len(rewards)):
    results.append([i, rewards[i]])
    reward_sum += rewards[i]

print(reward_sum)
x_values = [result[0] for result in results]
y_values = [result[1] for result in results]
# Plotting
plt.scatter(x_values, y_values, color='blue', alpha=0.5)
plt.xlabel('Tests')
plt.ylabel('Max Distance')
plt.title('Max Distance vs Resets')
plt.grid(True)
plt.show()
