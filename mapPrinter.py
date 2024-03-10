import pickle

import matplotlib.pyplot as plt

with open('states.pkl', 'rb') as f:
    data = pickle.load(f)

print(len(data))
positions = []
for i in range(len(data)):
    positions.append([data[i][0][0], data[i][0][2]])

x = [point[0] for point in positions]
y = [point[1] for point in positions]

# Plot the points
plt.scatter(x, y)
plt.title('Position Plot')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.show()
