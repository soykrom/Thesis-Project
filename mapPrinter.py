import json
import matplotlib.pyplot as plt

with open('data.json', 'r') as f:
    data = json.load(f)

positions = []
directions = []
for i in range(len(data)):
	positions.append([data[i]["telemetry"]["positionX"], data[i]["telemetry"]["positionZ"]])
        
for value in data[0]["telemetry"]:
      print(value)

x = [point[0] for point in positions]
y = [point[1] for point in positions]
u = [point[0] for point in directions]
v = [point[1] for point in directions]

# Create a figure and axis object
fig, ax = plt.subplots()

# Plot the data as a scatter plot
ax.quiver(x, y, u, v, angles='xy', scale_units='xy', scale=50)
ax.scatter(x, y)

# Remove the x and y axis labels
ax.set_xticklabels([])
ax.set_yticklabels([])

# Remove the x and y axis ticks
ax.tick_params(axis='both', which='both', length=0)

# Show the plot
plt.show()