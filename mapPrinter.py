import json
import matplotlib.pyplot as plt

with open('data.json', 'r') as f:
    data = json.load(f)

positions = []
for i in range(len(data)):
	positions.append([data[i]["telemetry"]["positionX"], data[i]["telemetry"]["positionZ"]])

print(positions[0], positions[999])

x = [point[0] for point in positions]
y = [point[1] for point in positions]

# Create a figure and axis object
fig, ax = plt.subplots()

# Plot the data as a scatter plot
ax.scatter(x, y)

# Remove the x and y axis labels
ax.set_xticklabels([])
ax.set_yticklabels([])

# Remove the x and y axis ticks
ax.tick_params(axis='both', which='both', length=0)

# Show the plot
plt.show()