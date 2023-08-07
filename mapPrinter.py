import json
import matplotlib.pyplot as plt
import numpy
import gameControler as gc


gpr_model = gc.get_grp_model()

with open('data.json', 'r') as f:
    data = json.load(f)

lapDist, positions, scaler = gc.get_trajectory()

# Create a figure and axis object
fig, ax = plt.subplots()

# Plot the data as a scatter plot
ax.scatter(positions)

# Remove the x and y axis labels
# ax.set_xticklabels([])
# ax.set_yticklabels([])
ax.axhline(0, color='black')
ax.axvline(0, color='black')

# Remove the x and y axis ticks
ax.tick_params(axis='both', which='both', length=0)

ax.set_xlim(50, 90)
ax.set_ylim(-135, -205)

ax.set_aspect('equal')
# Show the plot
plt.show()
