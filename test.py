import numpy
import matplotlib.pyplot as plt


def calculate_throttle_action(speed):
    diff = 50 - speed

    action = diff / max(50, speed)

    return max(-1, min(1, action))


values = numpy.linspace(0, 100, 25)

actions = []
squared_actions = []
for value in values:
    throttle_action = calculate_throttle_action(value)

    squared_action = throttle_action
    squared_actions.append(squared_action)

    print(f"Current Speed: {value}\tThrottle Action: {throttle_action}\tSquared Action: {squared_action}")

    actions.append(throttle_action)

plt.subplot(121)
plt.plot(values, actions, 'r--')
plt.xlabel("Speed")
plt.ylabel("Action")

plt.subplot(122)
plt.plot(values, squared_actions, 'bs')
plt.xlabel("Speed")
plt.ylabel("Action")

plt.show()
