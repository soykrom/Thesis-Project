import pickle

import matplotlib.pyplot as plt

array = pickle.load(open("environment/common/value_mean.pkl", 'rb'))

# Plotting
plt.plot(range(len(array)), array)
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Array values vs Index')
plt.yscale('log')
plt.grid(True)
plt.show()
