import pickle
import numpy as np

with open('common/scale_factors.pkl', 'rb') as file:
    data = pickle.load(file)
    data2 = pickle.load(file)

print(data)
print(data2)

# Define indexes to remove (1st, 2nd, and 5th)
indexes_to_remove = [0, 1, 4]

# Remove specified indexes from each element
data = np.round(np.delete(data, indexes_to_remove), decimals=7)
data2 = np.round(np.delete(data2, indexes_to_remove), decimals=7)


with open('common/scale_factors.pkl', 'wb') as file:
    pickle.dump(data, file)
    pickle.dump(data2, file)

with open('common/scale_factors.pkl', 'rb') as file:
    data = pickle.load(file)
    data2 = pickle.load(file)

print(data)
print(data2)
