import pickle

# Number of lists you want to generate
from itertools import product

# Define the range for each element (min and max values)
path_lateral_range = (0.75, 2.0)
distance_range = (1.0, 2.5)
done_range = (0.5, 1.2)

# Define the number of values to generate for each parameter
num_values = 4  # You can adjust this based on your preference

# Generate values at intervals for each parameter
path_lateral_values = [round(val, 2) for val in
                       sorted(set([path_lateral_range[0] + i * (path_lateral_range[1] - path_lateral_range[0]) /
                                  (num_values - 1) for i in range(num_values)]))]

distance_values = [round(val, 2) for val in
                   sorted(set([distance_range[0] + i * (distance_range[1] - distance_range[0]) /
                              (num_values - 1) for i in range(num_values)]))]

done_values = [round(val, 2) for val in
               sorted(set([done_range[0] + i * (done_range[1] - done_range[0]) /
                          (num_values - 1) for i in range(num_values)]))]

# Generate combinations of parameter values
combinations = list(product(path_lateral_values, distance_values, done_values))

print(combinations)
with open('common/coefficients.pkl', 'wb') as filename:
    pickle.dump(combinations, filename)

