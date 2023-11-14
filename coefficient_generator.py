import pickle
import random

# Number of lists you want to generate
num_lists = 100

# Define the range for each element (min and max values)
path_lateral = (0.75, 2.0)
distance = (0.8, 3.0)
done = (0.5, 1.2)

# List to store the generated lists
generated_list_pl = []
generated_list_dist = []
generated_list_done = []

for _ in range(num_lists):
    # Generate random floats within the specified ranges
    element1 = round(random.uniform(path_lateral[0], path_lateral[1]), 2)
    element2 = round(random.uniform(distance[0], distance[1]), 2)
    element3 = round(random.uniform(done[0], done[1]), 2)

    # Create a list with the generated elements
    generated_list_pl.append(element1)
    generated_list_dist.append(element2)
    generated_list_done.append(element3)

with open('common/coefficients.pkl', 'wb') as filename:
    pickle.dump(generated_list_pl, filename)
    pickle.dump(generated_list_dist, filename)
    pickle.dump(generated_list_done, filename)
