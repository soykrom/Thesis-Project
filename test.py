import pickle

with open('common/coefficients.pkl', 'rb') as file:
    co_list_pl = pickle.load(file)
    co_list_dist = pickle.load(file)
    co_list_done = pickle.load(file)

# Create a new list
combined_list = [list(x) for x in zip(co_list_pl, co_list_dist, co_list_done)]

# Element to search for
search_element = [1.5, 2.56, 0.74]

# Search for the specific element in the combined list
found_index = next((i for i, sublist in enumerate(combined_list) if sublist == search_element), None)

# Print the result
if found_index is not None:
    print(f"Element {search_element} found at index {found_index}.")
else:
    print(f"Element {search_element} not found in the combined list.")
