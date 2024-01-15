import numpy as np
import pandas

states_df = pandas.read_csv('common/transitions.csv')

previous_states_df = states_df['Previous State'].apply(lambda x: x.strip('[]').split(','))

min_vec = [10, 10, 10, 10]
max_vec = [0, 0, 0, 0]


for state in previous_states_df:
    state = np.array(state, dtype=float)

    min_vec = np.minimum(state, min_vec)
    max_vec = np.maximum(state, max_vec)

print(f"Minimum: {min_vec}\nMaximum: {max_vec}")
