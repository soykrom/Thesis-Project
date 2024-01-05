import pandas as pd
import os

# Assuming you have a pandas DataFrame named states_df
states_df = pd.read_csv('common/transitions.csv')

# Sample 5 random elements from 'Previous State' column
previous_states = states_df['Previous State'].apply(lambda x: x.strip('[]').split(',')).tolist()
new_states = states_df['New State'].apply(lambda x: x.strip('[]').split(',')).tolist()

# Define indexes to remove (1st, 2nd, and 5th)
indexes_to_remove = [0, 1, 4]

# Convert string representations to lists of numbers
prev_state = [
    [float(value) for idx, value in enumerate(element) if idx not in indexes_to_remove]
    for element in previous_states
]

new_state = [
    [float(value) for idx, value in enumerate(element) if idx not in indexes_to_remove]
    for element in new_states
]

state_transitions = []
for element in zip(prev_state, new_state):
    state_transitions.append(list(element))

states_df = pd.DataFrame(state_transitions, columns=['Previous State', 'New State'])
states_df.to_csv(os.path.abspath('common/transitions.csv'), mode='w', index=False)

