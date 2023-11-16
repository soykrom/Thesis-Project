import pickle

with open('common/rewards.pkl', 'rb') as file:
    element = pickle.load(file)
    print(element)
