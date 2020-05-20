import pickle

def init():
  # Load from file
  with open('model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

  print("Loaded Model from disk")
  return loaded_model