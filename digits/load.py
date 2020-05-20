from tensorflow import keras
import tensorflow as tf

def initCNN():
  json_file = open('model_cnn.json','r')
  loaded_model_json = json_file.read()
  json_file.close()
  #tf.compat.v1.disable_eager_execution()
  loaded_model = keras.models.model_from_json(loaded_model_json)
  #load weights into new model
  loaded_model.load_weights("model_cnn_weights.h5")
  print("Loaded Model from disk")
  #compile and evaluate loaded model
  loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
  graph = tf.compat.v1.get_default_graph()
  return loaded_model,graph

def initNN():
  json_file = open('model_nn.json','r')
  loaded_model_json = json_file.read()
  json_file.close()
  #tf.compat.v1.disable_eager_execution()
  loaded_model = keras.models.model_from_json(loaded_model_json)
  #load weights into new model
  loaded_model.load_weights("model_nn_weights.h5")
  print("Loaded Model from disk")
  #compile and evaluate loaded model
  loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
  graph = tf.compat.v1.get_default_graph()
  return loaded_model,graph