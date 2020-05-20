# requests are objects that flask handles (get set post, etc)
from flask import Flask, render_template, request
# scientific computing library for saving, reading, and resizing images
from scipy.misc.pilutil import imread, imresize
# for matrix math
import numpy as np
# for regular expressions, saves time dealing with string data
import re
# system level operations (like loading files)
import sys
# for reading operating system data
import os
# tell our app where our saved model is
sys.path.append(os.path.abspath("./model"))
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from load import *
# initalize our flask app
app = Flask(__name__)
# global vars for easy reusability
global model
# initialize these variables
model = init()
import base64

import base64
# decoding an image from base64 into raw representation
def convertImage(imgData1):
    imgstr = re.search(r'base64,(.*)', str(imgData1)).group(1)
    with open('output.png', 'wb') as output:
        output.write(base64.b64decode(imgstr))

def prediction(x):
    # in our computation graph
    #global graph
    # tf.compat.v1.reset_default_graph()
    # with graph.as_default():
    # perform the prediction
    print(f'data is {[x]}')
    out = model.predict(x)
    print(f'prediction is {out}')
    # print(np.argmax(out, axis=1))
    # convert the response to a string
    # response = np.argmax(out, axis=1)
    response = (out)
    return response

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    print("predict called")
    # whenever the predict method is called, we're going
    # to input the user drawn character as an image into the model
    # perform inference, and return the classification
    # get the raw data format of the image
    imgData = request.get_data()
    # encode it into a suitable format
    convertImage(imgData)

    image1 = Image.open('output.png').convert('L')
    #image1 = imread("output.png", flatten = 1)
    image2 = image1.resize((8, 8))
    # print(f'image1 is {image1}')
    # print(f'image2 is {image2}')
    image3 = np.reshape(image2, (1, -1))
    # print(image3.shape)
    # x.save('greyscale.png')
    # print(x)

    # plt.figure
    # plt.imshow(x)
    # plt.title('Original Image')

    # x = imresize(x, [28,28])
    # x = x.resize((28, 28), resample = Image.NEAREST)
    # print(x)
    # x.save('sized.png')

    print("predict called 2")
    response = prediction(image3)
    return str(response[0])
    # return str(0)


if __name__ == "__main__":
    # run the app locally on the given port
    app.run(host='localhost', port=5000)
# optional if we want to run in debugging mode
# app.run(debug=True)
