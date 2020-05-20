import pandas as pd
import pickle
import numpy as np

from flask import Flask, url_for, redirect, render_template, jsonify, request
from pycaret.regression import *

app = Flask(__name__)

# Global variables
# load the pycaret transormation pipline
model = load_model('models/deployment_05012020')
cols = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']


@app.route("/")
def main():
    # Return the homepage
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    """Return the predicted interest premium."""

    # get the features submitted from the form
    print(f'data: {request.form}')
    int_features = [x for x in request.form.values()]
    # print(int_features)

    # make it a numpy array
    final = np.array(int_features)
    data_unseen = pd.DataFrame([final], columns = cols)

    # run the pycaret pipeline with the trained model
    prediction = predict_model(model, data=data_unseen, round = 0)
    prediction = int(prediction.Label[0])
    print(f'Successful call of model returns: {prediction}')

    return render_template('index.html', pred=prediction)
