from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import itertools
# Load the Random Forest CLassifier model
filename = 'laon-approval-prediction-dt-model.pkl'
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features=[int(x) for x in request.form.values()]
    final_features=[np.array(features)]
    my_prediction = classifier.predict(final_features)   
    if my_prediction:
        prediction_text="Your loan can be apporved with the provided details"
    else:
        prediction_text="Your loan may not be approved with provided details"

    return render_template('index.html', prediction_text=prediction_text)

if __name__ == '__main__':
	app.run(debug=True)