from flask import Flask, render_template,request
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

filename='model.pkl'
classifier=pickle.load(open(filename,'rb'))
bow = pickle.load(open('bow.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/result',methods=['POST'])
def predict():
	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		bow_extract=bow.transform(data).toarray()
		my_prediction = classifier.predict(bow_extract)
		return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(host='0.0.0.0',port=5000,debug=True)

'''
from flask import Flask
import numpy as np
import pickle

app = Flask(__name__)
@app.route('/')
def home():
	return "FUckYOU"

def predict():
	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		class_predict=
'''

