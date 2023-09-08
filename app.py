from flask import Flask, render_template, request
import jsonify
import requests
import os
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

#Stemming Process to clean our Data
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
app = Flask(__name__)


# Load the saved vectorizer using pickle
vectorizer_path = os.path.join(os.getcwd(), 'models', 'temp', 'tfidf_vectorizer.pkl')
with open(vectorizer_path, 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Load the saved model from the subfolder
model_path = os.path.join(os.getcwd(), 'models', 'temp', 'model.pkl')
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

stemmer = PorterStemmer()


@app.route('/',methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user input from the form
        news_text = request.form['news_text']
        user_name = request.form['user_name']
        news_title = request.form['news_title']

        #  preprocessing  steps:
        content = user_name +news_title+news_text
        cleaned_text = re.sub('[^a-zA-Z]', ' ', content)
        lowercase_text = cleaned_text.lower()
        words = lowercase_text.split()
        stemmed_words = [stemmer.stem(word) for word in words if word not in stopwords.words('english')]
        preprocessed_text = ' '.join(stemmed_words)
        input_clean = vectorizer.transform([preprocessed_text])
        prediction_value = model['model_object'].predict(input_clean)
        # Send the prediction value to the html file to show up
        if prediction_value[0] == 0:
            return render_template('index.html', prediction="✅ This news have a high probability to be true", image= "fact.png")
        else:
            return render_template('index.html', prediction="❌ This news have a high probability to be fake", image= "fake.png")
    else:
        return render_template('index.html')
if __name__ == '__main__':
    app.run(debug=True)