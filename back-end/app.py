from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import re
import string
import pickle
import io

app = Flask(__name__)
CORS(app)

# Load the model
with open('model.pickle_rf_2_2024_06', 'rb') as f:
    model = pickle.load(f)

# Load stop words
f_stopWords = io.open("StopWords_425.txt", mode="r", encoding="utf-16")
sw = [x.split()[0] for x in f_stopWords]

# Load vocabulary
vocab = pd.read_csv('vocabulary.txt', header=None)
tokens = vocab[0].tolist()

def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text

def preprocessing(text):
    data = pd.DataFrame([text], columns=['Post'])
    data["Post"] = data["Post"].apply(lambda x: " ".join(x.lower() for x in x.split()))
    data["Post"] = data['Post'].apply(lambda x: " ".join(re.sub(r'^https?:\/\/.*[\r\n]*', '', x, flags=re.MULTILINE) for x in x.split()))
    data["Post"] = data["Post"].apply(remove_punctuations)
    data["Post"] = data['Post'].str.replace('\d+', '', regex=True)
    data["Post"] = data["Post"].apply(lambda x: " ".join(x for x in x.split() if x not in sw))
    return data["Post"]

def vectorizer(ds, vocabulary):
    vectorized_lst = []
    
    for sentence in ds:
        sentence_lst = np.zeros(len(vocabulary))
        
        for i in range(len(vocabulary)):
            if vocabulary[i] in sentence.split():
                sentence_lst[i] = 1
                
        vectorized_lst.append(sentence_lst)
        
    return np.asarray(vectorized_lst, dtype=np.float32)

def get_prediction(vectorized_text):
    prediction = model.predict(vectorized_text)
    return 'true' if prediction == 1 else 'false'


@app.route('/', methods=['POST'])
def index():
    return jsonify({'res': 'Back-end started'})

@app.route('/news', methods=['POST'])
def get_news():
    data = request.get_json()
    text = data.get('news', '')
    if text:
        preprocessed_txt = preprocessing(text)
        vectorized_txt = vectorizer(preprocessed_txt, tokens)
        prediction = get_prediction(vectorized_txt)
        return jsonify({'res': prediction, 'news': text}), 200
    else:
        return jsonify({'error': 'No text provided'}), 400

    # news =request.args.get('news')
    # messege = 'Hello, {}!'.format(news)
    # if news:  # Check if news is provided
    #     return jsonify({'res': True})
    # else:
    #     return jsonify({'res': False})

if __name__ == '__main__':
    app.run(debug=True)