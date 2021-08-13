from flask import Flask, render_template, request
from flask_cors import CORS

from utils.Classifier import classifier

from utils.bert_sim import BERT_recommendations
from utils.d2v_sim import doc2vec
from utils.word2vec_similarity import word2vec_similarity
from utils.highlighter import keyword_highlighter

import os
import json
import nltk

app = Flask(__name__)
CORS(app)


@app.route('/')
def index():
    # return 'Hello World'5
    return render_template('public/home.html')


@app.route('/form')
def form():
    return render_template('public/index.html')


@app.route('/result', methods=['POST'])
def result():
    query = request.form.get('data')
    print("query:", query)

    label = classifier(query)

    BERT_recommendations(query)
    word2vec_similarity(query, label)
    return render_template('index.html', context=query)


@app.route('/infer/class', methods=['POST'])
def infer_class():
    query = request.get_json()['query']
    return str(classifier(query))


@app.route('/infer/similar/bert', methods=['POST'])
def infer_similar_bert():
    query = request.get_json()['query']
    label = request.get_json()['label']
    result = BERT_recommendations(query, label)
    return json.dumps(result)

@app.route('/infer/similar/doc2vec', methods=['POST'])
def infer_similar_doc2vec():
    nltk.download('averaged_perceptron_tagger')
    nltk.download('stopwords')
    nltk.download('wordnet')

    query = request.get_json()['query']
    label = request.get_json()['label']
    result = doc2vec(query, label)
    return json.dumps(result)

@app.route('/infer/similar/word2vec', methods=['POST'])
def infer_similar_word2vec():
    nltk.download('wordnet')
    nltk.download('stopwords')
    nltk.download('stopwords')

    query = request.get_json()['query']
    label = request.get_json()['label']
    result = word2vec_similarity(query, label)
    return result

@app.route('/infer/highlight', methods=['POST'])
def infer_keyword():
    query = request.get_json()['query']
    top3 = request.get_json()['top3']
    model_path = 'model/w2v_10window'
    return keyword_highlighter(query, top3, model_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 8080)))

