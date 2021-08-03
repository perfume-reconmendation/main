from flask import Flask, render_template, request

from utils.Classifier import classifier

from utils.bert_sim import BERT_recommendations
from utils.d2v_sim import doc2vec
from utils.w2v_sim import word2vec_similarity

import json
app = Flask(__name__)


@app.route('/')
def index():
    # return 'Hello World'
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
    result = BERT_recommendations(query)
    return json.dumps(result)

@app.route('/infer/similar/doc2vec', methods=['POST'])
def infer_similar_doc2vec():
    query = request.get_json()['query']
    label = request.get_json()['label']
    result = doc2vec(query)
    return json.dumps(result)

@app.route('/infer/similar/word2vec', methods=['POST'])
def infer_similar_word2vec():
    query = request.get_json()['query']
    label = request.get_json()['label']
    result = word2vec_similarity(query, label)
    return json.dumps(result)


if __name__ == '__main__':
    app.run()
