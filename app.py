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
    doc2vec
    return render_template('index.html', context=query)


if __name__ == '__main__':
    app.run()
