from flask import Flask, render_template, request
from utils.WordRank_Keywords import WordRank_Keywords
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
    # return WordRank_Keywords(query, 0, "text", ['a'])
    return render_template('index.html', context=query)


if __name__ == '__main__':
    app.run()
