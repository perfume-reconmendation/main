from flask import Flask, render_template, request
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
    print("요청: ", request.form.get('data'))
    # params = json.loads(request.json, encoding='utf-8')
    # print(params)
    # str = ""
    # for key in params.keys():
    # str += f'key: {key}, value: {params[key]}'

    return request.form.get('data')


if __name__ == '__main__':
    app.run()
