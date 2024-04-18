import json
import pickle

import pandas as pd
from flask import Flask, jsonify, request, render_template, url_for, redirect

app = Flask(__name__)

pipeline = pickle.load(open('pipelines/latest_pipeline.pkl', 'rb'))

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/about/', methods=['GET'])
def about():
    return render_template('about.html')

@app.route('/create/', methods=('GET', 'POST'))
def create():
    if request.method == 'POST':
        carat = request.form['carat']
        cut = request.form['cut']
        color = request.form['color']
        clarity = request.form['clarity']
        depth = request.form['depth']
        table = request.form['table']
        x = request.form['x']
        y = request.form['y']
        z = request.form['z']

        data = pd.DataFrame(
            data=[[carat, cut, color, clarity, depth, table, x, y, z]],
            columns=['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z']
        )

        predicted = pipeline.predict(data)[0]

        return render_template('evaluation.html', predicted=predicted)

    return render_template('create.html')


if __name__ == '__main__':
    app.run(port=5000)
