import flask
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = '/uploads'
app.config['MAX_CONTENT_PATH'] = 10000

@app.route('/', methods = ['GET', 'POST'])
def main():
    if request.method == 'POST':
        f = request.files['f']
        f.save(secure_filename(f.filename))
        return flask.render_template('finished.html')
    else:
        return flask.render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
