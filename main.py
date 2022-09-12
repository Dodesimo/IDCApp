import flask
import torchvision.transforms
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import torch
import pandas as pd
from torch.utils.data import Dataset
from skimage import io
from skimage.transform import resize
import os
from torch import nn
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.c1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=(5, 5), stride=5)
        self.r1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.c2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(2, 2), stride=1)
        self.r2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=1)
        self.lin1 = nn.Linear(2 * 2 * 20, 150)
        self.lin2 = nn.Linear(150, 2)
        self.relu = nn.ReLU()
        self.logsoftmax = nn.LogSoftmax(dim=0)

    def forward(self, x):
        x = self.c1(x)
        x = self.r1(x)
        x = self.maxpool1(x)
        x = self.c2(x)
        x = self.r2(x)
        x = self.maxpool2(x)
        x = torch.flatten(x)
        x = self.lin1(x)
        x = self.lin2(x)
        x = self.logsoftmax(x)
        return torch.exp(x)


model = Net()
model.load_state_dict(torch.load('finalmodel.pt'))

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = '/Users/devam/PycharmProjects/IDCApp/uploads'
app.config['MAX_CONTENT_PATH'] = 10000


@app.route('/', methods=["GET", "POST"])
def main():
    if request.method == 'POST':
        f = request.files['file']
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))

        image = io.imread(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
        image = resize(image, (50, 50))

        tensorize = torchvision.transforms.ToTensor()
        image = tensorize(image)
        image = torch.from_numpy(np.asarray(image)).float()

        yhat = model(image)
        print(yhat)
        _, label = torch.max(yhat, 0)
        print(label)

        if label == 0:
            return redirect(url_for('negative'))
        elif label == 1:
            return redirect(url_for('positive'))

    else:
        return flask.render_template('index.html')


@app.route('/negative')
def negative():
    return flask.render_template('negative.html')


@app.route('/positive')
def positive():
    return flask.render_template('positive.html')


if __name__ == '__main__':
    app.run(debug=True)
