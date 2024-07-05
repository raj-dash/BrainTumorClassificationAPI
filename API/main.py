'''from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix

tl_model = load_model("my_effnet3.keras")

def make_predictions(model, X_test, y_test):
    pred = model.predict(X_test / 255)
    pred = np.argmax(pred, axis=1)
    y_test_new = np.argmax(y_test,axis=1)
    return y_test_new, pred

def plot_confusion_matrix(y_test_new, pred):
    print(classification_report(y_test_new, pred), end="\n\n")

    fig,ax=plt.subplots(1,1,figsize=(14,7))
    sns.heatmap(confusion_matrix(y_test_new,pred),ax=ax,xticklabels=labels,yticklabels=labels,annot=True,
               cmap=colors_green[::-1],alpha=0.7,linewidths=2,linecolor=colors_dark[3])
    fig.text(s='Heatmap of the Confusion Matrix',size=18,fontweight='bold',
                 fontname='monospace',color=colors_dark[1],y=0.92,x=0.28,alpha=0.8)

    plt.show()

data_path = "otherData/Testing"
labels = ['glioma', 'meningioma', 'notumor', 'pituitary']
X = []
y = []

image_size = 150

for label in labels:
  folder_path = os.path.join(data_path, label)
  for fn in tqdm(os.listdir(folder_path)):
    fnpath = os.path.join(folder_path, fn)
    img = cv2.imread(fnpath)
    img = cv2.resize(img, (image_size, image_size))
    X.append(img)
    y.append(label)

X = np.array(X)
y = np.array(y)

y_new = []
for i in y:
    y_new.append(labels.index(i))
y = y_new
y = tf.keras.utils.to_categorical(y)

# setting up the colours
colors_dark = ["#1F1F1F", "#313131", '#636363', '#AEAEAE', '#DADADA']
colors_red = ["#331313", "#582626", '#9E1717', '#D35151', '#E9B4B4']
colors_green = ['#01411C','#4B6F44','#4F7942','#74C365','#D0F0C0']

y_new, pred = make_predictions(tl_model, X, y)
plot_confusion_matrix(y_new, pred)'''


from typing import Union
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}