from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
import os, cv2
from PIL import Image

app = FastAPI()

img_path = "../images"
if not os.path.exists(img_path):
    os.mkdir(img_path)

tl_model = tf.keras.models.load_model("my_effnet3.keras")
labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
def img_pred(model, image_filepath):
    img = Image.open(image_filepath)
    opencvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img = cv2.resize(opencvImage,(150,150))
    img = img.reshape(1,150,150,3)
    p = model.predict(img / 255)
    pos = np.argmax(p,axis=1)[0]
    prob = round(p[0][pos] * 100, 2)
    return pos, prob

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Prediction(BaseModel):
    prediction: str

@app.post('/uploadfile/')
async def create_file_upload(image: UploadFile = File(...)):

    data = await image.read()
    file_path = f"../images/{image.filename}"
    with open(file_path, "wb") as f:
        f.write(data)
    
    prediction, probability = img_pred(tl_model, file_path)
    print(prediction)
    
    
    return {"prediction" : labels[prediction], 
            "probability" : probability}