from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import io
import os
import numpy as np
import tensorflow as tf
from PIL import Image
import uvicorn
from tensorflow.keras.models import load_model

app = FastAPI()


os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"



app = FastAPI()

model = load_model("model_3layer.hdf5")


class_names = ['Flowers', 'Non_Flowers']

@app.post("/classify/")
async def classify_image(file: UploadFile):

    image = await file.read()
    image = Image.open(io.BytesIO(image))
    image = image.resize((100, 100))
    image_array = np.array(image)
    image_array = image_array / 255.0  
    image_array = np.expand_dims(image_array, axis=0)  

   
    predictions = model.predict(image_array)
    predicted_class = predictions[0][0] > 0.7
    predicted_class_name = class_names[predicted_class]

    return JSONResponse(content={"predicted_class": predicted_class_name})
