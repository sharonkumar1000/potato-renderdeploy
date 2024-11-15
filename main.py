# from fastapi import FastAPI,File,UploadFile
# import uvicorn
# import numpy as np
# from io import BytesIO
# from PIL import Image
#
# app = FastAPI()
#
# @app.get("/ping")
#
# async def ping():
#     return "Hello, Iam alive"
#
# @app.post("/predict")
#
# def read_file_as_image(data)-> np.ndarray:
#     image = np.array(Image.open(BytesIO(data)))
#     return image
#
# async def predict(
#         file:UploadFile = File(...)
# ):
#     image = read_file_as_image(await file.read())
#     pass
#
# if __name__ == "__main__":
#     uvicorn.run(app,host = 'localhost',port = 8000)



from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()


origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model("1.h5")
CLASSES_NAMES = ["Early Binding","Late Binding","Healthy"]

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image,0)
    predictions = MODEL.predict(img_batch)
    # You can add your model prediction logic here
    # pass
    ind = np.argmax(predictions[0])
    print(ind)
    predicted_class = CLASSES_NAMES[ind]
    confidence = np.max(predictions[0])
    return {"message": predicted_class,"confidence":float(confidence)}

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
