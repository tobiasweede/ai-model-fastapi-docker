import io

from fastapi import FastAPI, UploadFile
from PIL import Image

from predict import model_predict

app = FastAPI()


@app.get("/")
def read_root():
    return "access /docs to see the post method!"


@app.post("/predict")
def predict_image(file: UploadFile):
    content = file.file.read()
    img = Image.open(io.BytesIO(content))
    result = model_predict(img)
    return {"answer": result}
