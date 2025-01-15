import sys
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import os
from src.constant import *


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocess import preprocess_img
from utils import list_folder_names, predict

model_dir = os.path.join(os.getcwd(), MODEL_DIR_PATH, 'jutemodel.keras')
class_names = list_folder_names(DATA_FOLDER_PATH=DATA_FOLDER_PATH)

app = FastAPI()

class TakePath(BaseModel):
    file_path: str


@app.get('/home')
def get_home():
    return {"message": "Welcome to the home page"}

@app.post('/predict_path')
async def predict_path(data: TakePath):
    try:
        file_path = os.path.normpath(data.file_path)
        #file_path = file_path.replace(r'\\', r'/')
        # check if the file path exists
        if not os.path.exists(file_path):
            raise HTTPException(status_code = 400, detail = f'{file_path} not exist')
        
        # check if the file is accessible and readable
        if not os.access(file_path, os.R_OK):
            raise HTTPException(status_code = 400, detail = f'{file_path} is not accessible or readable')
        
        # check for valid image extension
        if not file_path.lower().endswith(('jpg', 'jpeg', 'png')):
            raise HTTPException(status_code = 400, detail='File is not a valid image format')
        
        img_arr = preprocess_img(input_data= file_path, target_size=IMAGE_SIZE)
        
        result = predict(
            model_path=model_dir,
            img_arr=img_arr,
            class_names=class_names
        )

        return (
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(result[0], result[1])
    )
    except Exception as e:
        return {'Error' : str(e)}
