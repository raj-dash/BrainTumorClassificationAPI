from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post('/uploadfile/')
async def create_file_upload(image: UploadFile = File(...)):

    data = await image.read()
    save_to = f"../images/{image.filename}"
    with open(save_to, "wb") as f:
        f.write(data)
    
    return {"images" : image.filename}