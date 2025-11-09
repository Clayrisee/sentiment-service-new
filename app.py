# File for code app service using fastapi

from fastapi import FastAPI
from src.model import InferenceModel
from datetime import datetime

app = FastAPI()
model = InferenceModel()

@app.get("/")
async def predict(text: str):
    result_predict = model.predict(text=text)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    result_predict["timestamp"] = timestamp
    return result_predict

# localhost/siapa_nama_anda?

# Perlu pake post method construct input json -> {"text": ""}
# /predict/text

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)