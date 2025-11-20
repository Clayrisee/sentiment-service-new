# File for code app service using fastapi

from fastapi import FastAPI
from src.model import InferenceModel
from datetime import datetime
from src.dynamo_db import UncertaintyDynamoDB
from loguru import logger
from mangum import Mangum

logger.info("Start Sentiment Inference Service")
app = FastAPI()
handler = Mangum(app=app)

logger.info("Starting Inference Model")
model = InferenceModel()
uncertainty_db = UncertaintyDynamoDB(table_name='uncertainty-table')


@app.get("/")
async def predict(text: str):
    logger.info(f"Running Prediction Process. Input: {text}")
    result_predict = model.predict(text=text)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    result_predict["timestamp"] = timestamp
    logger.info(f"Prediction Success. Result: {result_predict}")
    uncertainty_db.write_event(
        data=result_predict
    )
    return result_predict

# localhost/siapa_nama_anda?

# Perlu pake post method construct input json -> {"text": ""}
# /predict/text

if __name__ == "__main__":
    import uvicorn
    logger.info("Running Main App")
    uvicorn.run(app, host="0.0.0.0", port=8000)