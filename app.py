from classification.learners.classification_learner import ClassificationLearner
import json
import os
import shutil
import argparse
import pandas as pd
from utils.learner_util import evaluate
import fastapi

from fastapi import FastAPI, Request, APIRouter
from fastapi.responses import HTMLResponse
import uvicorn, time
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)
learner = ClassificationLearner.from_serialization(serialization_dir="models/SentimentAttentionCLF-char")
@app.post("/infer")
async def infer(body:dict):
    
    prediction = learner.predict(body["text"])
    prediction["att_weight"] = None
    print(prediction)
    return prediction


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0")