from fastapi import FastAPI, UploadFile
from uuid import uuid4
from intent_detection import IntentDetector
from fastapi.middleware.cors import CORSMiddleware
import torch
from os import remove
from data import TextQuery, MongoDatabase, TextDatabase
import time

# create the objects
app = FastAPI()
database = TextDatabase("logs/")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
intent_detector = IntentDetector(device)

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/intent_detection/audio_upload")
async def detect_intent_from_audio(audio_file: UploadFile, domain: str):
    start  = time.time_ns()
    contents = await audio_file.read()
    file_path = "audio_files/" + str(uuid4()) + ".wav"
    with open(file_path, 'wb') as f:
        f.write(contents)
        f.close()

    transcript, intent = intent_detector.get_intent_from_audio(file_path, domain)    
    remove(file_path)

    database.db_insert_query(transcript, domain, intent)
    print("Time taken", time.time_ns() - start)
    return {
        "transcript": transcript,
        "intent": intent
    }


@app.post("/intent_detection/detect_intent")
async def detect_intent_from_text(query_text: TextQuery, domain: str):
    start  =time.time_ns()

    intent = intent_detector.get_intent_from_text(query_text.query_text, domain)
    database.db_insert_query(query_text.query_text, domain, intent)
    print("Time taken", time.time_ns() - start)

    return {
        "intent": intent
    }
