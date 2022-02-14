from fastapi import FastAPI, UploadFile, File
from uuid import uuid4
from asr import speech_to_text
from intent_detection import detect_indent
from pymongo import MongoClient
from secrets import MONGO_URI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# client = MongoClient(MONGO_URI)
# db = client['BTP_Project']
# collection = db['queries']

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

def db_insert_query(query_text: str, intent_detected: str):
    return
    return collection.insert_one({
        "query_text": query_text,
        "intent_detected": intent_detected
    })


@app.post("/audio_upload")
async def detect_intent_from_audio(audio_file: UploadFile):
    contents = await audio_file.read()
    file_path = "audio_files/" + str(uuid4()) + ".wav"
    with open(file_path, 'wb') as f:
        f.write(contents)
        f.close()

    transcript = speech_to_text(file_path)
    intent = detect_indent(transcript)
    db_insert_query(query_text=transcript, intent_detected=intent)
    return {
        "transcript": transcript,
        "intent": intent
    }


class TextQuery(BaseModel):
    query_text: str

@app.post("/detect_intent")
async def detect_intent_from_text(query_text: TextQuery):
    intent = detect_indent(query_text.query_text)
    db_insert_query(query_text=query_text.query_text, intent_detected=intent)
    return {
        "intent": intent
    }
