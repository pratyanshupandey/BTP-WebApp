from fastapi import FastAPI, UploadFile, File
from uuid import uuid4
from asr import speech_to_text
from intent_detection import detect_indent

app = FastAPI()


@app.post("/audio_upload")
async def detect_intent_from_audio(audio_file: UploadFile):
    contents = await audio_file.read()
    file_path = "audio_files/" + str(uuid4()) + ".wav"
    with open(file_path, 'wb') as f:
        f.write(contents)
        f.close()

    transcript = speech_to_text(file_path)
    intent = detect_indent(transcript)

    return {
        "transcript": transcript,
        "intent": intent
    }


@app.post("/detect_intent")
async def detect_intent_from_text(query_text: str):
    intent = detect_indent(query_text)
    return {
        "intent": intent
    }
