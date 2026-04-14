from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import shutil

from rag_pipleine import MedicalAGPipeline

app = FastAPI()
pipeline = MedicalAGPipeline()

@app.get("/")
def home():
    return {"message": "API is running"}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        file_path = f"temp_{file.filename}"

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print("File saved at:", file_path)

        pipeline.load_pdf(file_path)

        print("PDF processed successfully")

        return {"message": "PDF uploaded and processed"}

    except Exception as e:
        print("ERROR:", str(e))
        return {"error": str(e)}

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_question(req: QuestionRequest):
    return pipeline.ask(req.question)

@app.post("/ask-stream")
def ask_stream(req: QuestionRequest):

    def generate():
        for chunk in pipeline.ask_stream(req.question):
            yield chunk

    return StreamingResponse(generate(), media_type="text/plain")

@app.post("/smart_ask")
def smart_ask_(req: QuestionRequest):
    return pipeline.smart_ask(req.question)

@app.post("/test-retrieval")
def test_retrieval(req: QuestionRequest):
    return pipeline.compare_retrieval(req.question)

@app.get("/summary")
def summary():
    return {"summary": pipeline.summarize()}

@app.get("/abnormals")
def abnormals():
    return pipeline.detect_abnormals()

@app.post("/clear")
def clear():
    pipeline.clear_history()
    return {"message": "Chat history cleared"}
