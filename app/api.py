import warnings

warnings.filterwarnings("ignore")
from service import main
from fastapi import FastAPI, HTTPException, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic
import datetime
from pydantic import BaseModel
from service import main
from typing import Optional

################################################## FASTAPI ########################################
app = FastAPI(docs_url="/llm-rag/docs", openapi_url="/llm-rag/openapi.json")


class Item(BaseModel):
    q: str = "Ask a question from your document..."


security = HTTPBasic()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/llm-rag/list_docs")
async def question():
    return main.list_docs_pg()


@app.post("/llm-rag/upload_docs")
async def create_upload_file(upload_file: UploadFile = File(...)):
    if upload_file.content_type not in [
        "application/pdf",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/plain",
    ]:
        raise HTTPException(
            400,
            detail="Invalid document type. Only PDF, Doc, Docx, and txt files are allowed.",
        )
    else:
        return main.upload_file_pg(upload_file)


@app.post("/llm-rag/delete_doc")
async def question(file_name: Optional[str] = "document1.pdf"):
    if file_name is None:
        return {"error": "Please provide a file name."}
    return main.delete_doc_pg(file_name)


@app.post("/llm-rag/question")
async def question_pg(
    query: Item, llm_model: str = Query("OpenAI", enum=["OpenAI", "LLM"])
):
    return main.question_pg(query.q, llm_model)


@app.get("/llm-rag/service_status")
async def service_status():
    return {"Status: OK, Date:{}".format(datetime.datetime.now())}
