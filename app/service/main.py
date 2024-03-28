import warnings
import os
import psycopg2
import nltk
from huggingface_hub import login as hf_login
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.vectorstores.pgvector import DistanceStrategy
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_community.document_loaders.merge import MergedDataLoader
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores.pgvector import PGVector
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

# Suppress specific warnings and download necessary NLTK data
nltk.download("punkt")
hf_login(token=os.environ["HUGGINGFACE_TOKEN"], add_to_git_credential=False)
warnings.filterwarnings("ignore", category=UserWarning)

def write_file(file_path, file_contents):
    """Writes binary content to a file."""
    with open(file_path, "wb") as file:
        file.write(file_contents)

def db_connect():
    """Creates and returns a database connection and cursor."""
    conn = psycopg2.connect(
        user=os.environ["PGVECTOR_USER"],
        password=os.environ["PGVECTOR_PASSWORD"],
        host=os.environ["PGVECTOR_HOST"],
        port=os.environ["PGVECTOR_PORT"],
        database=os.environ["PGVECTOR_DB"],
    )
    cur = conn.cursor()
    return conn, cur

def list_docs_pg() -> list:
    """Lists document filenames from the database."""
    try:
        conn, cur = db_connect()
        cur.execute("SELECT file_name FROM files_table")
        files = cur.fetchall()
        return [row[0] for row in files]
    except psycopg2.Error as e:
        return f"Error retrieving documents from the database: {e}"
    finally:
        cur.close()
        conn.close()

def upload_file_pg(file):
    """Uploads a file to the database if it doesn't already exist."""
    filename = file.filename
    file_contents = file.file.read()
    try:
        conn, cur = db_connect()
        cur.execute("SELECT COUNT(*) FROM files_table WHERE file_name = %s", (filename,))
        if cur.fetchone()[0] == 0:
            cur.execute("INSERT INTO files_table (file_name, file_content) VALUES (%s, %s)", (filename, file_contents))
            conn.commit()
            return "File uploaded and inserted into the database."
        else:
            return "File with the same name already exists."
    except psycopg2.Error as e:
        conn.rollback()
        return f"Error inserting document into the database: {e}"
    finally:
        cur.close()
        conn.close()

def delete_doc_pg(file_name) -> str:
    """Deletes a document from the database by filename."""
    try:
        conn, cur = db_connect()
        cur.execute("SELECT 1 FROM files_table WHERE file_name = %s", (file_name,))
        if cur.fetchone():
            cur.execute("DELETE FROM files_table WHERE file_name = %s", (file_name,))
            conn.commit()
            return "Document deleted successfully."
        else:
            return "File does not exist."
    except psycopg2.Error as e:
        return f"Error deleting document: {e}"
    finally:
        cur.close()
        conn.close()

def question_pg(query: str, llm) -> str:
    """Answers a question based on documents in the database using a specified language model."""
    # Setup for OpenAI or HuggingFace models
    if llm == "OpenAI":
        local_llm = ChatOpenAI(model_name="gpt-4-0125-preview", temperature=0, top_p=0.95)
        instructor_embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    else:
        model_name = "google/flan-t5-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name, return_full_text=True, model_max_length=512, truncation=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=512, temperature=0, top_p=0.95, repetition_penalty=1.15, truncation=True, padding="max_length")
        local_llm = HuggingFacePipeline(pipeline=pipe)
        instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base", model_kwargs={"device": "cpu"})

    try:
        # Process documents for querying
        conn, cur = db_connect()
        cur.execute("SELECT file_name, file_content FROM files_table")
        documents = cur.fetchall()
        output_folder = "./extracted_documents/"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        for document in documents:
            filename, file_contents = document
            write_file(os.path.join(output_folder, filename), file_contents)

        # Load documents using various loaders
        loaders = [
            DirectoryLoader(output_folder, glob="./*.pdf", use_multithreading=True, silent_errors=True, loader_cls=PyPDFLoader),
            DirectoryLoader(output_folder, glob="./*.doc"),
            DirectoryLoader(output_folder, glob="./*.docx"),
            DirectoryLoader(output_folder, glob="./*.txt", loader_cls=TextLoader, use_multithreading=True, silent_errors=True)
        ]
        loader = MergedDataLoader(loaders=loaders)
        documents = loader.load()
        if not documents:
            return "No documents were found in the database. Please upload a document first and try again."

        # Split and embed documents for retrieval
        text_splitter = CharacterTextSplitter(separator="\n\n", chunk_size=2000, chunk_overlap=400, length_function=len, is_separator_regex=False)
        documents = text_splitter.split_documents(documents)
        connection_string = PGVector.connection_string_from_db_params(driver=os.environ.get("PGVECTOR_DRIVER", "psycopg2"), host=os.environ["PGVECTOR_HOST"], port=int(os.environ["PGVECTOR_PORT"]), database=os.environ["PGVECTOR_DB"], user=os.environ["PGVECTOR_USER"], password=os.environ["PGVECTOR_PASSWORD"])
        db = PGVector.from_documents(embedding=instructor_embeddings, documents=documents, collection_name="files_table", distance_strategy=DistanceStrategy.COSINE, connection_string=connection_string, pre_delete_collection=True)
        retriever = db.as_retriever(search_kwargs={"k": 3})

        # Setup and execute the question-answering chain
        instruction = """CONTEXT:/n/n If the information is not available in the retriever, reply with "I am sorry, this information is not in my knowledge base.". Disregard information of the table of content of the retriever. Here are retriever information: {context}/n Question: {question}"""
        llama_prompt = PromptTemplate(template=instruction, input_variables=["context", "question"])
        qa_chain = RetrievalQA.from_chain_type(llm=local_llm, chain_type="stuff", retriever=retriever, chain_type_kwargs={"prompt": llama_prompt}, return_source_documents=True, verbose=True)
        out = qa_chain(query)
        out["result"] = out["result"].replace(r"\"", " ")
        return out if out["result"] != "I am sorry, this information is not in my knowledge base." else {"result": out["result"]}
    except psycopg2.Error as e:
        return f"Error retrieving documents from the database: {e}"
    finally:
        cur.close()
        conn.close()