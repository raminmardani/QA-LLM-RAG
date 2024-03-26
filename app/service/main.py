import warnings
import os
import psycopg2
from huggingface_hub._login import _login
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.vectorstores.pgvector import DistanceStrategy
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores.pgvector import PGVector
from langchain_community.document_loaders.merge import MergedDataLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

_login(token=os.environ["HUGGINGFACE_TOKEN"], add_to_git_credential=False)
warnings.filterwarnings("ignore", category=UserWarning)


def write_file(file_path, file_contents):
    with open(file_path, "wb") as file:
        file.write(file_contents)


def list_docs_pg() -> list:
    try:
        # Connect to the database
        conn = psycopg2.connect(
            user=os.environ["PGVECTOR_USER"],
            password=os.environ["PGVECTOR_PASSWORD"],
            host=os.environ["PGVECTOR_HOST"],
            port=os.environ["PGVECTOR_PORT"],
            database=os.environ["PGVECTOR_DB"],
        )

        # Create a cursor to execute SQL commands
        cur = conn.cursor()

        # Now you can proceed with your SELECT query
        cur.execute("SELECT file_name FROM files_table")

        # Fetch all the rows
        files = cur.fetchall()

        # Close cursor and connection
        cur.close()
        conn.close()

        # Return the list of filenames
        return [row[0] for row in files]
    except psycopg2.Error as e:
        return "Error retrieving documents from the database:" + str(e)


def upload_file_pg(file):
    filename = file.filename
    file_contents = file.file.read()
    conn = psycopg2.connect(
        user=os.environ["PGVECTOR_USER"],
        password=os.environ["PGVECTOR_PASSWORD"],
        host=os.environ["PGVECTOR_HOST"],
        port=os.environ["PGVECTOR_PORT"],
        database=os.environ["PGVECTOR_DB"],
    )

    # Create a cursor to execute SQL commands
    cur = conn.cursor()

    try:
        # Check if the file already exists in the database
        cur.execute(
            "SELECT COUNT(*) FROM files_table WHERE file_name = %s", (filename,)
        )
        result = cur.fetchone()[0]

        if result == 0:
            # If the file doesn't exist, insert it into the database
            cur.execute(
                "INSERT INTO files_table (file_name, file_content) VALUES (%s, %s)",
                (filename, file_contents),
            )
            conn.commit()
            return "File uploaded and inserted into the database."
        else:
            return "File with the same name already exists."
    except psycopg2.Error as e:
        # Rollback the transaction in case of an error
        conn.rollback()
        return "Error inserting document into the database: {}".format(e)
    finally:
        # Close the cursor and the connection
        cur.close()
        conn.close()


def delete_doc_pg(file_name) -> str:
    try:
        conn = psycopg2.connect(
            user=os.environ["PGVECTOR_USER"],
            password=os.environ["PGVECTOR_PASSWORD"],
            host=os.environ["PGVECTOR_HOST"],
            port=os.environ["PGVECTOR_PORT"],
            database=os.environ["PGVECTOR_DB"],
        )
        with conn.cursor() as cur:
            # Check if the file exists
            cur.execute("SELECT 1 FROM files_table WHERE file_name = %s", (file_name,))
            if cur.fetchone():
                # File exists, delete it
                cur.execute(
                    "DELETE FROM files_table WHERE file_name = %s", (file_name,)
                )
                conn.commit()
                return "Document deleted successfully."
            else:
                return "File does not exist."
    except psycopg2.Error as e:
        return f"Error deleting document: {e}"


def question_pg(query: str, llm) -> str:
    if llm == "OpenAI":

        local_llm = ChatOpenAI(
            model_name="gpt-3.5-turbo-0301",
            temperature=0,
            top_p=0.95,
        )

        instructor_embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    else:
        model_name = "google/flan-t5-base"
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, return_full_text=True, model_max_length=512, truncation=True
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        pipe = pipeline(
            "text2text-generation",  # Specify the task as text-to-text generation
            model=model,  # Use the previously initialized model
            tokenizer=tokenizer,  # Use the previously initialized tokenizer
            max_length=512,  # Set the maximum length for generated text to 512 tokens
            temperature=0,  # Set the temperature parameter for controlling randomness (0 means deterministic)
            top_p=0.95,  # Set the top_p parameter for controlling the nucleus sampling (higher values make output more focused)
            repetition_penalty=1.15,  # Set the repetition_penalty to control the likelihood of repeated words or phrases
            truncation=True,  # Truncate the input to the model to the maximum length of the model
            padding="max_length",  # Pad the input to the model to the maximum length of the model
        )
        local_llm = HuggingFacePipeline(pipeline=pipe)
        embedding_model_name = "hkunlp/instructor-base"
        instructor_embeddings = HuggingFaceInstructEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={"device": "cpu"},
        )
    try:
        conn = psycopg2.connect(
            user=os.environ["PGVECTOR_USER"],
            password=os.environ["PGVECTOR_PASSWORD"],
            host=os.environ["PGVECTOR_HOST"],
            port=os.environ["PGVECTOR_PORT"],
            database=os.environ["PGVECTOR_DB"],
        )
        cur = conn.cursor()
        cur.execute("SELECT file_name, file_content FROM files_table")
        documents = cur.fetchall()

        # Close the cursor and the connection
        cur.close()
        conn.close()
        # Folder to save the extracted documents
        output_folder = "./extracted_documents/"
        if os.path.exists(output_folder):
            files = os.listdir(output_folder)
            for file_name in files:
                file_path = os.path.join(output_folder, file_name)
                os.remove(file_path)
        else:
            os.makedirs(output_folder)
        # Iterate through each document and save it
        for document in documents:
            filename, file_contents = document
            file_path = os.path.join(output_folder, filename)
            write_file(file_path, file_contents)

        files = os.listdir(output_folder)
        # Check if the output folder exists, if not, create it
        loader_pdf = DirectoryLoader(
            output_folder, glob="./*.pdf", use_multithreading=True, silent_errors=True
        )
        loader_doc = DirectoryLoader(
            output_folder, glob="./*.doc", use_multithreading=True, silent_errors=True
        )
        loader_docx = DirectoryLoader(
            output_folder, glob="./*.docx", use_multithreading=True, silent_errors=True
        )
        loader_txt = DirectoryLoader(
            output_folder,
            glob="./*.txt",
            loader_cls=TextLoader,
            use_multithreading=True,
            silent_errors=True,
        )

        loader = MergedDataLoader(
            loaders=[loader_pdf, loader_doc, loader_docx, loader_txt]
        )
        documents = loader.load()
        if len(documents) == 0:
            return "No documents were found in the database. Please upload a document first and try again."
        text_splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=2000,
            chunk_overlap=400,
            length_function=len,
            is_separator_regex=False,
        )

        documents = text_splitter.split_documents(documents)

        connection_string = PGVector.connection_string_from_db_params(
            driver=os.environ.get("PGVECTOR_DRIVER", "psycopg2"),
            host=os.environ["PGVECTOR_HOST"],
            port=int(os.environ["PGVECTOR_PORT"]),
            database=os.environ["PGVECTOR_DB"],
            user=os.environ["PGVECTOR_USER"],
            password=os.environ["PGVECTOR_PASSWORD"],
        )
        db = PGVector.from_documents(
            embedding=instructor_embeddings,
            documents=documents,
            collection_name="files_table",
            distance_strategy=DistanceStrategy.COSINE,
            connection_string=connection_string,
            pre_delete_collection=True,
        )
        retriever = db.as_retriever(search_kwargs={"k": 3})
        instruction = """CONTEXT:/n/n If the information is not available in the retriever, reply with "I am sorry, this information is not in my knowledge base.". Disregard information of the table of content of the retriever. Here are retriever information: {context}/n Question: {question}"""
        llama_prompt = PromptTemplate(
            template=instruction, input_variables=["context", "question"]
        )
        chain_type_kwargs = {"prompt": llama_prompt}
        qa_chain = RetrievalQA.from_chain_type(
            llm=local_llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs=chain_type_kwargs,
            return_source_documents=True,
            verbose=True,
        )
        out = qa_chain(query)
        out["result"] = out["result"].replace(r"\"", " ")
        if out["result"] == "I am sorry, this information is not in my knowledge base.":
            return out["result"]
        else:
            return out
    except psycopg2.Error as e:
        return f"Error retrieving documents from the database: {e}"
