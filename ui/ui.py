import requests
import streamlit as st

# Backend base URL
backend_url = "http://api:8100/llm-rag"


def list_documents():
    response = requests.post(f"{backend_url}/list_docs")
    if response.status_code == 200:
        return response.json()
    else:
        return []


def upload_document(upload_file):
    files = {"upload_file": (upload_file.name, upload_file, upload_file.type)}
    response = requests.post(f"{backend_url}/upload_docs", files=files)
    return response.json()


def delete_document(file_name):
    response = requests.post(
        f"{backend_url}/delete_doc", params={"file_name": file_name}
    )
    return response.json()


def ask_question(question, llm_model="OpenAI"):
    response = requests.post(
        f"{backend_url}/question", json={"q": question}, params={"llm_model": llm_model}
    )
    if response.status_code == 200:
        return response.json()
    else:
        return "Failed to get an answer."


def service_status():
    response = requests.get(f"{backend_url}/service_status")
    return response.json()


# UI
st.title("Document-based Q&A System")

# Service status
status = service_status()
st.write(f"Service Status: {status}")


# List documents
st.header("Available Documents in the knowledge base")
# Check if the session state variable has changed to refresh the document list
documents = (
    list_documents()
)  # This will re-fetch the documents every time the page renders or the refresh button is clicked
if documents:
    st.write(documents)
else:
    st.write("No documents available.")

# Initialize session state variable if it doesn't exist
if "refresh_docs" not in st.session_state:
    st.session_state["refresh_docs"] = 0

# Button to refresh the document list
if st.button("Refresh Document List"):
    # Increment the session state variable to trigger a refresh
    st.session_state["refresh_docs"] += 1

# Upload document
st.header("Upload a New Document to the knowledge base")
uploaded_file = st.file_uploader("Choose a file", type=["pdf", "doc", "docx", "txt"])
if uploaded_file is not None:
    upload_response = upload_document(uploaded_file)
    st.write(upload_response)

# Delete document
st.header("Delete a Document from the knowledge base")
file_name_to_delete = st.text_input("Enter the file name to delete")
if st.button("Delete Document"):
    delete_response = delete_document(file_name_to_delete)
    st.write(delete_response)

# Ask a question
st.header("Ask a Question")
question = st.text_input("Enter your question")
llm_model_option = st.selectbox("Choose LLM Model", ["OpenAI", "LLM"])
if st.button("Get Answer"):
    # Place a placeholder for the loading message or running logo
    with st.empty():
        # Display a loading message or running logo
        st.write("Running...")

        # Call the backend to get the answer
        answer = ask_question(question, llm_model=llm_model_option)

        # Now that we have the answer, clear the loading message
        # and display the answer instead
        st.write("Answer:", answer)
