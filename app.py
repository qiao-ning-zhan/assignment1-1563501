import streamlit as st
import os
from typing import List, Dict, Tuple
import requests
from docx import Document
from dotenv import load_dotenv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

# OpenAI API setup
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

class OpenAIInterface:
    def __init__(self):
        self.api_key = OPENAI_API_KEY
        self.base_url = "https://api.openai.com/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def _send_request(self, endpoint: str, payload: Dict) -> Dict:
        response = requests.post(f"{self.base_url}/{endpoint}", headers=self.headers, json=payload)
        response.raise_for_status()
        return response.json()

    def generate_chat_response(self, conversation: List[Dict[str, str]]) -> str:
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": conversation
        }
        response = self._send_request("chat/completions", payload)
        return response['choices'][0]['message']['content']

    def create_embedding(self, text: str) -> List[float]:
        payload = {
            "model": "text-embedding-3-large",
            "input": text
        }
        response = self._send_request("embeddings", payload)
        return response['data'][0]['embedding']

openai_interface = OpenAIInterface()

# Document processing functions
def chunk_document(text: str, chunk_size: int = 1000) -> List[str]:
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def read_docx(file):
    doc = Document(file)
    return " ".join([paragraph.text for paragraph in doc.paragraphs])

# Simple vector store
class SimpleVectorStore:
    def __init__(self):
        self.documents: List[str] = []
        self.embeddings: List[List[float]] = []
        self.metadata: List[Dict] = []

    def add(self, documents: List[str], metadata: List[Dict]):
        for doc, meta in zip(documents, metadata):
            embedding = openai_interface.create_embedding(doc)
            self.documents.append(doc)
            self.embeddings.append(embedding)
            self.metadata.append(meta)

    def search(self, query: str, k: int = 2) -> List[Tuple[str, float, Dict]]:
        query_embedding = openai_interface.create_embedding(query)
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        results = []
        for idx in top_k_indices:
            results.append((
                self.documents[idx],
                similarities[idx],
                self.metadata[idx]
            ))
        return results

# Initialize vector store
@st.cache_resource
def setup_vector_store():
    return SimpleVectorStore()

def preprocess_and_store_document(vector_store, document: str, doc_id: str):
    chunks = chunk_document(document)
    metadata = [{"doc_id": doc_id} for _ in chunks]
    vector_store.add(chunks, metadata)
    return len(chunks)

def query_document(vector_store, query: str, k: int = 2) -> List[Tuple[str, float, Dict]]:
    return vector_store.search(query, k)

def generate_response(context: str, question: str) -> str:
    prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    conversation = [
        {"role": "system", "content": "You are a helpful legal assistant. Provide clear and concise answers."},
        {"role": "user", "content": prompt}
    ]
    return openai_interface.generate_chat_response(conversation)

# Streamlit UI
def main():
    st.set_page_config(page_title="Legal Assistant", page_icon="⚖️", layout="wide")
    st.title("🤖 Your AI Legal Assistant")

    # Initialize session state
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = setup_vector_store()
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = {}

    col1, col2 = st.columns([2, 1])

    with col2:
        st.subheader("📄 Document Upload")
        uploaded_file = st.file_uploader("Upload a legal document", type=["txt", "docx"])
        
        if uploaded_file and uploaded_file.name not in st.session_state.uploaded_files:
            if uploaded_file.type == "text/plain":
                document = uploaded_file.getvalue().decode("utf-8")
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                document = read_docx(uploaded_file)
            else:
                st.error(f"Unsupported file type for {uploaded_file.name}. Please upload .txt or .docx files.")
                return

            with st.spinner(f"Processing {uploaded_file.name}..."):
                num_chunks = preprocess_and_store_document(st.session_state.vector_store, document, uploaded_file.name)
            st.success(f"✅ {uploaded_file.name} processed successfully! ({num_chunks} sections analyzed)")
            st.session_state.uploaded_files[uploaded_file.name] = num_chunks

        if st.session_state.uploaded_files:
            st.write("Uploaded Documents:")
            for filename, chunks in st.session_state.uploaded_files.items():
                st.write(f"- {filename} ({chunks} sections)")

    with col1:
        st.subheader("💬 Chat with Your Legal Assistant")
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        question = st.chat_input("Ask a question about the legal documents...")

        if question:
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.spinner("Searching for relevant information..."):
                results = query_document(st.session_state.vector_store, question)

            if results:
                with st.chat_message("assistant"):
                    context = " ".join([doc for doc, _, _ in results])
                    response = generate_response(context, question)
                    st.markdown(response)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})

                with st.expander("📚 View relevant document sections"):
                    for i, (doc, similarity, metadata) in enumerate(results, 1):
                        if metadata and 'doc_id' in metadata:
                            st.markdown(f"**Section {i}** (Document: {metadata['doc_id']}, Relevance: {similarity*100:.1f}%)")
                        else:
                            st.markdown(f"**Section {i}** (Relevance: {similarity*100:.1f}%)")
                        st.text(doc[:200] + "..." if len(doc) > 200 else doc)
            else:
                st.error("No relevant information found. Please try a different question or upload more documents.")

if __name__ == "__main__":
    main()