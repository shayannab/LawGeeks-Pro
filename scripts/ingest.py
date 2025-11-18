import os
import shutil
from dotenv import load_dotenv
# Import the PDF loader
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# Define paths
KNOWLEDGE_BASE_DIR = "../knowledge_base"
VECTOR_DB_DIR = "../vector_db"
ENV_PATH = ".env"

def main():
    # Load API Key
    load_dotenv(dotenv_path=ENV_PATH)
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in .env file")

    print("Loading documents from knowledge base...")
    
    # Updated to load .pdf files
    loader = DirectoryLoader(
        KNOWLEDGE_BASE_DIR,
        glob="**/*.pdf",         # Look for .pdf files
        loader_cls=PyPDFLoader  # Use the PDF loader
    )
    documents = loader.load()

    if not documents:
        print("No .pdf documents found. Check your KNOWLEDGE_BASE_DIR.")
        return

    print(f"Loaded {len(documents)} documents.")

    # Split documents into smaller chunks
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")

    # Initialize the embedding model
    print("Initializing embedding model...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)

    # Clear out the old vector database if it exists
    if os.path.exists(VECTOR_DB_DIR):
        print(f"Removing old vector database at {VECTOR_DB_DIR}...")
        shutil.rmtree(VECTOR_DB_DIR)

    # Create and persist the new vector database
    print("Creating and persisting vector database...")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB_DIR
    )

    print("Ingestion complete!")
    print(f"Vector database saved at: {VECTOR_DB_DIR}")

if __name__ == "__main__":
    main()