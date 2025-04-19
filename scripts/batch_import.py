import os
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def main():
    # Initialize embedding model
    print("Initializing embedding model...")
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    except Exception as e:
        print(f"Failed to initialize embedding model: {e}")
        return

    # Path configurations
    PDF_FOLDER = "../data/"
    INDEX_FOLDER = "../faiss_index/"
    
    # Create output directory if it doesn't exist
    os.makedirs(INDEX_FOLDER, exist_ok=True)

    # Text splitter configuration
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]  # Explicit separators for better splitting
    )

    # Process PDF files
    all_docs = []
    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]
    
    if not pdf_files:
        print(f"No PDF files found in {PDF_FOLDER}")
        return

    print(f"Found {len(pdf_files)} PDF files to process")

    for filename in pdf_files:
        pdf_path = os.path.join(PDF_FOLDER, filename)
        print(f"\nProcessing: {filename}...")
        
        try:
            start_time = time.time()
            
            # Load and split PDF
            loader = PyPDFLoader(pdf_path)
            docs = loader.load_and_split(text_splitter)
            all_docs.extend(docs)
            
            processing_time = time.time() - start_time
            print(f"  → Processed {len(docs)} chunks in {processing_time:.2f} seconds")
            
        except Exception as e:
            print(f"  → Error processing {filename}: {e}")
            continue

    if not all_docs:
        print("No documents were processed successfully")
        return

    print(f"\nTotal chunks processed: {len(all_docs)}")

    # Create and save FAISS index
    print("\nCreating FAISS vector store...")
    try:
        start_time = time.time()
        db = FAISS.from_documents(all_docs, embedding_model)
        db.save_local(INDEX_FOLDER)
        
        index_time = time.time() - start_time
        print(f"FAISS index created and saved at: {INDEX_FOLDER}")
        print(f"Indexing took {index_time:.2f} seconds")
        
    except Exception as e:
        print(f"Failed to create FAISS index: {e}")

if __name__ == "__main__":
    main()