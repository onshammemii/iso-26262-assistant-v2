"""
REBUILD Vector Store - Windows Compatible
Uses pypdf directly, bypasses langchain_community issues
"""

import os
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pypdf import PdfReader


def load_pdfs_from_folder(pdf_directory: str):
    """Load PDFs using pypdf directly (no langchain_community issues)"""
    documents = []

    for filename in sorted(os.listdir(pdf_directory)):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(pdf_directory, filename)
            print(f"  Loading: {filename}")
            try:
                reader = PdfReader(pdf_path)
                text = ""
                for page_num, page in enumerate(reader.pages):
                    text += page.extract_text()

                # Create a document for this PDF
                doc = Document(
                    page_content=text,
                    metadata={
                        "source": filename,
                        "pages": len(reader.pages)
                    }
                )
                documents.append(doc)
                print(f"    ✅ {len(reader.pages)} pages")
            except Exception as e:
                print(f"    ❌ Error: {e}")

    return documents


def rebuild_vector_store():
    """Rebuild vector store from PDFs"""

    print("\n" + "="*60)
    print("🔨 REBUILDING VECTOR STORE (Windows Safe)")
    print("="*60 + "\n")

    data_folder = "data/raw"
    persist_directory = "data/vector_store"

    # Delete old one
    if os.path.exists(persist_directory):
        print(f"🗑️  Deleting old vector store...")
        import shutil
        shutil.rmtree(persist_directory)

    # Load PDFs with pypdf
    print(f"\n📄 Loading PDFs from {data_folder}...")
    documents = load_pdfs_from_folder(data_folder)

    if not documents:
        print("❌ No PDFs loaded!")
        return

    print(f"\n✅ Total PDFs: {len(documents)}")

    # Split documents
    print(f"\n✂️  Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "]
    )

    chunks = text_splitter.split_documents(documents)
    print(f"✅ Total chunks: {len(chunks)}")

    # Create embeddings
    print(f"\n🧠 Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    print(f"✅ Embeddings model loaded")

    # Create vector store
    print(f"\n🗂️  Building FAISS index...")
    vector_store = FAISS.from_documents(chunks, embeddings)
    print(f"✅ Index created with {len(chunks)} chunks")

    # Save it
    print(f"\n💾 Saving to {persist_directory}...")
    os.makedirs(persist_directory, exist_ok=True)
    vector_store.save_local(persist_directory)
    print(f"✅ Saved successfully!")

    print("\n" + "="*60)
    print("✅ VECTOR STORE REBUILT SUCCESSFULLY!")
    print("="*60 + "\n")


if __name__ == "__main__":
    rebuild_vector_store()