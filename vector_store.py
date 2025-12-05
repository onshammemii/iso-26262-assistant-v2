"""
Vector Store module for ISO 26262 Safety Assistant
SIMPLE - Just load the vector store
"""

import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


def load_or_create_vector_store(persist_directory: str = "data/vector_store"):
    """
    Load existing vector store from disk

    Args:
        persist_directory: Path where vector store is saved

    Returns:
        FAISS vector store instance
    """
    print(f"\n📦 Loading vector store from {persist_directory}...")

    if not os.path.exists(persist_directory):
        print(f"❌ Vector store not found at {persist_directory}")
        print(f"📁 Please make sure you have the vector store files (index.faiss, index.pkl)")
        return None

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # FIXED: Remove allow_dangerous_deserialization parameter
        vector_store = FAISS.load_local(
            persist_directory,
            embeddings
        )
        print(f"✅ Vector store loaded successfully")
        return vector_store

    except Exception as e:
        print(f"❌ Error loading vector store: {e}")
        return None


def search_documents(vector_store, query: str, k: int = 5):
    """
    Search for relevant documents in the vector store
    """
    try:
        results = vector_store.similarity_search(query, k=k)
        return results
    except Exception as e:
        print(f"❌ Error searching documents: {e}")
        return []


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Testing Vector Store")
    print("="*60 + "\n")

    store = load_or_create_vector_store()

    if store:
        print("\n🧪 Testing search functionality...")
        results = search_documents(store, "What is ASIL?", k=3)

        print(f"\n📋 Found {len(results)} results for 'What is ASIL?':\n")
        for i, doc in enumerate(results, 1):
            print(f"--- Result {i} ---")
            print(f"Content: {doc.page_content[:150]}...")
            print(f"Source: {doc.metadata.get('source', 'Unknown')}")
            print()

        print("✅ Vector store is working correctly!")
    else:
        print("❌ Failed to load vector store")