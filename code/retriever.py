"""
Builds a FAISS vector store from the support corpus using HuggingFace
sentence-transformers embeddings (all-MiniLM-L6-v2, free, runs locally).

Provides a retriever that can filter by domain for targeted search.
"""

from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from corpus_loader import load_corpus, REPO_ROOT

INDEX_DIR = REPO_ROOT / "data" / "index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_embeddings_cache = None


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings_cache
    if _embeddings_cache is None:
        _embeddings_cache = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
        )
    return _embeddings_cache


def build_vector_store(force_rebuild: bool = False) -> FAISS:
    """Build or load the FAISS vector store."""
    embeddings = get_embeddings()

    if INDEX_DIR.exists() and not force_rebuild:
        try:
            return FAISS.load_local(
                str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True
            )
        except Exception:
            pass

    print("Building vector store from corpus...")
    documents = load_corpus()
    print(f"Loaded {len(documents)} raw documents")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    store = FAISS.from_documents(chunks, embeddings)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    store.save_local(str(INDEX_DIR))
    print(f"Vector store saved to {INDEX_DIR}")

    return store


def retrieve_relevant_docs(
    store: FAISS,
    query: str,
    domain: Optional[str] = None,
    k: int = 6,
) -> List[Document]:
    """Retrieve the most relevant documents for a query.

    Uses domain-filtered search first, then supplements with global results
    if needed.
    """
    search_kwargs = {"k": k}
    if domain and domain != "None":
        search_kwargs["filter"] = {"domain": domain}

    docs = store.similarity_search(query, **search_kwargs)

    # Supplement with global results if domain-filtered returned few results
    if len(docs) < 3 and domain and domain != "None":
        global_docs = store.similarity_search(query, k=k)
        seen_sources = {d.metadata.get("source") for d in docs}
        for d in global_docs:
            if d.metadata.get("source") not in seen_sources:
                docs.append(d)
                if len(docs) >= k:
                    break

    return docs


if __name__ == "__main__":
    store = build_vector_store()
    docs = retrieve_relevant_docs(store, "How do I reset my password?", domain="HackerRank")
    print(f"\nRetrieved {len(docs)} docs:")
    for d in docs:
        print(f"  [{d.metadata['domain']}] {d.metadata['source'][:60]}: {d.page_content[:100]}...")
