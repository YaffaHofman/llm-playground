from pathlib import Path
from typing import List, Dict


def load_and_chunk_document( 
    
    file_path: str,
    chunk_size: int = 1200,
    chunk_overlap: int = 200
) -> List[Dict]:
    """
    טוען קובץ טקסט/Markdown ומחלק אותו ל-chunks.
    מחזיר רשימת chunks עם metadata.
    """

    path = Path(file_path)
    text = path.read_text(encoding="utf-8")

    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "source": str(path),
                    "file_name": path.name,
                    "chunk_index": chunk_index,
                }
            })

        chunk_index += 1
        start += chunk_size - chunk_overlap

    return chunks



from openai import OpenAI
client = OpenAI()

def embed_chunk(text: str) -> list[float]:
    """
    מקבל טקסט ומחזיר embedding vector.
    """

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    return response.data[0].embedding



import chromadb
from typing import List


chroma_client = chromadb.PersistentClient(path="./chroma_db")

collection = chroma_client.get_or_create_collection(
    name="rag_documents"
)


def ingest_documents_to_vector_store(file_paths: List[str]) -> None:
    """
    טוען מסמכים, מחלק ל-chunks, יוצר embeddings ושומר ב-ChromaDB.
    """

    for file_path in file_paths:
        chunks = load_and_chunk_document(file_path, 1200, 5)

        for chunk in chunks:
            text = chunk["text"]
            metadata = chunk["metadata"]

            embedding = embed_chunk(text)

            chunk_id = f"{metadata['file_name']}_{metadata['chunk_index']}"

            collection.add(
                ids=[chunk_id],
                documents=[text],
                embeddings=[embedding],
                metadatas=[metadata]
            )

    print("Ingestion completed successfully")
    
    

def retrieve_relevant_chunks(
    question: str,
    top_k: int = 4,
) -> list[dict]:
    """
    מקבלת שאלה, יוצרת לה embedding,
    מחפשת ב-vector store,
    ומחזירה chunks רלוונטיים.
    """

    question_embedding = embed_chunk(question)

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    chunks = []

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances,
    ):
        chunks.append({
            "text": document,
            "metadata": metadata,
            "distance": distance,
        })

    return chunks