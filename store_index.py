"""Populate the Pinecone index from the PDFs in Data/.

Idempotent: creates the index only if it doesn't already exist. Safe to
re-run, though re-running will re-upsert (and duplicate, since no stable
IDs are set) chunks if the index is already populated -- run once per fresh
index unless you clear it first.
"""

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from src.config import get_settings
from src.helper import download_hugging_face_embeddings, load_pdf_file, text_split


def main() -> None:
    settings = get_settings()

    print(f"Loading PDFs from {settings.data_dir} ...")
    documents = load_pdf_file(settings.data_dir)
    print(f"Loaded {len(documents)} document page(s).")

    text_chunks = text_split(documents)
    print(f"Split into {len(text_chunks)} chunk(s).")

    embeddings = download_hugging_face_embeddings(settings.embedding_model)

    pc = Pinecone(api_key=settings.pinecone_api_key)
    existing_indexes = {idx["name"] for idx in pc.list_indexes()}

    if settings.pinecone_index_name not in existing_indexes:
        print(f"Creating Pinecone index '{settings.pinecone_index_name}' ...")
        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=settings.embedding_dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=settings.pinecone_cloud, region=settings.pinecone_region
            ),
        )
    else:
        print(f"Index '{settings.pinecone_index_name}' already exists, reusing it.")

    print("Embedding chunks and upserting into Pinecone (this may take a while)...")
    PineconeVectorStore.from_documents(
        documents=text_chunks,
        index_name=settings.pinecone_index_name,
        embedding=embeddings,
    )
    print(f"Done. Upserted {len(text_chunks)} chunks into '{settings.pinecone_index_name}'.")


if __name__ == "__main__":
    main()
