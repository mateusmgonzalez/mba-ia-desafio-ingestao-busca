"""Script de ingestão do PDF no banco vetorial (PostgreSQL + pgvector).

Lê o PDF, divide em chunks de 1000 caracteres com overlap de 150, gera
embeddings e armazena os vetores no PostgreSQL com pgvector.

Uso:
    python src/ingest.py
"""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


def _resolve_pdf_path() -> Path:
    """Resolve o caminho do PDF relativo à raiz do projeto."""
    path = Path(config.PDF_PATH)
    if not path.is_absolute():
        path = config.BASE_DIR / path
    return path


def ingest_pdf():
    if not config.check_credentials():
        return

    pdf_path = _resolve_pdf_path()
    if not pdf_path.exists():
        print(f"ERRO: PDF não encontrado em {pdf_path}")
        return

    print(f"Lendo PDF: {pdf_path}")
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()
    print(f"  {len(pages)} página(s) carregada(s).")

    print(
        f"Dividindo em chunks (tamanho={config.CHUNK_SIZE}, "
        f"overlap={config.CHUNK_OVERLAP})..."
    )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(pages)
    print(f"  {len(chunks)} chunk(s) gerado(s).")

    print("Conectando ao banco vetorial e gerando embeddings...")
    print(f"  Modelo de embedding: {config.EMBEDDING_MODEL}")
    print(f"  Collection: {config.COLLECTION_NAME}")
    embeddings = config.get_embeddings()
    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name=config.COLLECTION_NAME,
        connection=config.DB_CONNECTION,
        use_jsonb=True,
    )

    vectorstore.add_documents(chunks)
    print(f"Ingestão concluída! {len(chunks)} chunk(s) armazenado(s) no banco.")


if __name__ == "__main__":
    ingest_pdf()
