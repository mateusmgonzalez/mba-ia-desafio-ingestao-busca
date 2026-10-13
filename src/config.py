"""Configurações centralizadas do projeto.

Lê variáveis de ambiente (de um arquivo .env) para configurar o acesso ao
modelo de embeddings e ao LLM. O projeto usa um endpoint compatível com a
API da OpenAI, definido pelas variáveis OPENAI_BASE_URL e OPENAI_API_KEY.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Carrega .env se existir.
load_dotenv()

# Diretório base do projeto (pasta que contém docker-compose.yml e src/).
BASE_DIR = Path(__file__).resolve().parent.parent

# --- Acesso ao modelo (endpoint compatível com a OpenAI API) ---

API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
BASE_URL = os.getenv("OPENAI_BASE_URL", "").strip()

# --- Modelos ---

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# --- Ingestão ---

PDF_PATH = os.getenv("PDF_PATH", "document.pdf")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# --- Banco de dados (PostgreSQL + pgvector) ---

DB_CONNECTION = os.getenv(
    "DB_CONNECTION",
    "postgresql+psycopg://postgres:postgres@localhost:5432/rag",
)
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "pdf_collection")

# Quantos chunks recuperar por busca (k do similarity_search).
SEARCH_K = 10


def _model_kwargs():
    """Monta os kwargs opcionais de autenticação/endpoint para os modelos."""
    kwargs = {}
    if API_KEY:
        kwargs["api_key"] = API_KEY
    if BASE_URL:
        kwargs["base_url"] = BASE_URL
    return kwargs


def get_embeddings():
    """Cria o modelo de embeddings."""
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=EMBEDDING_MODEL, **_model_kwargs())


def get_llm():
    """Cria o LLM de chat."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=LLM_MODEL, temperature=0, **_model_kwargs())


def check_credentials() -> bool:
    """Verifica se as credenciais estão configuradas. Retorna False se faltar a API key."""
    if not API_KEY:
        print("ERRO: OPENAI_API_KEY não encontrada.")
        print(
            "Defina OPENAI_API_KEY (e OPENAI_BASE_URL se usar um gateway próprio) "
            "no arquivo .env. Veja .env.example."
        )
        return False
    return True
