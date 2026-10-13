"""Módulo de busca: monta a chain RAG (retriever + prompt + LLM).

A função search_prompt() retorna um Runnable que recebe a pergunta do usuário
(string) e retorna a resposta do LLM (string) baseada apenas no conteúdo do PDF.
"""

import config
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_postgres import PGVector

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""


def _build_vectorstore() -> PGVector:
    embeddings = config.get_embeddings()
    return PGVector(
        embeddings=embeddings,
        collection_name=config.COLLECTION_NAME,
        connection=config.DB_CONNECTION,
        use_jsonb=True,
    )


def _retrieve_context(vectorstore: PGVector, question: str) -> str:
    """Busca os k chunks mais relevantes e concatena como contexto."""
    results = vectorstore.similarity_search_with_score(question, k=config.SEARCH_K)
    if not results:
        return ""
    return "\n\n".join(doc.page_content for doc, _ in results)


def search_prompt(question=None):
    """Cria e retorna a chain RAG.

    Retorna None se as credenciais não estiverem configuradas.
    O parâmetro `question` existe apenas para compatibilidade de assinatura e
    não é usado aqui — a pergunta real é passada em runtime via chain.invoke().
    """
    if not config.check_credentials():
        return None

    vectorstore = _build_vectorstore()
    llm = config.get_llm()
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

    chain = (
        {
            "contexto": RunnableLambda(lambda q: _retrieve_context(vectorstore, q)),
            "pergunta": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain
