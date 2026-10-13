# Ingestão e Busca Semântica com LangChain e Postgres

## Objetivo

Você deve entregar um software capaz de:

- Ingestão: Ler um arquivo PDF e salvar suas informações em um banco de dados PostgreSQL com extensão pgVector.
- Busca: Permitir que o usuário faça perguntas via linha de comando (CLI) e receba respostas baseadas apenas no conteúdo do PDF.

## Exemplo no CLI

Faça sua pergunta:

```
PERGUNTA: Qual o faturamento da Empresa SuperTechIABrazil?
RESPOSTA: O faturamento foi de 10 milhões de reais.

---

Perguntas fora do contexto:

PERGUNTA: Quantos clientes temos em 2024?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.
```

## Tecnologias obrigatórias

- Linguagem: Python
- Framework: LangChain
- Banco de dados: PostgreSQL + pgVector
- Execução do banco de dados: Docker & Docker Compose (docker-compose fornecido no repositório de exemplo)

## Pacotes recomendados

- Split: `from langchain_text_splitters import RecursiveCharacterTextSplitter`
- Embeddings (OpenAI): `from langchain_openai import OpenAIEmbeddings`
- Embeddings (Gemini): `from langchain_google_genai import GoogleGenerativeAIEmbeddings`
- PDF: `from langchain_community.document_loaders import PyPDFLoader`
- Ingestão: `from langchain_postgres import PGVector`
- Busca: `similarity_search_with_score(query, k=10)`

## OpenAI

- Crie uma API Key da OpenAI.
- Você vai precisar de um modelo de embeddings e de um modelo de LLM para responder. Consulte a documentação oficial da OpenAI para ver os modelos disponíveis.

## Gemini

- Crie uma API Key da Google.
- Você vai precisar de um modelo de embeddings e de um modelo de LLM para responder. Consulte a documentação oficial do Google para ver os modelos disponíveis.

Os limites de requisições gratuitas dos modelos podem mudar com frequência. Para informações atualizadas, consulte a documentação oficial do Google.

## Escolha dos modelos

Este desafio não fixa modelos. Nomes e versões mudam com frequência e alguns são descontinuados, então faz parte do desafio consultar a documentação oficial do provedor que você escolher, ver quais modelos estão disponíveis no momento e selecionar os que atendem ao objetivo. Para o volume deste desafio, os modelos mais leves e baratos de cada provedor são suficientes.

Atenção: modelos de embedding diferentes geram vetores com dimensões diferentes. A tabela de vetores é criada na primeira ingestão, já com a dimensão do modelo que você escolheu. Se você trocar de modelo de embeddings depois disso, a ingestão passa a falhar por incompatibilidade de dimensão. Nesse caso é responsabilidade sua apagar a collection existente (ou o volume do banco) e refazer a ingestão do zero com o novo modelo.

## Requisitos

### 1. Ingestão do PDF

- O PDF deve ser dividido em chunks de 1000 caracteres com overlap de 150.
- Cada chunk deve ser convertido em embedding.
- Os vetores devem ser armazenados no banco de dados PostgreSQL com pgVector.

### 2. Consulta via CLI

Criar um script Python para simular um chat no terminal.

Passos ao receber uma pergunta:

- Vetorizar a pergunta.
- Buscar os 10 resultados mais relevantes (k=10) no banco vetorial.
- Montar o prompt e chamar a LLM.
- Retornar a resposta ao usuário.

Prompt a ser utilizado:

```
CONTEXTO:
{resultados concatenados do banco de dados}

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
{pergunta do usuário}

RESPONDA A "PERGUNTA DO USUÁRIO"
```

## Estrutura obrigatória do projeto

Faça um fork do repositório para utilizar a estrutura abaixo: https://github.com/devfullcycle/mba-ia-desafio-ingestao-busca

```
├── docker-compose.yml
├── requirements.txt      # Dependências
├── .env.example          # Template das variáveis de ambiente
├── src/
│   ├── ingest.py         # Script de ingestão do PDF
│   ├── search.py         # Script de busca
│   ├── chat.py           # CLI para interação com usuário
├── document.pdf          # PDF para ingestão
└── README.md             # Instruções de execução
```

## VirtualEnv para Python

Crie e ative um ambiente virtual antes de instalar dependências:

```
python3 -m venv venv
source venv/bin/activate
```

## Ordem de execução

1. Subir o banco de dados:

```
docker compose up -d
```

2. Executar ingestão do PDF:

```
python src/ingest.py
```

3. Rodar o chat:

```
python src/chat.py
```

## Implementação deste projeto

Esta implementação acessa o modelo de embeddings e o LLM via um endpoint compatível com a API da OpenAI, configurado pelas variáveis `OPENAI_BASE_URL` e `OPENAI_API_KEY`. Isso permite usar a OpenAI diretamente ou qualquer gateway compatível.

### Modelos utilizados

| Função | Modelo | Observação |
|---|---|---|
| Embeddings | `text-embedding-3-small` | Dimensão 1536 |
| LLM (resposta) | `gpt-4o-mini` | Alternativa leve: `gemini-2.5-flash` |

### Pré-requisitos

- Docker e Docker Compose
- Python 3.12+
- Uma API key válida (OpenAI ou gateway compatível com a API da OpenAI)

### Estrutura dos arquivos

```
src/
├── config.py    # Configurações centralizadas (URL, modelos, conexão, token)
├── ingest.py    # Ingestão do PDF no banco vetorial
├── search.py    # Chain RAG (retriever k=10 + prompt + LLM)
└── chat.py      # CLI interativo de perguntas e respostas
```

### Variáveis de ambiente

Crie um arquivo `.env` (veja `.env.example`) com as variáveis abaixo:

| Variável | Padrão | Descrição |
|---|---|---|
| `OPENAI_API_KEY` | Sim | Chave de acesso ao modelo |
| `OPENAI_BASE_URL` | Não | URL do endpoint (omitir para usar a OpenAI padrão) |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Modelo de embeddings |
| `LLM_MODEL` | `gpt-4o-mini` | Modelo de LLM |
| `PDF_PATH` | `document.pdf` | Caminho do PDF |
| `DB_CONNECTION` | `postgresql+psycopg://postgres:postgres@localhost:5432/rag` | String de conexão |
| `COLLECTION_NAME` | `pdf_collection` | Nome da collection vetorial |

### Ordem de execução

1. Subir o banco de dados:

```
docker compose up -d
```

2. Criar e ativar o ambiente virtual:

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. Configurar as variáveis de ambiente (copie `.env.example` para `.env` e preencha `OPENAI_API_KEY`).

4. Executar ingestão do PDF:

```
python src/ingest.py
```

5. Rodar o chat:

```
python src/chat.py
```

### Observação sobre troca de modelo de embeddings

Modelos de embedding diferentes geram vetores com dimensões diferentes. Se você trocar o `EMBEDDING_MODEL` após a primeira ingestão, é necessário apagar a collection existente (ou o volume do banco) e reingestar:

```
docker compose down -v
docker compose up -d
python src/ingest.py
```

## Entregável

Repositório público no GitHub contendo todo o código-fonte e README com instruções claras de execução do projeto.