"""CLI para interação com o usuário (chat RAG no terminal).

Recebe perguntas via linha de comando, busca o contexto no banco vetorial e
exibe a resposta do LLM. Digite 'sair' para encerrar.

Uso:
    python src/chat.py
"""

from search import search_prompt

COMANDOS_SAIDA = {"sair", "exit", "quit", "/sair", "/exit", "/quit"}


def main():
    print("Inicializando chat RAG...")
    chain = search_prompt()

    if not chain:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return

    print("\n" + "=" * 60)
    print("Chat RAG pronto! Faça sua pergunta (ou digite 'sair' para encerrar).")
    print("=" * 60 + "\n")

    while True:
        try:
            pergunta = input("PERGUNTA: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEncerrando chat. Até mais!")
            break

        if not pergunta:
            continue

        if pergunta.lower() in COMANDOS_SAIDA:
            print("Encerrando chat. Até mais!")
            break

        try:
            resposta = chain.invoke(pergunta)
        except Exception as e:
            print(f"ERRO ao processar a pergunta: {e}")
            print("Verifique se o banco de dados está rodando e se a ingestão foi feita.\n")
            continue

        print(f"RESPOSTA: {resposta}\n")


if __name__ == "__main__":
    main()
