def start_conversation(vector_index, vector_store, llm):
    first_question = True
    while True:
        if first_question:
            query_text = input("\nEnter your question (or type 'quit' to exit): ").strip()
        else:
            query_text = input("\nWhat's your next question (or type 'quit' to exit): ").strip()

        if query_text.lower() == "quit":
            break

        if query_text == "":
            continue

        first_question = False

        print("\nQUESTION: \"%s\"" % query_text)
        answer = vector_index.query(query_text, llm=llm).strip()
        print("ANSWER: \"%s\"\n" % answer)

        print("FIRST DOCUMENTS BY RELEVANCE:")
        for doc, score in vector_store.similarity_search_with_score(query_text, k=4):
            print("    [%0.4f] \"%s ...\"" % (score, doc.page_content[:84]))