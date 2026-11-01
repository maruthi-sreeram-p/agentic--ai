"""
Exercise 5 - Multi-document Campus RAG

Loads every .txt file in docs/ (academic handbook, exam rules, department brochure),
embeds the chunks into one Chroma store, retrieves for a student query and
answers from the retrieved text. The source snippets are printed under the answer.
"""
from pathlib import Path

from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

DOCS_DIR = Path(__file__).parent / "docs"
TOP_K = 3


class LocalEmbeddings(Embeddings):
    """Small MiniLM model that ships with chromadb, runs locally, no extra API key."""

    def __init__(self):
        self.model = DefaultEmbeddingFunction()

    def embed_documents(self, texts):
        return [[float(x) for x in vec] for vec in self.model(texts)]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def build_vector_store():
    docs = []
    for path in sorted(DOCS_DIR.glob("*.txt")):
        docs.append(Document(page_content=path.read_text(encoding="utf-8"), metadata={"source": path.name}))

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"Loaded {len(docs)} documents -> {len(chunks)} chunks")

    return Chroma.from_documents(chunks, LocalEmbeddings())


prompt = PromptTemplate.from_template(
    """You are a helpful assistant for college students.
Answer the question using only the context below. Mention the source file name in brackets,
like [exam_rules.txt], for the facts you use. If the answer is not in the context, say you don't know.

Context:
{context}

Question: {question}

Answer:"""
)
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
chain = prompt | llm | StrOutputParser()


def answer_question(store, question):
    results = store.similarity_search(question, k=TOP_K)

    context = "\n\n".join(f"[{doc.metadata['source']}]\n{doc.page_content}" for doc in results)
    answer = chain.invoke({"context": context, "question": question})

    print("\nAnswer:", answer)
    print("\nReference snippets:")
    for i, doc in enumerate(results, 1):
        snippet = doc.page_content.replace("\n", " ")[:200]
        print(f"  {i}. ({doc.metadata['source']}) {snippet}...")


def main():
    store = build_vector_store()
    print("Ask about academics, exams or the department (type 'exit' to quit)\n")
    while True:
        question = input("Student: ").strip()
        if question.lower() == "exit":
            break
        if not question:
            continue
        try:
            answer_question(store, question)
        except Exception as e:
            print(f"Something went wrong: {e}")
        print()


if __name__ == "__main__":
    main()
