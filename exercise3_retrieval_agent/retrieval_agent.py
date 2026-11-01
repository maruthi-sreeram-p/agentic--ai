"""
Exercise 3 - Simple Retrieval Agent with LangChain

campus_rules.txt -> split into chunks -> embed -> Chroma vector store
question -> retrieve the most relevant chunks -> LLM answers from those chunks
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

load_dotenv()  # ChatGroq picks up GROQ_API_KEY from the environment

DOC_PATH = Path(__file__).parent / "campus_rules.txt"
TOP_K = 2


class LocalEmbeddings(Embeddings):
    """Small MiniLM model that ships with chromadb, runs locally, no extra API key."""

    def __init__(self):
        self.model = DefaultEmbeddingFunction()

    def embed_documents(self, texts):
        return [[float(x) for x in vec] for vec in self.model(texts)]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def build_vector_store():
    text = DOC_PATH.read_text(encoding="utf-8")
    docs = [Document(page_content=text, metadata={"source": DOC_PATH.name})]

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"Loaded {DOC_PATH.name}: {len(chunks)} chunks")

    return Chroma.from_documents(chunks, LocalEmbeddings())


prompt = PromptTemplate.from_template(
    """You are a helpful assistant for college students.
Answer the question using only the context below. If the answer is not in the context, say you don't know.

Context:
{context}

Question: {question}

Answer:"""
)
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
chain = prompt | llm | StrOutputParser()


def answer_question(store, question):
    results = store.similarity_search_with_score(question, k=TOP_K)

    print("\nRetrieved chunks (lower score = closer match):")
    for doc, score in results:
        preview = doc.page_content.replace("\n", " ")[:150]
        print(f"  [{score:.3f}] {preview}...")

    context = "\n\n".join(doc.page_content for doc, _ in results)
    return chain.invoke({"context": context, "question": question})


def main():
    store = build_vector_store()
    print("Ask about the campus rules (type 'exit' to quit)\n")
    while True:
        question = input("Student: ").strip()
        if question.lower() == "exit":
            break
        if not question:
            continue
        try:
            print("\nAnswer:", answer_question(store, question))
        except Exception as e:
            print(f"Something went wrong: {e}")
        print()


if __name__ == "__main__":
    main()
