from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from .prompts import QA_PROMPT, RECONTEXT_PROMPT


class ResearchEngine:
    def __init__(self):
        # Temperature 0 for factual research accuracy
        self.llm = ChatOllama(model="llama3.1", temperature=0)
        self.embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    def get_hybrid_retriever(self, chunks):
        # 1. Semantic Search (meaning)
        vectorstore = Chroma.from_documents(chunks, self.embeddings)
        v_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # 2. Keyword Search (technical terms)
        bm25_retriever = BM25Retriever.from_documents(chunks)
        bm25_retriever.k = 3

        # 3. Hybrid (70% weight to semantics)
        return EnsembleRetriever(
            retrievers=[v_retriever, bm25_retriever],
            weights=[0.7, 0.3]
        )

    def create_rag_chain(self, retriever):
        # Makes the chain "remember" what you asked previously
        history_aware_retriever = create_history_aware_retriever(
            self.llm, retriever, RECONTEXT_PROMPT
        )
        # Combines retrieved docs into the prompt
        combine_docs_chain = create_stuff_documents_chain(self.llm, QA_PROMPT)

        return create_retrieval_chain(history_aware_retriever, combine_docs_chain)