from json import load
import os
import logging
from typing import List, Optional
from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class RAGConfig:
    """Configuration constants for the RAG pipeline."""
    MODEL_ID = "HuggingFaceH4/zephyr-7b-beta"
    EMBEDDING_MODEL = "BAAI/bge-m3"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 100
    VECTOR_DB_PATH = "vectorstore/db_faiss"
    PDF_PATH = r'E:\pandas\MLOPS\RAG Mlflow\data\3 laws.pdf'



class RAGPipeline:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
        self.llm = self._initialize_llm()
        self.vectorstore = None
        self.rag_chain = None

    def _initialize_llm(self):
        """Initializes the LLM with a chat wrapper."""
        endpoint = HuggingFaceEndpoint(
            repo_id=self.config.MODEL_ID,
            task="text-generation",
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=1.03,
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
        )
        return ChatHuggingFace(llm=endpoint)

    def load_and_split_docs(self, pdf_path: str):
        """Loads PDF and returns document chunks."""
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        
        # Clean text
        for page in pages:
            page.page_content = page.page_content.replace('\n', ' ')

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )
        return text_splitter.split_documents(pages)

    def build_vector_store(self, force_reload: bool = False):
        """Creates or loads a FAISS vector store."""
        if not force_reload and os.path.exists(self.config.VECTOR_DB_PATH):
            logger.info("Loading existing vector store...")
            self.vectorstore = FAISS.load_local(
                self.config.VECTOR_DB_PATH, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
        else:
            logger.info("Generating new vector store...")
            chunks = self.load_and_split_docs(self.config.PDF_PATH)
            self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
            self.vectorstore.save_local(self.config.VECTOR_DB_PATH)
        
        self._build_chain()

    def _build_chain(self):
        """Constructs the retrieval chain."""
        prompt = ChatPromptTemplate.from_template("""
            You are a professional assistant. Use only the provided context to answer the question.
            If the answer is not in the context, say you don't know.
            
            Context: {context}
            Question: {input}
            Answer:
        """)

        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        combine_docs_chain = create_stuff_documents_chain(self.llm, prompt)
        self.rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

    def answer(self, question: str) -> str:
        """Invokes the chain to get a response."""
        if not self.rag_chain:
            raise ValueError("RAG Chain not initialized. Run build_vector_store() first.")
        
        response = self.rag_chain.invoke({"input": question})
        return response["answer"]

# --- Local Testing Block ---
if __name__ == "__main__":
    cfg = RAGConfig()
    rag = RAGPipeline(cfg)
    rag.build_vector_store()
    
    while True:
        query = input("\nAsk a question (or 'q' to quit): ")
        if query.lower() == 'q': break
        print(f"AI: {rag.answer(query)}")