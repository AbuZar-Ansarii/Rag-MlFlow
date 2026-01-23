from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma, FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
load_dotenv()

PERSIST_DIRECTORY = "./faiss_db"
pdf_path = r'E:\pandas\MLOPS\RAG Mlflow\data\ML_Algorithms_Interview_Handbook.pdf'

llm_engine = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-Guard-3-1B-INT4", 
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
)


llm = ChatHuggingFace(llm=llm_engine)


def load_process_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    for page in pages:
        page.page_content = page.page_content.replace('\n', ' ')
        page.metadata = {"source": pdf_path,
            "subject": "49 law of power",
            "doc_id": "syllabus_001" }

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(pages)
    return chunks

chunks = load_process_pdf(pdf_path)

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

vectorstore = FAISS.from_documents(
    documents=chunks,
        embedding=embeddings
)

vectorstore.save_local(PERSIST_DIRECTORY)