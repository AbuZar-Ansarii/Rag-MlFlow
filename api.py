import shutil
import os
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
load_dotenv()
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
import uuid


# --- Configurations ---
PERSIST_DIRECTORY = "./api_vector_db"
os.makedirs("uploads", exist_ok=True)


# --- Helper Functions ---

def process_pdf_to_chunks(file_path: str):
    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load()

        for page in pages:
            page.page_content = " ".join(page.page_content.split()) 
            page.metadata.update({
                "subject": file_path,
                "doc_id": uuid.uuid4()
            })

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=80,
            separators=["\n\n", "\n", ".", " ", ""]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")
    return text_splitter.split_documents(pages)

# Global variables to store loaded models
resources = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load heavy models once when the API starts."""
    print("🚀 Loading Models and Embeddings...")
    
    # 1. Load Embeddings
    resources["embeddings"] = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    
    # 2. Load LLM
    llm_engine = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta", 
        task="text-generation",
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
    )
    resources["llm"] = ChatHuggingFace(llm=llm_engine)

    resources["vectorstore"] = None
    
    yield   
    # Clean up on shutdown if needed
    resources.clear()


app = FastAPI(title="RAG with MLflow API", lifespan=lifespan)


# --- API Endpoints ---

@app.get("/")
def read_root():
    return {"message": "Welcome to the RAG API"}

@app.post('/uploadfile/')
async def upload_file(file: UploadFile = File(...)):
    
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_path = f"uploads/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Process and save to Vector DB
    chunks = process_pdf_to_chunks(file_path)
    
    if resources["vectorstore"] is not None:
        vectorstore = resources["vectorstore"]
        vectorstore.add_documents(chunks)
    else:
        vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=resources["embeddings"]
        )

    vectorstore.save_local(PERSIST_DIRECTORY)

    return {
        "filename": file.filename, 
        "status": "Vector DB Updated", 
        "chunks": len(chunks),
        "document_id": str(uuid.uuid4())
    }

@app.post('/query')
async def query_rag(query: str):
    if not os.path.exists(PERSIST_DIRECTORY):
        raise HTTPException(status_code=404, detail="Vector database not found. Please upload a file first.")

    # 1. Load Vectorstore (Reuse global embeddings)
    
    vectorstore = FAISS.load_local(
        PERSIST_DIRECTORY, 
        resources["embeddings"],
        allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 2. Setup Prompt
    prompt = PromptTemplate(
        input_variables=["context", "input"],
        template=(
            "You are a helpful assistant. Answer ONLY from the context provided.\n"
            "If the answer is not in the context, say 'Not found in the syllabus.'\n\n"
            "Context: {context}\n\n"
            "Question: {input}"
        )
    )

    # 3. Create and Invoke Chain
    combine_docs_chain = create_stuff_documents_chain(resources["llm"], prompt)
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

    result = await rag_chain.ainvoke({"input": query})

    return {
        "question": query,
        "answer": result['answer'],
        "sources": [doc.metadata for doc in result['context']],
        "resources_used": list(resources.keys())
    }