from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta", 
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
)


llm = ChatHuggingFace(llm=llm)


pdf_path = r'E:\pandas\MLOPS\RAG Mlflow\data\3 laws.pdf'

def load_process_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    for page in pages:
        page.page_content = page.page_content.replace('\n', ' ')

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(pages)
    return chunks

prompt = PromptTemplate(
        input_variables=["context", "input"],
        template=(
            "You are a helpful assistant. Use ONLY the following context to answer.\n\n"
            "Context:\n{context}\n\n"
            "Question: {input}"
        ),
    )

chunks = load_process_pdf(pdf_path)

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":2})

combine_docs_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, combine_docs_chain)


# *********** Q&A ***********
while True:
    user_input  = input("Enter your question: ")
    if user_input == 'q':
        break

    response = rag_chain.invoke({"input": user_input})
    print(f'User:- {user_input}')
    print(f'AI:- {response}')