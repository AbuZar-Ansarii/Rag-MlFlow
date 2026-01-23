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
import mlflow
import time

PERSIST_DIRECTORY = "./faiss_db"

CONFIG = {
    'chunk_size': 1000,
    'overlap': 100,
    'embedding_model': "BAAI/bge-m3",
    'llm_repo_id': "meta-llama/Llama-3.2-3B-Instruct",
    'k' : 3
}

mlflow.set_experiment("RAG_Experiment2")
mlflow.langchain.autolog()

def run_rag_chain():
    llm = HuggingFaceEndpoint(
        repo_id=CONFIG['llm_repo_id'], 
        task="text-generation",
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
    )

    llm = ChatHuggingFace(llm=llm)

    embeddings = HuggingFaceEmbeddings(model_name=CONFIG['embedding_model'])
    vectorstore = FAISS.load_local(PERSIST_DIRECTORY, embeddings,allow_dangerous_deserialization=True)

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":CONFIG['k']})

    prompt = PromptTemplate(
            input_variables=["context", "input"],
            template=(
                '''You are an exam assistant. Answer ONLY from the given context.If the answer is not in the context, say:
                    "Not found in the syllabus.".\n\n'''
                "Context:\n{context}\n\n"
                "Question: {input}"
            )
        )

    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

    # Chat Loop
    print("Assistant ready! Type 'q' to exit.")
    while True:
        user_input = input("\nEnter your question: ")
        if user_input.lower() == 'q':
            break

        with mlflow.start_run(run_name="RAG_Chat_session"):
            mlflow.log_param("Configuration",CONFIG)
            start_time = time.time()
            result = rag_chain.invoke({"input": user_input})
            latency = time.time() - start_time

            retrival_score = 1.0 if len(result['context'])>0 else 0.0
            response_length = len(result['answer'])

            mlflow.log_metric("latency", latency)
            mlflow.log_metric("retrival_score", retrival_score)
            mlflow.log_metric("response_length", response_length)
        
            print(f"\nAI: {result['answer']}") 
            print(f"\nSource: {result['context'][0].metadata}")
            print(f'\nLogged to mlflow: Latency={latency:.2f}')

if __name__ == "__main__":
    run_rag_chain()
