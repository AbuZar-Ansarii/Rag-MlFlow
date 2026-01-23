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

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta", 
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
)


llm = ChatHuggingFace(llm=llm)


embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
vectorstore = FAISS.load_local(PERSIST_DIRECTORY, embeddings,allow_dangerous_deserialization=True)


retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":2})


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

    # FIX 2: Retrieval chain returns a dict, we want the "answer" key
    result = rag_chain.invoke({"input": user_input})
    
    print(f"\nAI: {result['answer']}") 
    print(f"\nSource: {result['context'][0].metadata}")