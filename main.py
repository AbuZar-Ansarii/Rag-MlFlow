from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, SystemMessage
import os
load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta", 
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
)


chat_model = ChatHuggingFace(llm=llm)

while True:
    user_input  = input("Enter your question: ")
    if user_input == 'q':
        break

    messages = [
            SystemMessage(content="You are a helpful assistant that explains complex topics simply."),
            HumanMessage(content=user_input),
        ]

    response = chat_model.invoke(messages)
    print(f'User:- {user_input}')
    print(f'AI:- {response.content}')
