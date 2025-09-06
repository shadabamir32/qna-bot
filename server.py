from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_perplexity import ChatPerplexity
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

app = FastAPI(title="Q&A Chatbot", version="1.0.0")

load_dotenv()
@app.get("/chat")
def chat(query: str):
    # Placeholder for chat logic
    return {"response": f"Received query: {query}"}

def create_chain():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a question, provide a concise and accurate answer based on reliable sources in 50 words or less. If the question is not answerable, respond with 'I don't know'."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])

    llm = ChatPerplexity(
        model="sonar",
        temperature=0.5
    )

    parser = StrOutputParser()

    chain = prompt | llm | parser

    return chain

def process_chat(chain, question, chat_history):
    response = chain.invoke({
        "chat_history": chat_history,
        "input": question,
    })
    return response



if __name__ == "__main__":
    chain = create_chain()

    chat_history = []

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        response = process_chat(chain, user_input, chat_history)
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))
        print("Assistant:", response)