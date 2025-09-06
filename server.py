from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_perplexity import ChatPerplexity
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from sqlalchemy import create_engine

app = FastAPI(title="Q&A Chatbot", version="1.0.0")

load_dotenv()
engine = create_engine("sqlite:///chat_history.db")

# Endpoint to handle chat messages
@app.post("/chat/{session_id}", tags=["Chat"])
def chat(question: str, session_id: str):
    chain = create_chain()
    response = process_chat(chain, question, session_id)
    return JSONResponse(content=response, status_code=200)


# Retruve all chat history for a session
@app.get("/history/{session_id}", tags=["Chat"])
def get_history(session_id: str):
    history = create_db(session_id)
    messages = history.messages
    formatted_history = [{"role": "user" if isinstance(msg, HumanMessage) else "assistant", "content": msg.content} for msg in messages]
    return JSONResponse(content={"history": formatted_history}, status_code=200)

# Clear chat history for a session
@app.delete("/history/{session_id}", tags=["Chat"])
def clear_history(session_id: str):
    history = create_db(session_id)
    history.clear()
    return JSONResponse(content={"message": "History cleared"}, status_code=200)

def create_chain():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Your a question answering bot. Use very little words to answer the question. Dont use citations unless asked."),
        MessagesPlaceholder(variable_name="history"),
        ("user", "{question}")
    ])

    llm = ChatPerplexity(
        model="sonar",
        temperature=0.2
    )

    # parser = StrOutputParser()

    chain = prompt | llm

    chain_with_history = RunnableWithMessageHistory(
        chain,
        create_db,
        input_messages_key="question",
        history_messages_key="history",
    )

    return chain_with_history
def create_db(session_id: str):
    return SQLChatMessageHistory(
        session_id=session_id, connection=engine
    )

def process_chat(chain, question, session_id: str="default_session"):
    response = chain.invoke(
        {"question": question},
        config={"configurable": {"session_id": session_id}}
    )

    # Convert into a clean dict
    response_dict = {
        "answer": response.content,
        "citations": response.additional_kwargs.get("citations", []),
        "search_results": response.additional_kwargs.get("search_results", []),
        "metadata": response.response_metadata,
        "usage": response.usage_metadata,
    }
    return response_dict

if __name__ == "__main__":
    chain = create_chain()

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        response = process_chat(chain, user_input)
        print("Assistant:", response['answer'])