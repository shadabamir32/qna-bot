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
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import bs4
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain

app = FastAPI(title="Q&A Chatbot", version="1.0.0")

load_dotenv()
engine = create_engine("sqlite:///chat_history.db")


# Endpoint to handle chat messages
@app.post("/chat/{session_id}", tags=["Chat"])
def chat(question: str, session_id: str):
    vectorStore = retrive_document_from_web()
    chain = create_chain(vectorStore)

    response = process_chat(chain, question, session_id)
    return JSONResponse(content=response['answer'], status_code=200)


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

# Create the chat chain with history
def create_chain(vectorStore):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the context: {context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])

    llm = ChatPerplexity(
        model="sonar",
        temperature=0.1
    )

    combine_docs_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt
    )
    
    retriever = vectorStore.as_retriever(search_kwargs={"k": 3})

    # History-aware retriever prompt
    retriever_prompt = ChatPromptTemplate.from_messages([
        ("system", "Based on the conversation, rewrite the user query to improve retrieval."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=retriever,
        prompt=retriever_prompt
    )

    retrieval_chain = create_retrieval_chain(
        history_aware_retriever,
        combine_docs_chain
    )

    chain_with_history = RunnableWithMessageHistory(
        retrieval_chain,
        create_db,
        output_messages_key="answer",
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    return chain_with_history

# Initialize or retrieve the database for chat history
def create_db(session_id: str):
    return SQLChatMessageHistory(
        session_id=session_id, connection=engine
    )

# Process chat input and generate response
def process_chat(chain, question, session_id: str="default"):
    return chain.invoke(
        {"input": question},
        config={"configurable": {"session_id": session_id}}
    )

#RAG
def retrive_document_from_web():
    # Load the web page
    loader = WebBaseLoader(
        web_path=(
            "https://www.barandbench.com/news/litigation/delhi-court-allows-engineer-rashid-to-attend-parliament-to-vote-in-vice-president-election",
        )
    )
    data = loader.load()

    if not data:
        raise ValueError(f"No data found at {url}")
    # Split the document into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    splitedDocument = text_splitter.split_documents(data)
    if not splitedDocument:
        raise ValueError("No text chunks created from document")
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorStore = FAISS.from_documents(splitedDocument, embeddings)

    return vectorStore

vectorStore=None
if __name__ == "__main__":
    vectorStore = retrive_document_from_web()
    chain = create_chain(vectorStore)

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        response = process_chat(chain, user_input)
        # print(type(response))
        print(response['answer'])

# def stream_chat(chain, question, session_id: str="default_session"):
#     def event_generator():
#         response_stream = chain.stream(
#             {"question": question},
#             config={"configurable": {"session_id": session_id}}
#         )
#         for chunk in response_stream:
#             yield f"data: {chunk.content}\n\n"
#     return StreamingResponse(event_generator(), media_type="text/event-stream") 