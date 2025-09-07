import streamlit as st
import sqlite3
import uuid
from datetime import datetime
import requests

# ---------- Database Setup ----------
def init_db():
    conn = sqlite3.connect("chat_app.db")
    c = conn.cursor()
    # Users table
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT UNIQUE
                )''')
    # Chats table
    c.execute('''CREATE TABLE IF NOT EXISTS chats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    session_id TEXT UNIQUE,
                    session_name TEXT,
                    created_at TEXT,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )''')
    conn.commit()
    conn.close()

def get_user(email):
    conn = sqlite3.connect("chat_app.db")
    c = conn.cursor()
    c.execute("SELECT id FROM users WHERE email=?", (email,))
    user = c.fetchone()
    conn.close()
    return user

def create_user(email):
    conn = sqlite3.connect("chat_app.db")
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO users (email) VALUES (?)", (email,))
    conn.commit()
    conn.close()

def create_chat(user_id):
    session_id = str(uuid.uuid4())
    conn = sqlite3.connect("chat_app.db")
    c = conn.cursor()
    c.execute("INSERT INTO chats (user_id, session_id, session_name, created_at) VALUES (?, ?, ?, ?)",
              (user_id, session_id, session_id, datetime.now().isoformat()))
    conn.commit()
    conn.close()
    return session_id

def get_chats(user_id):
    conn = sqlite3.connect("chat_app.db")
    c = conn.cursor()
    c.execute("SELECT id, session_id, session_name, created_at FROM chats WHERE user_id=? ORDER BY created_at DESC", (user_id,))
    chats = c.fetchall()
    conn.close()
    return chats

@st.dialog(title="Rename Chat", on_dismiss="rerun")
def rename_chat(session_id: str):
    """Rename chat session in local DB"""
    new_name = st.text_input("Enter new name?")
    if st.button("Confirm", use_container_width=True) and new_name:
        # Update in local DB
        conn = sqlite3.connect("chat_app.db")
        c = conn.cursor()
        c.execute("UPDATE chats SET session_name=? WHERE session_id=?", (new_name, session_id))
        conn.commit()
        conn.close()
        st.rerun()

@st.dialog(title="Confirmation", on_dismiss="ignore")
def delete_chat(session_id: str):
    """Delete chat: first call backend, then remove locally"""
    st.subheader("Are you sure?")
    if st.button("Delete", use_container_width=True, type="primary"):
        try:
            resp = requests.delete(f"http://localhost:8000/history/{session_id}", timeout=30)
            if resp.status_code != 200:
                st.error(f"Failed to delete chat from backend: {resp.status_code}")
                return False
        except Exception as e:
            st.error(f"Error deleting chat from backend: {e}")
            return False

        # Delete from local DB
        conn = sqlite3.connect("chat_app.db")
        c = conn.cursor()
        c.execute("DELETE FROM chats WHERE session_id=?", (session_id,))
        conn.commit()
        conn.close()
        if st.session_state.get("selected_session") == session_id:
            st.session_state.selected_session = None
        st.rerun()
        return True

def initiateChat(session_id):
    st.session_state.chat_history = []
    if session_id:
        try:
            resp = requests.get(f"http://localhost:8000/history/{session_id}")
            if resp.status_code == 200:
                st.session_state.chat_history = resp.json().get("history", [])
        except Exception as e:
            st.error(f"Error fetching history: {e}")
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        elif msg["role"] == "assistant":
            with st.chat_message("assistant"):
                st.markdown(msg["content"])

    # Input for new message
    if prompt := st.chat_input("Type your message..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        processChat(prompt, session_id)
def processChat(question, session_id):
    st.session_state.chat_history.append({"role": "user", "content": question})
    try:
        resp = requests.post(
            f"http://localhost:8000/chat/{session_id}",
            params={"question": question},
            timeout=300
        )
        # print(resp, resp.json())
        if resp.status_code == 200:
            answer = resp.json().get("response", "")
            with st.chat_message("assistant"):
                st.markdown(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
        else:
            st.error(f"Error: {resp.status_code} - {resp.text}")
    except Exception as e:
        st.error(f"Request failed: {e}")

def sidebar():
    with st.sidebar:
        if st.button("New Chat", use_container_width=True):
            sid = create_chat(st.session_state.user_id)
            st.session_state.selected_session = sid
            st.rerun()   
        chats = get_chats(st.session_state.user_id)
        if chats:
            st.header("Chats")
            for cid, sid, session_name, cdate in chats:
                label = f"{session_name[:10]}"
                if st.session_state.get("selected_session") == sid:
                    button_label = f"👉 {label}"
                else:
                    button_label = label
                chat, delete = st.columns([4,1], gap=None, border=False, vertical_alignment="center")
                with chat:
                    if st.button(button_label, key=f"chat-{sid}", use_container_width=True):
                        st.session_state.selected_session = sid
                        st.rerun()
                with delete:
                    with st.popover("", help="Actions",width="stretch"):
                        if st.button(":material/edit: Rename", key=f"edit-{sid}", use_container_width=True):
                            rename_chat(sid)
                        if st.button(":material/delete: Delete", key=f"delete-{sid}", use_container_width=True):
                            delete_chat(sid)
                        
        else:
            st.info("No previous chats found.")
# ---------- Streamlit App ----------
st.set_page_config(page_title="Chat App", layout="centered")
init_db()

if "user_id" not in st.session_state:
    st.session_state.user_id = None

if st.session_state.user_id is None:
    st.title("Login / Sign Up")
    email = st.text_input("Enter your email")
    if st.button("Submit") and email:
        create_user(email)
        user = get_user(email)
        if user:
            st.session_state.user_id = user[0]
            st.rerun()
else:
    sidebar()
    if "selected_session" not in st.session_state:
        st.warning("Select or start a new chat from the sidebar.")
    else:
        initiateChat(st.session_state.selected_session)