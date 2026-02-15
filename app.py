import os
import shutil
import streamlit as st
import tempfile

from src.rag_engine import RagController

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
)

# --- Session: Store user state ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Upload a document and ask me anything about it."}
    ]

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
    
if "deleted_files" not in st.session_state:
    st.session_state.deleted_files = set()
    
if "rag_controller" not in st.session_state:
        st.session_state.rag_controller = RagController()

with st.sidebar:
    st.header("Upload document")
    
    uploaded = st.file_uploader(
        "Upload your documents",
        type=["pdf"],
        accept_multiple_files=True,
        key="file_uploader",
    )
    
    if uploaded:
        for file in uploaded:
            # Avoid re-processing files already stored
            if file.name in st.session_state.deleted_files:
                continue
            existing_names = [f["name"] for f in st.session_state.uploaded_files]
            if file.name not in existing_names:
                file_data = {
                    "name": file.name,
                    "type": file.type,
                    "size": file.size
                }
                st.session_state.uploaded_files.append(file_data)
                # TODO
                try:
                    with st.spinner('Importing and processing file...'):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_data["name"])[1]) as tmp_file:
                            file.seek(0)
                            shutil.copyfileobj(file, tmp_file)
                            tmp_file_path = tmp_file.name
                            print(f'file is saved in {tmp_file_path}')
                            
                        # Backend Logic
                        try:
                            st.session_state.rag_controller.index_data(tmp_file_path)
                        finally:
                            if os.path.exists(tmp_file_path):
                                os.remove(tmp_file_path)
                        
                except Exception as e:
                    st.error(f'Error processing file: {e}') 
                
                
    if st.session_state.uploaded_files:
        st.divider()  # draws a horizontal line
        st.subheader("Uploaded Files")
        for i, file in enumerate(st.session_state.uploaded_files):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.text(f"📎 {file['name']}")
            with col2:
                if st.button("🗑️", key=f"del_{i}"):
                    st.session_state.deleted_files.add(file["name"])
                    st.session_state.uploaded_files.pop(i)
                    st.rerun()  
    else:
        st.info("No documents uploaded yet.")
        
    st.divider()
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! Upload a document and ask me anything about it."}
        ]
        st.rerun()
        
st.title("RAG Chatbot")

chat_container = st.container()

with chat_container:
    for msg in st.session_state.messages[-50:]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            
# --- Chat input ---
prompt = st.chat_input("Ask a question about your documents...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with chat_container:
        with st.chat_message("user"):
            st.markdown(prompt)

    with chat_container:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                history = st.session_state.messages[1:]
                response = st.session_state.rag_controller.ask(prompt, history)
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()
