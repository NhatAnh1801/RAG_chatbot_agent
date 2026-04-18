import streamlit as st
import re
import time

from src.rag_engine import RagController
# --- 1. App Configuration ---
st.set_page_config(
    page_title="Legal AI Assistant",
    page_icon="⚖️",
    layout="wide",
)

# --- Define the Knowledge Hierarchy ---
# This dictionary maps the jurisdiction to its specific legal domains.
LEGAL_HIERARCHY = {
    "Vietnam": [
        "AI Law", 
        "Labor Law", 
        "Cybersecurity Law",
    ],
    # "United States": [
    #     "Civil Procedure"
    # ]
}

# --- 2. Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your Legal AI Assistant. Please select your jurisdiction and legal domain to begin."}
    ]

t0 = time.perf_counter()
if "rag_controller" not in st.session_state:
    st.session_state.rag_controller = RagController()
    
    with st.spinner("📚 Loading server... This may take a moment."):
        st.session_state.rag_controller.ingest_legal_docs()
t1 = time.perf_counter()
print(f"-> [App Init]: RAG Controller initialized and documents ingested in {t1 - t0: .4f} seconds")

# --- 3. Sidebar: Database Info & Controls ---
with st.sidebar:
    st.header("⚖️ Legal Database Info")
    
    # 1st Dropdown: Jurisdiction Selector
    selected_jurisdiction = st.selectbox(
        "🌐 Select Jurisdiction",
        options=list(LEGAL_HIERARCHY.keys()),
        index=0,
    )
    
    # 2nd Dropdown: Domain Selector (Updates dynamically based on the 1st dropdown)
    selected_domain = st.selectbox(
        "📚 Select Legal Domain",
        options=LEGAL_HIERARCHY[selected_jurisdiction],
        index=0,
        help="Narrows the search down to a specific field of law."
    )
    
    # Store selections in session state
    st.session_state.jurisdiction = selected_jurisdiction
    st.session_state.domain = selected_domain

    if (st.session_state.get("current_jurisdiction") != selected_jurisdiction or st.session_state.get("current_domain") != selected_domain):
        st.session_state.agent = st.session_state.rag_controller.build_legal_agent(selected_jurisdiction, selected_domain)
        
        st.session_state.current_jurisdiction = selected_jurisdiction
        st.session_state.current_domain = selected_domain
    
    st.info(
        f"**Targeting:**\n"
        f"{selected_domain}\n"
        f"from {selected_jurisdiction} law\n\n"
        "The assistant will filter the database to match these criteria."
    )
    
    st.divider()
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": f"Ready to answer questions regarding {st.session_state.jurisdiction} - {st.session_state.domain}."}
        ]
        st.rerun()

# --- 4. Main Chat Interface ---
st.title("🏛️ Legal Counsel RAG")

chat_container = st.container()

with chat_container:
    for msg in st.session_state.messages[-50:]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
# --- 5. Chat Input & Processing ---
question = st.chat_input(f"Ask a question about {selected_domain}...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with chat_container:
        with st.chat_message("user"):
            st.markdown(question)

    with chat_container:
        with st.chat_message("assistant"):
            with st.spinner(f"Thinking..."):
                history = st.session_state.messages[1:]
                response = None
                sources = []
                try:
                    agent_response = st.session_state.rag_controller.ask(
                        agent = st.session_state.agent,
                        question=question,
                        history=history
                    )
                    response = agent_response["answer"]
                    sources = agent_response["sources"]
                except Exception as e:
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                        match = re.search(r"retry in (\d+(?:\.\d+)?)s", str(e), re.IGNORECASE)
                        wait = int(float(match.group(1))) if match else None
                        response = f"⚠️ API rate limit reached. Please wait a moment and try again after {wait}s."
                    else:
                        response = f"An error occurred: {str(e)}"
                    
                st.markdown(response)
                if sources:
                    st.divider()
                    st.caption("📎Reference")
                    for src in sources:
                        with st.expander(
                            f"📄 {src['source']} — {src['parent_title']}"
                        ):
                            st.text(src.get("content", "Empty"))

    st.session_state.messages.append({"role": "assistant", "content": response})
