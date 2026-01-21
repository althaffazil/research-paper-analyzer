import streamlit as st
import os
from engine.ingestion import process_pdf
from engine.hybrid_engine import ResearchEngine
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title="Research Paper Analyzer", layout="wide", page_icon="🧪")


# Initialize the engine once
@st.cache_resource
def load_engine():
    return ResearchEngine()


engine = load_engine()


# --- State Management Helper ---
def reset_chat_state():
    st.session_state.chat_history = []
    if "retriever" in st.session_state:
        # Optional: You could also delete the temp.pdf here
        pass


# --- Initialize Session State ---
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("🧪 Advanced Research Paper Analyzer")

with st.sidebar:
    st.header("Step 1: Upload")
    pdf_file = st.file_uploader("Upload Research Paper (PDF)", type="pdf")

    # Logic: Automatic delete on new indexing
    if st.button("Index & Analyze") and pdf_file:
        # Clear old data before processing new file
        reset_chat_state()

        with st.spinner("Building Hybrid Index..."):
            if not os.path.exists("data"): os.makedirs("data")

            with open("temp.pdf", "wb") as f:
                f.write(pdf_file.getbuffer())

            chunks = process_pdf("temp.pdf")
            st.session_state.retriever = engine.get_hybrid_retriever(chunks)
            st.success("New Paper Indexed! History cleared.")

    st.divider()

    # Logic: Manual Clear Button
    if st.session_state.retriever:
        if st.button("🗑️ Clear Chat History"):
            reset_chat_state()
            st.rerun()  # Refresh UI to show empty chat

# --- Step 2: Chat UI ---
if st.session_state.retriever:
    # Render History
    for msg in st.session_state.chat_history:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)

    if prompt := st.chat_input("Ask about methodology, results, or conclusions..."):
        st.chat_message("user").markdown(prompt)

        # Build the RAG chain with current history
        chain = engine.create_rag_chain(st.session_state.retriever)

        with st.chat_message("assistant"):
            with st.spinner("Reviewing Paper..."):
                response = chain.invoke({
                    "input": prompt,
                    "chat_history": st.session_state.chat_history
                })
                answer = response["answer"]
                st.markdown(answer)

                with st.expander("Source Evidence"):
                    for doc in response["context"]:
                        pg = doc.metadata.get('page', '?')
                        st.caption(f"**Page {pg}**: {doc.page_content[:200]}...")

        # Update History
        st.session_state.chat_history.extend([
            HumanMessage(content=prompt),
            AIMessage(content=answer)
        ])
else:
    st.info("👋 Welcome! Please upload a PDF in the sidebar to begin your analysis.")