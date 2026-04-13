from __future__ import annotations

import os

import requests
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError


def resolve_api_base_url() -> str:
    """Resolve the backend URL without requiring a Streamlit secrets file."""
    default_url = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")
    try:
        return st.secrets.get("api_base_url", default_url)
    except StreamlitSecretNotFoundError:
        return default_url


API_BASE_URL = resolve_api_base_url()

st.set_page_config(page_title="Codebase Q&A Assistant", layout="wide")
st.title("Codebase Q&A Assistant")
st.caption("Ingest a GitHub repository, then ask questions about the codebase.")

with st.sidebar:
    st.subheader("Repository")
    repo_url = st.text_input(
        "GitHub URL",
        placeholder="https://github.com/owner/repository",
    )
    branch = st.text_input("Branch", placeholder="Leave empty to use the default branch")
    force_refresh = st.checkbox("Force refresh local clone", value=False)

    if st.button("Ingest Repository", use_container_width=True):
        if not repo_url:
            st.error("Enter a GitHub repository URL first.")
        else:
            payload = {
                "repo_url": repo_url,
                "branch": branch or None,
                "force_refresh": force_refresh,
            }
            with st.spinner("Indexing repository..."):
                response = requests.post(f"{API_BASE_URL}/repos/ingest", json=payload, timeout=300)
            if response.ok:
                body = response.json()
                st.session_state["repo_id"] = body["repo_id"]
                st.success(f"Indexed {body['repo_id']} with {body['total_chunks']} chunks.")
            else:
                st.error(response.text)

st.subheader("Ask Questions")
question = st.text_area(
    "Question",
    placeholder="Where is authentication handled?",
    height=120,
)
top_k = st.slider("Sources to return", min_value=3, max_value=12, value=6)

if st.button("Ask", type="primary", use_container_width=True):
    if not question:
        st.error("Enter a question first.")
    elif not repo_url and "repo_id" not in st.session_state:
        st.error("Ingest a repository or provide a repo URL first.")
    else:
        payload = {
            "question": question,
            "repo_id": st.session_state.get("repo_id"),
            "repo_url": repo_url or None,
            "top_k": top_k,
        }
        with st.spinner("Thinking over the codebase..."):
            response = requests.post(f"{API_BASE_URL}/qa/ask", json=payload, timeout=180)
        if response.ok:
            body = response.json()
            st.markdown(body["answer"])
            st.subheader("Sources")
            for source in body["sources"]:
                symbol = f"::{source['symbol_name']}" if source.get("symbol_name") else ""
                lines = ""
                if source.get("start_line") and source.get("end_line"):
                    lines = f" ({source['start_line']}-{source['end_line']})"
                st.code(f"{source['file_path']}{symbol}{lines}", language="text")
        else:
            st.error(response.text)
