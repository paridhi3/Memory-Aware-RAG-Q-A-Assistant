import streamlit as st
import faiss
import numpy as np
import time
from sentence_transformers import SentenceTransformer
from diskcache import Cache
from openai import AzureOpenAI
from dotenv import load_dotenv
import os

# ----------------------------
# 🔧 Load environment variables
# ----------------------------
load_dotenv()
API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
MODEL_NAME = os.getenv("MODEL_NAME")
API_VERSION = os.getenv("API_VERSION")

# ----------------------------
# 🔧 Azure OpenAI client setup
# ----------------------------
client = AzureOpenAI(
    api_key=API_KEY,
    api_version=API_VERSION,
    azure_endpoint=ENDPOINT
)

# ----------------------------
# 🔧 Embedding + FAISS setup
# ----------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embedding_dim = 384
faiss_index = faiss.IndexFlatIP(embedding_dim)
semantic_cache = []
disk_cache = Cache("./llm_cache")

# ----------------------------
# 🔧 Response generation (with history)
# ----------------------------
def generate_response_with_history(prompt: str) -> str:
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for msg in st.session_state.chat_history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=150,
        temperature=0.7
    )
    return response.choices[0].message.content

# ----------------------------
# 🔧 Cached GPT call (history-aware key + memory vs cache status)
# ----------------------------
def cached_gpt_call(prompt: str, threshold=0.80) -> tuple[str, str, float]:
    start = time.time()

    # Create a cache key that includes history + current prompt
    cache_key = str(st.session_state.chat_history) + prompt

    # Semantic cache check
    query_embedding = embedding_model.encode(
        prompt, normalize_embeddings=True, convert_to_numpy=True
    ).astype("float32").reshape(1, -1)

    if len(semantic_cache) > 0:
        D, I = faiss_index.search(query_embedding, k=1)
        if D[0][0] > threshold:
            elapsed = time.time() - start
            return semantic_cache[I[0][0]], f"Cache Hit ✅ (Semantic)", elapsed

    # Disk cache check (history-aware)
    if cache_key in disk_cache:
        elapsed = time.time() - start
        return disk_cache[cache_key], "Cache Hit ✅ (Disk)", elapsed

    # Generate new response (via memory/history)
    response = generate_response_with_history(prompt)
    faiss_index.add(query_embedding)
    semantic_cache.append(response)
    disk_cache[cache_key] = response
    elapsed = time.time() - start
    return response, "Memory 🧠 (History Used)", elapsed

# ----------------------------
# 🔧 Streamlit Chat UI
# ----------------------------
st.set_page_config(page_title="Semantic Cache Chatbot", layout="centered")
st.title("💬 Semantic Cache Chatbot")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Reset button
if st.button("🔄 Reset Chat"):
    st.session_state.chat_history = []
    st.rerun()

# Chat input
user_input = st.chat_input("Ask me anything:")

if user_input:
    response, status, elapsed = cached_gpt_call(user_input)

    # Save to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": f"**Source:** {status}\n\n**Time Taken:** {elapsed:.2f} sec\n\n{response}"
    })

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
