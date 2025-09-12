import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import time

# 🔧 Model setup
MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# 🗄️ Cache setup
dimension = embedder.get_sentence_embedding_dimension()
faiss_index = faiss.IndexFlatL2(dimension)
semantic_cache = []  # stores responses
disk_cache = {}      # dict-based persistent cache


# 🚀 Generate response with history
def generate_response_with_history(prompt, history):
    history_texts = [str(h) for h in history]
    input_text = " ".join(history_texts + [prompt])
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 🔍 Semantic similarity search
def semantic_search(query, threshold=0.7):
    if len(semantic_cache) == 0:
        return None

    query_embedding = embedder.encode([query])
    D, I = faiss_index.search(np.array(query_embedding).astype("float32"), k=1)

    if D[0][0] < (1 - threshold):  # similarity check
        return semantic_cache[I[0][0]]
    return None

# ⚡ Cache + Generation Pipeline
def get_response(prompt):
    start = time.time()

    # Build cache key with history
    cache_key = str(st.session_state.chat_history) + prompt

    # 1. Disk cache
    if cache_key in disk_cache:
        elapsed = time.time() - start
        return disk_cache[cache_key], "Cache Hit ✅ (Disk)", elapsed

    # 2. Semantic cache
    semantic_match = semantic_search(prompt)
    if semantic_match:
        elapsed = time.time() - start
        return semantic_match, "Cache Hit ✅ (Semantic)", elapsed

    # 3. Fresh generation (with or without memory)
    response = generate_response_with_history(prompt, st.session_state.chat_history)

    # Update caches
    query_embedding = embedder.encode([prompt])
    faiss_index.add(np.array([query_embedding]).astype("float32"))
    semantic_cache.append(response)
    disk_cache[cache_key] = response

    elapsed = time.time() - start

    # Decide status
    if len(st.session_state.chat_history) == 0:
        status = "Fresh Generation ✨ (No History)"
    else:
        status = "Memory 🧠 (History Used)"

    return response, status, elapsed


# 🎨 Streamlit UI
st.title("🧠 Caching Demo with Memory vs Cache")

# Init session history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input
prompt = st.text_input("Enter your prompt:")

if st.button("Submit") and prompt:
    response, source, elapsed = get_response(prompt)

    # Update chat history
    st.session_state.chat_history.append(prompt)

    # Display
    st.write("### 🤖 Response:")
    st.success(response)
    st.info(f"Source: {source} | Time: {elapsed:.2f}s")

    # Show history
    st.write("### 📜 Chat History")
    for i, msg in enumerate(st.session_state.chat_history, 1):
        st.write(f"{i}. {msg}")
