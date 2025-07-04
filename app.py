import streamlit as st
from dotenv import load_dotenv
import os
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory

# 📦 Load environment variables
load_dotenv()

# 1. Load and split document
loader = PyPDFLoader("rabbits.pdf")
pages = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=70)
splits = splitter.split_documents(pages)

# 2. Embeddings
embedding = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-ada-002",
    openai_api_version=os.getenv("API_VERSION"),
    azure_endpoint="https://gen-cim-eas-dep-genai-train-openai.openai.azure.com/",
    api_key=os.getenv("AZURE_OPENAI_API_KEY")
)

# 3. Vector store
persist_dir = "docs/chroma"
vectordb = Chroma.from_documents(splits, embedding=embedding, persist_directory=persist_dir)

# 4. Custom prompt
template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" 
at the end of the answer. 
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# 5. LLM
llm = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_END_POINT"],
    azure_deployment=os.environ["DEPLOYMENT_NAME"],
    openai_api_version=os.environ["API_VERSION"],
)

# 6. Memory (with Streamlit session state)
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

# 7. Conversational Retrieval Chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=vectordb.as_retriever(),
    memory=st.session_state.memory,
    combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT}
)

# 8. Streamlit UI
st.set_page_config(page_title="Rabbit Q&A Bot", layout="centered")
st.title("🐰 Ask Me Anything About Rabbits!")

query = st.text_input("Type your question:")

if query:
    result = qa_chain({"question": query})
    st.markdown("### 🤖 Answer")
    st.write(result["answer"])

    with st.expander("📚 Sources used"):
        for doc in result["source_documents"]:
            st.write(doc.page_content[:300] + "...")

# Show chat history
with st.expander("💬 Chat History"):
    for msg in st.session_state.memory.chat_memory.messages:
        st.write(f"**{msg.type.capitalize()}**: {msg.content}")
