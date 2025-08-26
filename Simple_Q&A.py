import os
import traceback
import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint

# ✅ Set your Hugging Face API token (replace with your actual token)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "YOUR-API-KEY"

# Correct model and task
llm = HuggingFaceEndpoint(
    repo_id="google/flan-t5-base",           # Must support 'text2text-generation'
    task="text2text-generation",
    model_kwargs={"generate_kwargs": {"max_new_tokens": 100}}
)


# Streamlit UI
st.title("🤖 Chat with FLAN-T5 (HuggingFace)")

user_input = st.text_input("Ask a question:")

if user_input:
    with st.spinner("Thinking..."):
        try:
            response = llm.invoke(user_input)
            st.success(response)
        except Exception:
            st.error("⚠️ An error occurred:")
            st.code(traceback.format_exc(), language="python")
