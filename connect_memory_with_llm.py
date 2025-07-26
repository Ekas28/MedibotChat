import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")


# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Hugging Face Model
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# Initialize HuggingFace Inference Client for conversational API
hf_client = InferenceClient(model=HUGGINGFACE_REPO_ID, token=HF_TOKEN)

# Load FAISS DB
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Chat history for conversational context
chat_history = []

# Function to query HuggingFace conversational model
def query_mistral(prompt, history):
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for user_msg, bot_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})
    messages.append({"role": "user", "content": prompt})

    response = hf_client.chat_completion(model=HUGGINGFACE_REPO_ID, messages=messages)
    return response.choices[0].message["content"]


# Main loop
while True:
    user_query = input("\nAsk your question (or type 'exit' to quit): ")
    if user_query.lower() == "exit":
        break

    # Retrieve context from FAISS
    docs = db.similarity_search(user_query, k=3)
    context_text = "\n".join([doc.page_content for doc in docs])

    # Create prompt
    custom_prompt = f"""
You are a medical assistant. Only answer questions strictly related to the provided context.
If the question is unrelated to medical topics or outside the context, reply with:
"I'm sorry, I can only answer medical-related questions."
Context: {context_text}
Question: {user_query}


Answer:
"""
    # Get response from Mistral
    bot_response = query_mistral(custom_prompt, chat_history)

    print("\nRESULT:", bot_response)
    print("\nSOURCE DOCUMENTS:")
    for i, doc in enumerate(docs, start=1):
        print(f"{i}. {doc.page_content[:300]}...\n")

    # Update chat history
    chat_history.append((user_query, bot_response))
