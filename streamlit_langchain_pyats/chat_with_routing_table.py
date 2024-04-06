import os
import requests
import streamlit as st
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import JSONLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceInstructEmbeddings 

@st.cache_resource
def load_model():
    with st.spinner("Downloading Instructor XL Embeddings Model locally....please be patient"):
        embedding_model=HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large", model_kwargs={"device": "cuda"})
    return embedding_model

class ChatWithRoutingTable:
    def __init__(self):
        self.embedding_model = load_model()
        self.conversation_history = []
        self.load_text()
        self.split_into_chunks()
        self.store_in_chroma()
        self.setup_conversation_memory()
        self.setup_conversation_retrieval_chain()

    def load_text(self):
        self.loader = JSONLoader(
            file_path='Show_IP_Route.json',
            jq_schema=".info[]",
            text_content=False
        )
        self.pages = self.loader.load_and_split()

    def split_into_chunks(self):
        # Create a text splitter
        self.text_splitter = SemanticChunker(self.embedding_model)
        self.docs = self.text_splitter.split_documents(self.pages)

    def store_in_chroma(self):
        embeddings = self.embedding_model
        self.vectordb = Chroma.from_documents(self.docs, embedding=embeddings)
        self.vectordb.persist()

    def setup_conversation_memory(self):
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    def setup_conversation_retrieval_chain(self):
        st.write("Starting Conversational Retrieval Chain")
        llm = Ollama(model=st.session_state['selected_model'], base_url="http://ollama:11434")
        self.qa = ConversationalRetrievalChain.from_llm(llm, self.vectordb.as_retriever(search_kwargs={"k": 10}), memory=self.memory)

    def chat(self, question):
        # Format the user's prompt and add it to the conversation history
        user_prompt = f"User: {question}"
        self.conversation_history.append({"text": user_prompt, "sender": "user"})

        # Format the entire conversation history for context, excluding the current prompt
        conversation_context = self.format_conversation_history(include_current=False)

        # Concatenate the current question with conversation context
        combined_input = f"Context: {conversation_context}\nQuestion: {question}"

        # Generate a response using the ConversationalRetrievalChain
        response = self.qa.invoke(combined_input)

        # Extract the answer from the response
        answer = response.get('answer', 'No answer found.')

        # Format the AI's response
        ai_response = f"Cisco IOS XE: {answer}"
        self.conversation_history.append({"text": ai_response, "sender": "bot"})

        # Update the Streamlit session state by appending new history with both user prompt and AI response
        st.session_state['conversation_history'] += f"\n{user_prompt}\n{ai_response}"

        # Return the formatted AI response for immediate display
        return ai_response


    def format_conversation_history(self, include_current=True):
        formatted_history = ""
        history_to_format = self.conversation_history[:-1] if not include_current else self.conversation_history
        for msg in history_to_format:
            speaker = "You: " if msg["sender"] == "user" else "Bot: "
            formatted_history += f"{speaker}{msg['text']}\n"
        return formatted_history

def get_ollama_models(base_url):
    try:       
        response = requests.get(f"{base_url}api/tags")  # Corrected endpoint
        response.raise_for_status()
        models_data = response.json()
        
        # Extract just the model names for the dropdown
        models = [model['name'] for model in models_data.get('models', [])]
        return models
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to get models from Ollama: {e}")
        return []

# Function to run pyATS job
def run_pyats_job():
    os.system("pyats run job show_ip_route_langchain_job.py")

def page_pyats_job():
    # Placeholder for running the pyATS job and model selection
    if st.button("Run pyATS Job"):
        # Run the job and update session state
        run_pyats_job()
        st.session_state['pyats_job_run'] = True
        st.success("pyATS Job Completed Successfully!")
    
    # Model selection dropdown
    try:
        models = get_ollama_models("http://ollama:11434/")
        if models:
            selected_model = st.selectbox("Select Model", models)
            st.session_state['selected_model'] = selected_model
        else:
            st.markdown('No models available. Please visit [localhost:3002](http://localhost:3002) to download models.')
    except Exception as e:
        st.error(f"Failed to fetch models: {str(e)}")
    
    if st.session_state.get('selected_model'):
        if st.button("Proceed to Chat"):
            st.session_state['page'] = 'chat'

def page_chat():
    # Placeholder for the chat interface
    user_input = st.text_input("Ask a question about the routing table:", key="user_input")
    if st.button("Ask"):
        if 'selected_model' in st.session_state and st.session_state['selected_model']:
            chat_instance = ChatWithRoutingTable()  # Initialize with the selected model
            ai_response = chat_instance.chat(user_input)
            st.session_state['conversation_history'] += f"\nUser: {user_input}\nAI: {ai_response}"
            st.text_area("Conversation History:", value=st.session_state['conversation_history'], height=300, key="conversation_history_display")
        else:
            st.error("Please select a model to proceed.")

# Streamlit UI setup
st.title("Routing Table Buddy")

# Initialize session state keys if they are not already set
if 'page' not in st.session_state:
    st.session_state['page'] = 'pyats_job'
if 'conversation_history' not in st.session_state:
    st.session_state['conversation_history'] = ""
if 'pyats_job_run' not in st.session_state:
    st.session_state['pyats_job_run'] = False

# Page routing
if st.session_state['page'] == 'pyats_job':
    page_pyats_job()
elif st.session_state['page'] == 'chat':
    page_chat()