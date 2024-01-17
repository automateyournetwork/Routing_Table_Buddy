import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import JSONLoader
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Function to run pyATS job
def run_pyats_job():
    os.system("pyats run job show_ip_route_langchain_job.py")

# Use Streamlit's caching to run the job only once
if 'job_done' not in st.session_state:
    st.session_state['job_done'] = st.cache_resource(run_pyats_job)()

# Instantiate openAI client
load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

class ChatWithRoutingTable:
    def __init__(self):
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
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
        )
        self.docs = self.text_splitter.split_documents(self.pages)

    def store_in_chroma(self):
        embeddings = OpenAIEmbeddings()
        self.vectordb = Chroma.from_documents(self.docs, embedding=embeddings)
        self.vectordb.persist()

    def setup_conversation_memory(self):
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    def setup_conversation_retrieval_chain(self):
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

# Create an instance of your class
chat_instance = ChatWithRoutingTable()

# Streamlit UI for chat
st.title("Chat with Cisco IOS XE Routing Table")

# Initialize conversation history in session state if not present
if 'conversation_history' not in st.session_state:
    st.session_state['conversation_history'] = ""

user_input = st.text_input("Ask a question about the routing table:", key="user_input")

if st.button("Ask"):
    with st.spinner('Processing...'):
        # Call the chat method and get the AI's response
        ai_response = chat_instance.chat(user_input)
        # Display the conversation history
        st.text_area("Conversation History:", value=st.session_state['conversation_history'], height=300, key="conversation_history_display")
