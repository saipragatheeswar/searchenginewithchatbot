import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchResults
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv

#Tools
# Initialize the API wrappers
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wikipedia_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)

# Tools
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)
wikipedia = WikipediaQueryRun(api_wrapper=wikipedia_wrapper)
search = DuckDuckGoSearchResults(name = "search")

st.title("Search Engine with chatbot")
# Sidebar for settings
groq_api = os.getenv("groq_api")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hello! I am a search engine with a chatbot. How can I help you?"}
    ]

for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="Ask me anything...."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    llm=ChatGroq(groq_api_key=groq_api or "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", model_name="Llama3-8b-8192", streaming=True)
    tools=[search, arxiv, wikipedia]

    search_results = initialize_agent(llm = llm, tools = tools, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handling_parsing_errors=True)

    with st.chat_message("assistant"):
        try:
          response = search_results.run(st.session_state.messages)
        except ValueError as e:
            st.write(f"An error occurred: {e}")
            response = "Sorry, I encountered an issue processing your request."
        
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
# Display the current state of session_state.messages for debugging purposes
st.write("Current st.session_state.messages:", st.session_state["messages"])