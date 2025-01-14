from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.utilities import ArxivAPIWrapper,WikipediaAPIWrapper,DuckDuckGoSearchAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchResults
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents import  create_tool_calling_agent,initialize_agent,AgentType
from langchain.tools.retriever import create_retriever_tool
import streamlit as st
import os


arxiv_api_wrapper = ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=300)
arxiv = ArxivQueryRun(api_wrapper=arxiv_api_wrapper)

wiki_api_wrapper = WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=300)
wiki = WikipediaQueryRun(api_wrapper=wiki_api_wrapper)

search = DuckDuckGoSearchResults(name="Search")

st.set_page_config(page_title="Langchain: Search Engine",page_icon="🦜")
st.title('🦜Langchain: Search Engine')
"""
In the example we're using streamlitcallbackhandler to display the throughts and actions of an agent in
an interactive streamlit app. Try more LangChain Streamlit Agent example at github.com/langchain-ai/streamlit-agent

"""


# api key of groq
st.sidebar.title("Setting")
api_key = st.sidebar.text_input("Provide your Groq api key",type='password')


if "messages" not in st.session_state:
    st.session_state['messages']=[
        {'role':'assistant','content':'Hi i am chat bot how can i assist you'}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

if prompt:=st.chat_input(placeholder='Ask anything whatever you want'):
    st.session_state.messages.append({'role':'user','content':prompt})
    st.chat_message(msg['role']).write(prompt)

    llm = ChatGroq(model='Gemma2-9b-It',api_key=api_key,streaming=True)

    tools = [search,arxiv,wiki]

    agent = initialize_agent(llm=llm,tools=tools,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handle_parsing_error=True)

    with st.chat_message('assistant'):
        st_call = StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
        response = agent.run(st.session_state.messages,callbacks=[st_call])
        st.session_state.messages.append({'role':'assistant','content':response})
        st.success(response)





        