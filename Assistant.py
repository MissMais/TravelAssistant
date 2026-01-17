import os
from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    groq_api_key=API_KEY,
    model="llama-3.1-8b-instant"
)

search = DuckDuckGoSearchAPIWrapper()

@tool
def web_search_tool(query: str) -> str:
    """
    Use this tool when you need to search for recent travel information or places
    
    Args:
        query: The query to search for
    Returns:
        The search result

    """
    print(f"Searching for {query}")
    return search.run(query)

tools = [web_search_tool]

agent = create_agent(model=llm, tools=tools)

def travel_assistant(user_query: str):
    decision_prompt = f"""
    You are a travel assistant.
    If the question needs up-to-date or factual information, respond ONLY with:
    SEARCH

    Otherwise respond ONLY with:
    ANSWER

    Question: {user_query}
    """

    decision = llm.invoke(decision_prompt).content.strip()

    if decision == "SEARCH":
        search_result = web_search_tool.run(user_query)

        final_prompt = f"""
        Use the information below to answer clearly and concisely:

        {search_result}
        """
        return llm.invoke(final_prompt).content

    else:
        return llm.invoke(user_query).content



# ------- Streamlit UI --------

st.set_page_config(page_title="Travel Assistant", layout="centered")
st.title("✈️ Travel Assistant")
st.subheader("Plan your journeys with real-time travel info...")

# Initializing chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Displaying previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Where do you want to travel?")

if prompt:
    # Show user message
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        response = travel_assistant(prompt)
        st.markdown(response)

    # Save assistant message
    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )

